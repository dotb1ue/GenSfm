"""
NaN/Inf diagnostics for fp16 training.

Three abstractions:

- ``NaNActivationProbe``: forward-output hooks on a chosen subset of modules.
  Records per-module activation stats (``max_abs``, ``rms``, ``has_nan``,
  ``has_inf``) on device; sync to host is deferred until ``summary()`` /
  ``first_nonfinite()``.

- ``NaNDiagnostic``: top-level wrapper holding one probe per training model,
  rate-limited dump, and output formatting. Use
  ``NaNDiagnostic.maybe_install(...)`` to conditionally install (returns
  ``None`` when disabled → trainer pays zero cost).

- ``make_default_include(max_dots)``: topology-agnostic include filter keyed
  on qualified-name depth.
"""
from typing import Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from .general_utils import unwrap_model


def make_default_include(max_dots: int = 1) -> Callable[[str, nn.Module], bool]:
    """Include filter: hook every ``nn.Module`` whose qualified name has at most
    ``max_dots`` dots.

    - ``0``: root + direct children (e.g. ``input_layer``, ``blocks``).
    - ``1``: above + grandchildren (e.g. ``blocks.0``).
    - ``2``: above + great-grandchildren (e.g. ``blocks.0.attn``).
    """
    def _include(name: str, _m: nn.Module) -> bool:
        if name == '':
            return True
        return name.count('.') <= max_dots
    return _include


class NaNActivationProbe:
    """Forward-output probe for a chosen subset of modules."""

    def __init__(self, model: nn.Module, include: Callable[[str, nn.Module], bool]):
        # name -> (max_abs, rms, has_nan, has_inf); all 0-D GPU tensors.
        self.stats_gpu: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._order: List[str] = []
        self.handles = []
        for name, module in model.named_modules():
            if not include(name, module):
                continue
            self._order.append(name)
            self.handles.append(module.register_forward_hook(self._mk(name)))

    def _mk(self, name):
        def hook(_m, _inp, out):
            if isinstance(out, torch.Tensor):
                self._record(name, out)
            elif isinstance(out, (tuple, list)):
                for i, t in enumerate(out):
                    if isinstance(t, torch.Tensor):
                        self._record(f"{name}#{i}", t)
        return hook

    @torch.no_grad()
    def _record(self, name, t: torch.Tensor):
        if not t.is_floating_point() or t.numel() == 0:
            return
        t = t.detach()
        # rms = sqrt(mean(x^2)); cast to fp32 first to avoid overflow in the
        # square when |x| is near fp16 max.
        t_f = t.float()
        self.stats_gpu[name] = (
            t.abs().amax(),
            t_f.pow(2).mean().sqrt(),
            torch.isnan(t).any(),
            torch.isinf(t).any(),
        )

    def reset(self):
        self.stats_gpu.clear()

    @torch.no_grad()
    def _materialize(self) -> Dict[str, Tuple[float, float, bool, bool]]:
        """Sync all GPU stats to CPU in one pass."""
        return {name: (mx.item(), rms.item(), bool(n.item()), bool(i.item()))
                for name, (mx, rms, n, i) in self.stats_gpu.items()}

    def first_nonfinite(self) -> Optional[Tuple[str, float, float, bool, bool]]:
        """First hooked module (forward order) with NaN/Inf output.
        Returns ``(name, max_abs, rms, has_nan, has_inf)`` or ``None``."""
        stats = self._materialize()
        for name in self._order:
            s = stats.get(name)
            if s is not None and (s[2] or s[3]):
                return (name, s[0], s[1], s[2], s[3])
        return None

    def summary(self, top_k: int = 8) -> str:
        """Formatted top-k activations plus any non-finite modules.

        ``peak/rms`` ≈ 1 when values are spread uniformly; large values
        (>~10) indicate outlier spikes relative to the typical magnitude.
        """
        stats = self._materialize()
        if not stats:
            return "[NaNDiag] (no forward stats)"
        rows = sorted(stats.items(), key=lambda kv: -kv[1][0])
        out = [f"[NaNDiag] top-{top_k} by max|x| (peak/rms shows outlier spikiness):"]
        for name, (m, rms, n, i) in rows[:top_k]:
            pk = (m / rms) if rms > 0 else float('nan')
            flags = (' nan' if n else '') + (' inf' if i else '')
            out.append(f"  {name:70s} max|x|={m:.3e} rms={rms:.3e} "
                       f"peak/rms={pk:6.1f}{flags}")
        bad = [(n, stats[n]) for n in self._order
               if n in stats and (stats[n][2] or stats[n][3])]
        if bad:
            out.append("[NaNDiag] non-finite modules (forward order):")
            for name, (m, rms, nan, inf) in bad:
                fs = ('nan' if nan else '') + (' inf' if inf else ('' if nan else 'inf'))
                out.append(f"  {name:70s} max|x|={m:.3e} rms={rms:.3e} {fs}")
        return "\n".join(out)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


@torch.no_grad()
def first_nonfinite_grads(named_params, k: int = 8):
    """Up to ``k`` params with non-finite grad.

    Returns ``[(name, nan_count, inf_count, numel, max_abs_finite), ...]``.
    Counts are absolute; caller renders as percentages. ``max_abs_finite``
    is the max over elements that are neither NaN nor Inf (via
    ``nan_to_num``), giving a meaningful magnitude even when some elements
    blew up.
    """
    out = []
    for name, p in named_params:
        if p.grad is None:
            continue
        g = p.grad
        nan_count = int(torch.isnan(g).sum().item())
        inf_count = int(torch.isinf(g).sum().item())
        if nan_count == 0 and inf_count == 0:
            continue
        max_finite = g.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).abs().amax().item()
        out.append((name, nan_count, inf_count, g.numel(), max_finite))
        if len(out) >= k:
            break
    return out


class NaNDiagnostic:
    """Top-level wrapper: per-model probes + rate-limited dump. The trainer
    just calls ``reset()`` per step and ``dump(step, tag)`` on NaN events."""

    def __init__(self, training_models, *, depth: int = 1, rate_limit_steps: int = 100):
        include = make_default_include(max_dots=depth)
        self.training_models = training_models
        self.probes: Dict[str, NaNActivationProbe] = {
            name: NaNActivationProbe(unwrap_model(m), include)
            for name, m in training_models.items()
        }
        self._last_dump_step = -(10 ** 9)
        self._rate_limit = rate_limit_steps

    def reset(self):
        for p in self.probes.values():
            p.reset()

    def dump(self, *, step: int, tag: str = ""):
        """Print diagnostics if rate-limit allows. First event always prints."""
        if step - self._last_dump_step < self._rate_limit and self._last_dump_step >= 0:
            return
        self._last_dump_step = step
        print(f"\n[NaNDiag] step={step} {tag}")
        for mname, probe in self.probes.items():
            fn = probe.first_nonfinite()
            if fn is not None:
                name, mx, rms, n, i = fn
                fs = ('nan' if n else '') + (' inf' if i else '')
                print(f"  model[{mname}] first non-finite forward: {name}  "
                      f"max|x|={mx:.3e} rms={rms:.3e} {fs}")
            else:
                print(f"  model[{mname}] forward all finite; probe summary:")
                print(probe.summary(top_k=5))
        for mname, model in self.training_models.items():
            bad = first_nonfinite_grads(
                [(n, p) for n, p in unwrap_model(model).named_parameters()
                 if p.grad is not None],
                k=8,
            )
            if bad:
                print(f"  model[{mname}] non-finite grads (up to 8):")
                for name, nan_c, inf_c, numel, mx_fin in bad:
                    nf = 100.0 * nan_c / max(numel, 1)
                    ii = 100.0 * inf_c / max(numel, 1)
                    print(f"    {name:70s} nan={nf:5.2f}% inf={ii:5.2f}% "
                          f"of {numel}  max|finite|={mx_fin:.3e}")
            else:
                print(f"  model[{mname}] all grads finite")

    def remove(self):
        for p in self.probes.values():
            p.remove()
        self.probes.clear()
