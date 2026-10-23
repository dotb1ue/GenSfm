"""Per-sample debug NPZ + GLB writer for flow-matching evaluators.

Activated by ``save_debug=True`` on ``FlowMatchingEvaluator`` (or its sparse
subclass). At eval time, captures cross-attention weights + intermediate latent
decodes, then writes one ``eval_results/debug/sample_<idx>.npz`` per sample
plus sibling ``mesh.glb``/``gs.ply`` (and optional ``gt_mesh.glb``) under
``debug/sample_<idx>/``.

Self-contained: no edits to ``trellis/modules/attention/*`` or
``trellis/pipelines/samplers/*``. The saver registers PyTorch forward hooks on
cross ``MultiHeadAttention`` modules + parent
``MultiModalModulatedTransformerCrossBlock`` blocks at debug-init time and
monkey-patches ``sampler.sample_once`` to receive a per-step callback.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ...modules.sparse import SparseTensor
from ...utils.align_utils import align_extrinsics_by_similarity


# ---------------------------------------------------------------------------
# Capture state + hook factories
# ---------------------------------------------------------------------------

class _DebugAttnState:
    """Per-saver capture state. Eval is single-threaded per process, so plain
    attributes are safe (no threading.local needed)."""

    def __init__(self):
        self.enabled: bool = False
        self.is_cond_pass: bool = False
        # Set by parent-block pre-hook before each cross-attn call inside a
        # MultiModalModulatedTransformerCrossBlock; consulted by the cross-attn
        # forward hook to derive the 'joint' / 'shape' / 'pose' tag.
        self.current_block_layout: Optional[str] = None
        # id(cross_attn module) -> call idx within the current block. Reset to 0
        # by parent-block pre-hook so per_view layouts can distinguish 1st (shape)
        # from 2nd (pose) cross-attn call on the shared module.
        self.block_call_count: Dict[int, int] = {}
        # tag -> list of [B, L, Lkv] cpu fp16. Aggregated to step_list at each
        # step boundary then cleared.
        self.block_buffer: Dict[str, List[torch.Tensor]] = {}
        # list of dict[tag, Tensor], one per sampler step.
        self.step_list: List[Dict[str, torch.Tensor]] = []
        # 0 = cond pass (first model forward of the step), 1+ = neg-cond pass.
        # CFG mixins in this repo always call cond first, neg-cond second.
        self.pass_idx: int = 0


def _make_block_pre_hook(state: _DebugAttnState):
    """forward_pre_hook on MultiModalModulatedTransformerCrossBlock — stamps
    layout + resets the cross-attn call counter for this block.
    """
    def hook(module, inputs):
        if not state.enabled or not state.is_cond_pass:
            return
        state.current_block_layout = getattr(module, 'cross_attn_layout', None)
        cross_attn = getattr(module, 'cross_attn', None)
        if cross_attn is not None:
            state.block_call_count[id(cross_attn)] = 0
    return hook


def _make_cross_attn_hook(state: _DebugAttnState):
    """forward_hook on a cross MultiHeadAttention. Recomputes attn weights
    from inputs (re-runs to_q / to_kv / softmax under no_grad) and appends
    a mean-over-heads ``[B, L, Lkv]`` fp16 CPU tensor to ``block_buffer[tag]``.

    Tag inference (no edits to multimodal_modulated.py):
      * layout=='full'      → 'joint'
      * layout=='per_view'  → 'shape' on 1st call in block, 'pose' on 2nd
      * layout is None      → unsupported parent (e.g. plain ModulatedTransformerCrossBlock); skip.
    """
    def hook(module, inputs, output):
        if not state.enabled or not state.is_cond_pass:
            return
        layout = state.current_block_layout
        if layout is None:
            return  # not inside a tagged multimodal block
        if len(inputs) < 2:
            return  # not a cross-attn call (no context arg)
        x, context = inputs[0], inputs[1]

        if layout == 'full':
            tag = 'joint'
        elif layout == 'per_view':
            idx = state.block_call_count.get(id(module), 0)
            tag = 'shape' if idx == 0 else 'pose'
            state.block_call_count[id(module)] = idx + 1
        else:
            return  # validator should reject anything else; defensive skip

        with torch.no_grad():
            B, L, _ = x.shape
            Lkv = context.shape[1]
            q = module.to_q(x).reshape(B, L, module.num_heads, -1)
            kv = module.to_kv(context).reshape(B, Lkv, 2, module.num_heads, -1)
            if getattr(module, 'qk_rms_norm', False):
                q = module.q_rms_norm(q)
                k = module.k_rms_norm(kv[..., 0, :, :].contiguous())
            else:
                k = kv[..., 0, :, :]
            scale = 1.0 / math.sqrt(q.size(-1))
            q_h = q.permute(0, 2, 1, 3)   # [B, H, L, C]
            k_h = k.permute(0, 2, 1, 3)
            logits = (q_h @ k_h.transpose(-2, -1)) * scale
            weights = torch.softmax(logits, dim=-1)   # [B, H, L, Lkv]
            rec = weights.mean(dim=1)                 # [B, L, Lkv]
        # Keep leading B dim (writer reshapes); fixes V=1 squeeze bug.
        state.block_buffer.setdefault(tag, []).append(rec.detach().half().cpu())
    return hook


def _make_denoiser_pre_hook(state: _DebugAttnState):
    """forward_pre_hook on the top-level denoiser module — flips
    ``is_cond_pass`` based on per-step call counter. CFG mixins call cond
    first, then neg-cond, so pass_idx==0 is always cond.
    """
    def hook(module, inputs):
        if not state.enabled:
            return
        state.is_cond_pass = (state.pass_idx == 0)
        state.pass_idx += 1
    return hook


# ---------------------------------------------------------------------------
# DebugSaver — owns hooks + writer
# ---------------------------------------------------------------------------

class DebugSaver:
    """Owns capture state, hook handles, and the writer logic.

    One instance per evaluator. Subclass for sparse-evaluator extras
    (alt-intrinsics renders, alignment block, test-view block, mesh.glb).
    """

    SUPPORTED_CLASSES: Tuple[str, ...] = ()
    REQUIRES_ATTN_CAPTURE: bool = True
    PATCH_PARENT_BLOCK: str = 'MultiModalModulatedTransformerCrossBlock'

    def __init__(self, evaluator, *, topk: int = 128):
        self.evaluator = evaluator
        self.topk = int(topk)
        self.state = _DebugAttnState()
        self._hook_handles: List = []
        self._sampler_orig_sample_once: Dict[int, callable] = {}

    # -- validation --------------------------------------------------------

    def validate(self):
        ev = self.evaluator
        cls_name = type(ev).__name__
        if cls_name not in self.SUPPORTED_CLASSES:
            raise NotImplementedError(
                f"save_debug is only supported for {sorted(self.SUPPORTED_CLASSES)}, got {cls_name}"
            )
        if ev.sample_ssuv is not True:
            raise ValueError("save_debug=True requires sample_ssuv=True")
        if not ev.eval_intermediate_steps:
            raise ValueError("save_debug=True requires eval_intermediate_steps to be truthy")

        if not self.REQUIRES_ATTN_CAPTURE:
            return

        denoiser = ev.models['denoiser']
        # Reject torch.compile-wrapped modules.
        if hasattr(denoiser, '_orig_mod'):
            raise RuntimeError(
                "save_debug=True requires a non-compiled denoiser (got torch.compile wrapper); "
                "set denoiser.compile=false in the config."
            )

        for name, m in denoiser.named_modules():
            layout = getattr(m, 'cross_attn_layout', None)
            if layout is not None and layout not in ('full', 'per_view'):
                raise NotImplementedError(
                    f"save_debug=True supports cross_attn_layout in ('full', 'per_view'), "
                    f"got {layout!r} at {name}"
                )

    def detect_cross_attn_layout(self) -> str:
        for m in self.evaluator.models['denoiser'].modules():
            layout = getattr(m, 'cross_attn_layout', None)
            if layout is not None:
                return layout
        return 'full'

    # -- hook setup --------------------------------------------------------

    def setup_hooks(self):
        """Register forward hooks on the denoiser tree + monkey-patch the
        ssuv_sampler's ``sample_once`` to fire per-step callbacks.

        Idempotent: calling twice is fine (existing handles are reused).
        """
        if not self.REQUIRES_ATTN_CAPTURE:
            return  # subclasses that skip capture (e.g. sparse v1) need no hooks

        denoiser = self.evaluator.models['denoiser']
        block_pre_hook = _make_block_pre_hook(self.state)
        cross_attn_hook = _make_cross_attn_hook(self.state)
        denoiser_pre_hook = _make_denoiser_pre_hook(self.state)

        # Top-level denoiser pre-hook for cond/neg distinction.
        self._hook_handles.append(
            denoiser.register_forward_pre_hook(denoiser_pre_hook)
        )

        # Per-block pre-hooks + cross-attn forward hooks.
        from ...modules.attention.modules import MultiHeadAttention
        from ...modules.transformer.multimodal_modulated import (
            MultiModalModulatedTransformerCrossBlock,
        )
        for name, m in denoiser.named_modules():
            if isinstance(m, MultiModalModulatedTransformerCrossBlock):
                self._hook_handles.append(
                    m.register_forward_pre_hook(block_pre_hook)
                )
                # Hook the inner cross_attn (a MultiHeadAttention) directly so
                # hooks fire on EVERY forward of it within the block, not on
                # the outer block's single forward.
                cross = getattr(m, 'cross_attn', None)
                if isinstance(cross, MultiHeadAttention):
                    self._hook_handles.append(
                        cross.register_forward_hook(cross_attn_hook)
                    )

        self._patch_sampler()

    def _patch_sampler(self):
        """Wrap ``sampler.sample_once`` so step_callback fires after each step."""
        sampler = self._get_sampler()
        if sampler is None or id(sampler) in self._sampler_orig_sample_once:
            return
        orig = sampler.sample_once

        def wrapped(*args, **kwargs):
            out = orig(*args, **kwargs)
            self.step_callback()
            return out

        # Bind on the instance (shadows the class method).
        sampler.sample_once = wrapped
        self._sampler_orig_sample_once[id(sampler)] = orig

    def _get_sampler(self):
        return getattr(self.evaluator, 'ssuv_sampler', None)

    def teardown_hooks(self):
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles.clear()

        sampler = self._get_sampler()
        orig = self._sampler_orig_sample_once.pop(id(sampler), None) if sampler is not None else None
        if sampler is not None and orig is not None:
            sampler.sample_once = orig

    # -- per-sample lifecycle ---------------------------------------------

    def begin_sample(self):
        """Reset capture state at the start of each sample's sampling pass."""
        st = self.state
        st.step_list = []
        st.block_buffer = {}
        st.block_call_count = {}
        st.pass_idx = 0
        st.is_cond_pass = True
        st.current_block_layout = None
        st.enabled = True

    def end_sample(self):
        """Disable capture (called even on exceptions to avoid leaks)."""
        self.state.enabled = False
        self.state.current_block_layout = None

    def step_callback(self):
        """End of a sampler step: average each tag's block_buffer → step_list,
        then reset pass counter and per-block call counters for the next step.
        """
        st = self.state
        if st.block_buffer:
            averaged = {tag: torch.stack(lst, dim=0).mean(dim=0)
                        for tag, lst in st.block_buffer.items()}
            st.step_list.append(averaged)
            st.block_buffer = {}
        st.pass_idx = 0
        st.is_cond_pass = True
        st.block_call_count = {}

    def collected_steps(self) -> List[Dict[str, torch.Tensor]]:
        return self.state.step_list

    # -- writer entry points ----------------------------------------------

    def write(self, indices, data, debug_steps, debug_extras: Optional[dict] = None):
        """Build per-sample npz payload and write it. Subclasses override to
        extend the schema (sparse adds alt-intrinsics renders, alignment, NV)."""
        if not debug_steps:
            return  # nothing to write — guards against empty `eval_intermediate_steps` filters
        payload = self.build_payload(indices, data, debug_steps)
        self.save_debug_file(indices, payload)

    # -- payload assembly --------------------------------------------------

    def build_payload(self, indices, data, debug_steps) -> dict:
        """Assemble per-sample debug bundle. B=1 enforced upstream, so leading
        batch dim is stripped. Mirrors the schema the visualizer consumes.
        """
        ev = self.evaluator
        T = len(debug_steps)
        N = data['x_0'].shape[0] if not isinstance(data['x_0'], SparseTensor) else 1
        V = data['P_0'].shape[1]
        # Resolution from the decoded ssuv_recon (always dense), NOT from x_0
        # — sparse x_0 is a SparseTensor whose shape[-1] is feature channels.
        ss_recon0 = debug_steps[0]['preds_i']['ss_recon']
        ssuv_recon0 = debug_steps[0]['preds_i']['ssuv_recon']
        R_shape = int(ss_recon0.shape[-1])
        R_pose = int(data['P_0'].shape[-1])
        L_shape = R_shape ** 3
        L_pose_per_view = R_pose ** 3

        intrinsics = data['intrinsics']
        extrinsics = data['extrinsics']
        intrinsics_flat = intrinsics.reshape(N * V, *intrinsics.shape[2:])
        crop_bbox = data.get('crop_bbox')
        crop_bbox_flat = crop_bbox.reshape(N * V, 2, 2) if crop_bbox is not None else None

        # Per-step decoded reconstructions + per-step PnP-solved poses + alignment.
        ss_recon_s, ssuv_recon_s, uv_vol_s = [], [], []
        ext_pred_s, int_pred_s, ext_aln_s, int_aln_s = [], [], [], []
        int_pred_raw_s = []
        for s in debug_steps:
            p = s['preds_i']
            ss_recon_s.append(p['ss_recon'])
            ssuv_recon_s.append(p['ssuv_recon'])
            uv_vol_s.append(p['uv_volume_pred'])
            ssuv_recon_flat = p['ssuv_recon'].reshape(N * V, *p['ssuv_recon'].shape[2:])
            uv_pred_flat = p['uv_volume_pred'].reshape(N * V, *p['uv_volume_pred'].shape[2:])
            ext_p, int_p, _ = ev._solve_pose_from_ssuv(
                ssuv_recon_flat, uv_pred_flat, intrinsics_gt=intrinsics_flat)
            ext_pred_nv = ext_p.reshape(N, V, 4, 4)
            int_pred_nv = int_p.reshape(N, V, 3, 3)
            ext_pred_s.append(ext_pred_nv)
            int_pred_s.append(int_pred_nv)
            int_pred_raw_s.append(
                ev._adjust_raw_intrinsics(int_p, crop_bbox_flat).reshape(N, V, 3, 3))
            if V < 3:
                ext_aln_s.append(torch.full_like(ext_pred_nv, float('nan')))
                int_aln_s.append(int_pred_nv.clone())
            else:
                align = align_extrinsics_by_similarity(ext_pred_nv, extrinsics)
                ext_aln_s.append(align['extrinsics_pred_aligned'])
                int_aln_s.append(int_pred_nv.clone())

        def _stack0(lst):
            return torch.stack(lst, dim=0)[:, 0].detach().cpu()

        # Metric matrix [T, C].
        step0 = debug_steps[0]
        bases = sorted({k.replace(f"_step{step0['step_i']}", '')
                        for k, v in step0['metrics_i'].items()
                        if isinstance(v, torch.Tensor) and v.ndim == 1})
        metric_matrix = np.zeros((T, len(bases)), dtype=np.float32)
        for ti, s in enumerate(debug_steps):
            for ki, base in enumerate(bases):
                v = s['metrics_i'].get(f"{base}_step{s['step_i']}")
                if isinstance(v, torch.Tensor) and v.numel() > 0:
                    metric_matrix[ti, ki] = float(v[0].item())

        # Cross-attn assembly — index attn_list by step_i so
        # eval_intermediate_steps=[…] subsets work correctly.
        attn_list = self.collected_steps()
        layout = self.detect_cross_attn_layout()
        R_voxel = int(data['ss'].shape[-1]) if 'ss' in data else 64
        image_size = int(data['cond'].shape[-1]) if 'cond' in data else 518
        patch_size = 14
        patch_grid = image_size // patch_size
        meta = {
            'L_shape': L_shape, 'L_pose_per_view': L_pose_per_view,
            'V': V, 'R_shape': R_shape, 'R_pose': R_pose,
            'R_voxel': R_voxel,
            'image_size': image_size, 'patch_size': patch_size,
            'patch_grid': patch_grid,
            'layout': layout, 'format': 'dense',
        }

        if attn_list and self.REQUIRES_ATTN_CAPTURE:
            shape_attn, pose_attn = self._assemble_attn(
                attn_list, debug_steps, layout, T, V, L_shape, L_pose_per_view, meta)
        else:
            meta['Lkv'] = 0
            if self.REQUIRES_ATTN_CAPTURE:
                meta['capture_skipped'] = (
                    f'got {len(attn_list)} steps, expected >= max(step_i)+1')
            shape_attn = torch.empty(0, dtype=torch.float16)
            pose_attn = torch.empty(0, dtype=torch.float16)

        if meta['Lkv'] > 0 and V > 0:
            tokens_per_view = meta['Lkv'] // V
            meta['tokens_per_view'] = tokens_per_view
            meta['num_prefix_tokens'] = max(0, tokens_per_view - patch_grid ** 2)
        else:
            meta['tokens_per_view'] = 0
            meta['num_prefix_tokens'] = 0

        # Optional top-k sparsification. Index dtype branches on Lkv:
        # int16 only fits up to 32767; switch to int32 past that.
        k = self.topk
        if k > 0 and shape_attn.numel() > 0 and k < meta['Lkv']:
            meta['format'] = 'topk'
            meta['k'] = k
            idx_dtype = torch.int16 if meta['Lkv'] < (1 << 15) else torch.int32
            s_vals, s_idx = torch.topk(shape_attn.float(), k=k, dim=-1, largest=True)
            p_vals, p_idx = torch.topk(pose_attn.float(), k=k, dim=-1, largest=True)
            attn_keys = {
                'cross_attn_shape_vals': s_vals.half().numpy(),
                'cross_attn_shape_idx': s_idx.to(idx_dtype).numpy(),
                'cross_attn_pose_vals': p_vals.half().numpy(),
                'cross_attn_pose_idx': p_idx.to(idx_dtype).numpy(),
            }
        else:
            attn_keys = {
                'cross_attn_shape': shape_attn.numpy(),
                'cross_attn_pose': pose_attn.numpy(),
            }

        imgs = data['cond'][0].clamp(0, 1).mul(255.0).to(torch.uint8).cpu().numpy()
        ss_gt = (data['ss'][0, 0] if data['ss'].ndim == 6 else data['ss'][0]).to(torch.uint8).cpu().numpy()
        ssuv_gt = data['ssuv'][0].to(torch.uint8).cpu().numpy()
        uv_vol_gt = data['uv_volume'][0].to(torch.float16).cpu().numpy()

        raw_keys: dict = {}
        if 'raw_cond' in data:
            raw_keys['images_raw'] = (
                data['raw_cond'][0].clamp(0, 1).mul(255.0).to(torch.uint8).cpu().numpy()
            )
        if 'cond_rgba' in data:
            raw_keys['images_cropped_rgba'] = (
                data['cond_rgba'][0].clamp(0, 1).mul(255.0).to(torch.uint8).cpu().numpy()
            )
        if 'raw_cond_rgba' in data:
            raw_keys['images_raw_rgba'] = (
                data['raw_cond_rgba'][0].clamp(0, 1).mul(255.0).to(torch.uint8).cpu().numpy()
            )
        if 'crop_bbox' in data:
            raw_keys['crop_bbox'] = data['crop_bbox'][0].cpu().float().numpy()
        if 'raw_intrinsics' in data:
            raw_keys['intrinsics_raw'] = data['raw_intrinsics'][0].cpu().float().numpy()

        return {
            't_values': np.array([float(s['t_i']) for s in debug_steps], dtype=np.float32),
            'images_cropped': imgs,
            **raw_keys,
            'ss_gt': ss_gt,
            'ssuv_gt': ssuv_gt,
            'uv_volume_gt': uv_vol_gt,
            'ss_recon': _stack0(ss_recon_s).to(torch.uint8).numpy(),
            'ssuv_recon': _stack0(ssuv_recon_s).to(torch.uint8).numpy(),
            'uv_volume_pred': _stack0(uv_vol_s).to(torch.float16).numpy(),
            'extrinsics_gt': extrinsics[0].cpu().float().numpy(),
            'intrinsics_gt': intrinsics[0].cpu().float().numpy(),
            'extrinsics_pred': _stack0(ext_pred_s).float().numpy(),
            'intrinsics_pred': _stack0(int_pred_s).float().numpy(),
            'intrinsics_pred_raw': _stack0(int_pred_raw_s).float().numpy(),
            'extrinsics_aligned': _stack0(ext_aln_s).float().numpy(),
            'intrinsics_aligned': _stack0(int_aln_s).float().numpy(),
            'metrics': metric_matrix,
            'metric_names': np.array(list(bases)),
            'cross_attn_meta': json.dumps(meta),
            **attn_keys,
        }

    def _assemble_attn(self, attn_list, debug_steps, layout, T, V,
                       L_shape, L_pose_per_view, meta):
        """Assemble cross-attn into ``shape_attn [T, L_shape, Lkv]`` and
        ``pose_attn [T, V, L_pose, Lkv]``. Indexes attn_list by step_i so
        filtered ``eval_intermediate_steps`` work."""
        # Pick the entries matching the saved steps; out-of-range step_i
        # falls back to setting capture_skipped.
        try:
            picked = [attn_list[s['step_i']] for s in debug_steps]
        except IndexError:
            meta['Lkv'] = 0
            meta['capture_skipped'] = (
                f'step_i out of range (have {len(attn_list)} steps captured)')
            return (torch.empty(0, dtype=torch.float16),
                    torch.empty(0, dtype=torch.float16))

        if layout == 'full':
            # B dim preserved as 1 — squeeze it now per [T, L, Lkv] / [T, V, L, Lkv] schema.
            attn_stack = torch.stack([s['joint'][0] for s in picked], dim=0)
            expected_L = L_shape + V * L_pose_per_view
            assert attn_stack.shape[1] == expected_L, \
                f"joint attn L={attn_stack.shape[1]} expected={expected_L}"
            meta['Lkv'] = int(attn_stack.shape[2])
            shape_attn = attn_stack[:, :L_shape, :].contiguous().half()
            pose_attn = attn_stack[:, L_shape:, :].reshape(T, V, L_pose_per_view, -1).contiguous().half()
        else:  # per_view
            # shape: [B=1, L_shape, V*Lkv_per_view] → [T, L_shape, V*Lkv_per_view]
            shape_stack = torch.stack([s['shape'][0] for s in picked], dim=0)
            # pose: [B*V, L_pose, Lkv_per_view] → [T, V, L_pose, Lkv_per_view]
            pose_per_step = []
            for s in picked:
                p = s['pose']  # [B*V, L_pose, Lkv_per_view] with B=1
                pose_per_step.append(p.reshape(V, L_pose_per_view, p.shape[-1]))
            pose_stack = torch.stack(pose_per_step, dim=0)
            Lkv_per_view = pose_stack.shape[-1]
            Lkv_full = V * Lkv_per_view
            assert shape_stack.shape[-1] == Lkv_full, \
                f"per_view shape attn Lkv={shape_stack.shape[-1]} expected={Lkv_full}"
            meta['Lkv'] = Lkv_full
            shape_attn = shape_stack.contiguous().half()
            pose_attn = torch.zeros(T, V, L_pose_per_view, Lkv_full, dtype=torch.float16)
            for v in range(V):
                col = v * Lkv_per_view
                pose_attn[:, v, :, col:col + Lkv_per_view] = pose_stack[:, v].half()
        return shape_attn, pose_attn

    # -- file I/O ----------------------------------------------------------

    def save_debug_file(self, indices, payload: dict):
        ev = self.evaluator
        idx_np = indices.cpu().numpy()
        inst_idx, view_idx = ev.get_instance_view_indices(idx_np)
        b = 0
        dataset_index = int(idx_np[b])
        instance_index = int(inst_idx[b])
        viewset_idx = int(view_idx[b])
        inst_row = ev.instances[instance_index]
        if hasattr(inst_row, '__len__') and not isinstance(inst_row, str):
            root = str(inst_row[0])
            sha256 = str(inst_row[1]) if len(inst_row) > 1 else ''
        else:
            root = str(inst_row)
            sha256 = ''
        payload = {**payload,
                   'dataset_index': np.int64(dataset_index),
                   'instance_index': np.int64(instance_index),
                   'viewset_idx': np.int64(viewset_idx),
                   'root': root, 'sha256': sha256}
        out_dir = os.path.join(ev.results_root, 'debug')
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f'sample_{dataset_index}.npz')
        np.savez_compressed(path, **payload)


# ---------------------------------------------------------------------------
# Sparse subclass — adds alt-intrinsics renders, alignment, NV block, GLB/PLY.
# ---------------------------------------------------------------------------

class SparseDebugSaver(DebugSaver):
    SUPPORTED_CLASSES = (
        'VisualImageConditionedMultiPoseSparseFlowMatchingEvaluator',
    )
    REQUIRES_ATTN_CAPTURE = False  # sparse v1: skip cross-attn capture

    @staticmethod
    def _to_uint8_np(t):
        return t.detach().clamp(0, 1).mul(255.0).to(torch.uint8).cpu().numpy()

    @staticmethod
    def _to_f16_np(t):
        return t.detach().to(torch.float16).cpu().numpy()

    @staticmethod
    def _to_f32_np(t):
        return t.detach().cpu().float().numpy()

    def step_payload_to_debug_steps(self, step_payload, indices, data):
        """Decode each captured intermediate latent and compute its metrics,
        producing the debug_steps list shape consumed by ``build_payload``.
        """
        if step_payload is None:
            return None
        ev = self.evaluator
        out = []
        for entry in step_payload:
            step_i, t_i = entry[0], entry[1]
            latents = entry[2:]  # (x0,) for SingleView, (x0, P0) for MultiPose
            preds_i = ev._decode_ssuv(*latents)
            m_i, _ = ev._compute_metrics_from_preds(
                preds_i, data, indices, suffix=f'_step{step_i}')
            out.append({'step_i': int(step_i), 't_i': float(t_i),
                        'preds_i': preds_i, 'metrics_i': m_i})
        return out

    def write(self, indices, data, debug_steps, debug_extras: Optional[dict] = None):
        """Sparse: stage-1 base schema + stage-2 finals + alignment + test-view block.
        Mesh/3DGS go to sibling files under ``debug/sample_<idx>/``.
        """
        debug_extras = debug_extras or {}
        # Stage-1: build from intermediate slat decodes (or skip if absent).
        steps = self.step_payload_to_debug_steps(
            debug_extras.get('ssuv_step_payload'), indices, data)
        if steps:
            payload = self.build_payload(indices, data, steps)
        else:
            payload = {}

        ev = self.evaluator

        # Overwrite cond GT image with the composited-vis tensor used in evaluator.
        cond_gt = debug_extras.get('cond_gt_for_save')
        if cond_gt is not None:
            key = 'images_raw' if ev.metrics_on_raw_cond else 'images_cropped'
            payload[key] = self._to_uint8_np(cond_gt[0])

        # Stage-2 finals.
        if 'cond_render_color_raw' in debug_extras:
            payload['cond_render_color_raw'] = self._to_uint8_np(
                debug_extras['cond_render_color_raw'][0])
            payload['cond_render_color_cropped'] = self._to_uint8_np(
                debug_extras['cond_render_color_cropped'][0])
            payload['cond_intrinsics_raw'] = self._to_f32_np(
                debug_extras['cond_intrinsics_raw'][0])
            payload['cond_intrinsics_cropped'] = self._to_f32_np(
                debug_extras['cond_intrinsics_cropped'][0])
        elif 'cond_render_color' in debug_extras:
            payload['cond_render_color_final'] = self._to_uint8_np(
                debug_extras['cond_render_color'][0])
        if 'cond_render_depth' in debug_extras:
            payload['cond_render_depth_final'] = self._to_f16_np(
                debug_extras['cond_render_depth'][0])

        # Alignment.
        align = debug_extras.get('align')
        payload['align_method'] = str(getattr(ev, 'alignment_method', 'none'))
        if align is not None:
            payload['align_s'] = np.float32(
                float(align['s'][0]) if torch.is_tensor(align['s']) else float(align['s']))
            payload['align_R_pa'] = self._to_f32_np(
                align['R_pa'][0] if align['R_pa'].ndim == 3 else align['R_pa'])
            payload['align_T_pa'] = self._to_f32_np(
                align['T_pa'][0] if align['T_pa'].ndim == 2 else align['T_pa'])
        else:
            payload['align_s'] = np.float32('nan')
        if 'ext_input_eff' in debug_extras:
            payload['ext_input_eff'] = self._to_f32_np(debug_extras['ext_input_eff'][0])
        if 'cond_intrinsics_used' in debug_extras:
            payload['cond_intrinsics_used'] = self._to_f32_np(
                debug_extras['cond_intrinsics_used'][0])

        # Test-view block.
        nv = debug_extras.get('nv')
        if nv is not None:
            payload['nv_images_gt'] = self._to_uint8_np(nv['gt_for_save'][0])
            if nv.get('gt_rgba') is not None:
                payload['nv_images_gt_rgba'] = self._to_uint8_np(nv['gt_rgba'][0])
            payload['nv_extrinsics_gt'] = self._to_f32_np(nv['extrinsics_gt'][0])
            payload['nv_intrinsics_gt'] = self._to_f32_np(nv['intrinsics_gt'][0])
            payload['nv_extrinsics_pred'] = self._to_f32_np(nv['extrinsics_pred'][0])
            payload['nv_intrinsics_pred'] = self._to_f32_np(nv['intrinsics_pred'][0])
            payload['nv_render_color'] = self._to_uint8_np(nv['render_color'][0])
            payload['nv_pose_source'] = str(nv['pose_source'])

        if payload:
            self.save_debug_file(indices, payload)

        # 3DGS + mesh + GT mesh as sibling files.
        final_preds = debug_extras.get('final_preds')
        if final_preds is not None:
            self._save_final_reps(indices, final_preds, data)

    # Default slat encoder name on disk under ``<root>/latents/<name>/``.
    # Standard TRELLIS pre-encoded slats use this; override on the saver
    # subclass or set ``ev.gt_slat_encoder_name`` to point elsewhere.
    DEFAULT_SLAT_ENCODER_NAME = 'dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16'

    def _save_final_reps(self, indices, preds, data):
        """Dump final 3DGS + mesh into ``debug/sample_<idx>/``: writes
        ``mesh.glb`` (textured + simplified via ``to_glb``) and ``gs.ply``,
        plus ``gt_mesh.glb`` baked from the dataset-side raw slat npz when
        present (``<root>/latents/<encoder_name>/<sha256>.npz``).
        """
        from ...utils import postprocessing_utils
        ev = self.evaluator
        idx_np = indices.cpu().numpy()
        dataset_index = int(idx_np[0])
        out_dir = os.path.join(ev.results_root, 'debug', f'sample_{dataset_index}')
        os.makedirs(out_dir, exist_ok=True)

        with torch.enable_grad():
            glb = postprocessing_utils.to_glb(
                preds['gaussian'][0], preds['mesh'][0],
                simplify=0.95, texture_size=1024, verbose=False,
            )
        glb.export(os.path.join(out_dir, 'mesh.glb'))
        preds['gaussian'][0].save_ply(os.path.join(out_dir, 'gs.ply'))

        # GT mesh: load raw slat from disk and decode directly (the in-batch
        # ``data['x_0']`` for sparse multipose is the SUV latent, not a slat).
        if 'slat_decoder_mesh' not in ev.models or 'slat_decoder_gs' not in ev.models:
            return
        inst_idx, _ = ev.get_instance_view_indices(idx_np)
        inst_row = ev.instances[int(inst_idx[0])]
        if not (hasattr(inst_row, '__len__') and not isinstance(inst_row, str)) or len(inst_row) < 2:
            return  # no (root, sha256) to locate the npz
        root, sha256 = str(inst_row[0]), str(inst_row[1])
        encoder_name = getattr(ev, 'gt_slat_encoder_name', self.DEFAULT_SLAT_ENCODER_NAME)
        slat_path = os.path.join(root, 'latents', encoder_name, f'{sha256}.npz')
        if not os.path.isfile(slat_path):
            return

        z = np.load(slat_path, allow_pickle=False)
        feats_np, coords_np = z['feats'], z['coords']
        device = next(ev.models['slat_decoder_mesh'].parameters()).device
        batch_col = np.zeros((coords_np.shape[0], 1), dtype=np.int32)
        coords = torch.from_numpy(
            np.concatenate([batch_col, coords_np.astype(np.int32)], axis=1)
        ).contiguous().to(device)
        feats = torch.from_numpy(feats_np).float().to(device)
        st = SparseTensor(coords=coords, feats=feats)
        # Disk feats are already raw (un-normalized); call decoders directly,
        # bypassing ``_decode_slat``'s ``feats * std + mean`` un-normalization.
        with torch.no_grad():
            gt_mesh = ev.models['slat_decoder_mesh'](st[0:1])[0]
            gt_gs = ev.models['slat_decoder_gs'](st[0:1])[0]
        with torch.enable_grad():
            gt_glb = postprocessing_utils.to_glb(
                gt_gs, gt_mesh,
                simplify=0.95, texture_size=1024, verbose=False,
            )
        gt_glb.export(os.path.join(out_dir, 'gt_mesh.glb'))
