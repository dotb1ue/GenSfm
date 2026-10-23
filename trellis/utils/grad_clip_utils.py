from typing import *
import torch
import numpy as np
import torch.utils


@torch.no_grad()
def _adaptive_clip_step(
    grads: list,
    max_norm: torch.Tensor,
    norm_buf: torch.Tensor,
    buf_ptr: torch.Tensor,
    buf_len: torch.Tensor,
    buffer_size: int,
    k_index: int,
    hard_max_norm: torch.Tensor,
) -> torch.Tensor:
    """
    Fully GPU-resident adaptive gradient clipping step.
    No CUDA→CPU synchronization in this function.
    """
    # 1. Compute total L2 norm (foreach, no sync)
    norms = torch._foreach_norm(grads, 2.0)
    total_norm = torch.linalg.vector_norm(torch.stack(norms), 2.0)

    # 2. Clip grads unconditionally (matching PyTorch's sync-free approach)
    clip_coef = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    torch._foreach_mul_(grads, clip_coef)

    # 3. Update ring buffer on GPU (branchless via torch.where)
    is_finite = torch.isfinite(total_norm)
    # Use gather instead of direct indexing to avoid data-dependent scalar access
    old_val = norm_buf.gather(0, buf_ptr.unsqueeze(0)).squeeze(0)
    norm_buf.scatter_(
        0, buf_ptr.unsqueeze(0),
        torch.where(is_finite, total_norm, old_val).unsqueeze(0),
    )
    new_ptr = torch.where(is_finite, (buf_ptr + 1) % buffer_size, buf_ptr)
    new_len = torch.where(is_finite, torch.clamp(buf_len + 1, max=buffer_size), buf_len)
    buf_ptr.copy_(new_ptr)
    buf_len.copy_(new_len)

    # 4. Update adaptive max_norm when buffer is full
    is_full = (new_len >= buffer_size)
    percentile_val = torch.kthvalue(norm_buf, k_index).values
    new_max = torch.minimum(
        torch.where(is_full, percentile_val, max_norm),
        hard_max_norm,
    )
    max_norm.copy_(new_max)

    return total_norm


class AdaptiveGradClipper:
    """
    Adaptive gradient clipping for training.
    All hot-path operations run on GPU without CUDA→CPU synchronization.
    """
    def __init__(
        self,
        max_norm=None,
        clip_percentile=95.0,
        buffer_size=1000,
    ):
        self.max_norm = max_norm
        self.clip_percentile = clip_percentile
        self.buffer_size = buffer_size
        self._k_index = max(1, int(clip_percentile / 100.0 * buffer_size))

        # CPU state (used for checkpoint compat and pre-GPU-init)
        self._grad_norm = np.zeros(buffer_size, dtype=np.float32)
        self._max_norm = max_norm
        self._buffer_ptr = 0
        self._buffer_length = 0

        # GPU state (lazily initialized on first __call__)
        self._gpu_norm_buf: Optional[torch.Tensor] = None
        self._gpu_buf_ptr: Optional[torch.Tensor] = None
        self._gpu_buf_len: Optional[torch.Tensor] = None
        self._gpu_max_norm: Optional[torch.Tensor] = None
        self._gpu_hard_max_norm: Optional[torch.Tensor] = None
        self._gpu_initialized = False

        # Compiled clip step (created on first __call__)
        self._compiled_clip_step = None

    def _init_gpu_state(self, device: torch.device):
        """Initialize GPU tensors from current CPU state."""
        self._gpu_norm_buf = torch.from_numpy(self._grad_norm.copy()).to(device=device, dtype=torch.float32)
        self._gpu_buf_ptr = torch.tensor(self._buffer_ptr, device=device, dtype=torch.long)
        self._gpu_buf_len = torch.tensor(self._buffer_length, device=device, dtype=torch.long)

        init_max = self._max_norm if self._max_norm is not None else float('inf')
        self._gpu_max_norm = torch.tensor(init_max, device=device, dtype=torch.float32)

        hard_max = self.max_norm if self.max_norm is not None else float('inf')
        self._gpu_hard_max_norm = torch.tensor(hard_max, device=device, dtype=torch.float32)

        try:
            self._compiled_clip_step = torch.compile(
                _adaptive_clip_step, fullgraph=True,
            )
        except Exception:
            self._compiled_clip_step = _adaptive_clip_step

        self._gpu_initialized = True

    def __repr__(self):
        return f'AdaptiveGradClipper(max_norm={self.max_norm}, clip_percentile={self.clip_percentile})'

    def state_dict(self):
        if self._gpu_initialized:
            return {
                'grad_norm': self._gpu_norm_buf.cpu().numpy(),
                'max_norm': self._gpu_max_norm.item(),
                'buffer_ptr': self._gpu_buf_ptr.item(),
                'buffer_length': self._gpu_buf_len.item(),
            }
        return {
            'grad_norm': self._grad_norm,
            'max_norm': self._max_norm,
            'buffer_ptr': self._buffer_ptr,
            'buffer_length': self._buffer_length,
        }

    def load_state_dict(self, state_dict):
        self._grad_norm = state_dict['grad_norm']
        self._max_norm = state_dict['max_norm']
        self._buffer_ptr = state_dict['buffer_ptr']
        self._buffer_length = state_dict['buffer_length']
        # Reset GPU state so it re-initializes from loaded CPU state
        self._gpu_initialized = False

    def log(self):
        if self._gpu_initialized:
            return {'max_norm': self._gpu_max_norm.item()}
        return {'max_norm': self._max_norm}

    def __call__(self, parameters):
        """Clip gradient norm in-place and return total L2 norm. Fully GPU-resident, no CUDA sync."""
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        else:
            parameters = list(parameters)
        grads = [p.grad for p in parameters if p.grad is not None]
        if len(grads) == 0:
            return torch.tensor(0.0)

        if not self._gpu_initialized:
            self._init_gpu_state(grads[0].device)

        total_norm = self._compiled_clip_step(
            grads,
            self._gpu_max_norm,
            self._gpu_norm_buf,
            self._gpu_buf_ptr,
            self._gpu_buf_len,
            self.buffer_size,
            self._k_index,
            self._gpu_hard_max_norm,
        )
        return total_norm
