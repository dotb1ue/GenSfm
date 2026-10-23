from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .full_attn import scaled_dot_product_attention


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.normalize(x.float(), dim = -1) * self.gamma * self.scale).to(x.dtype)




class RotaryPositionEmbedder(nn.Module):
    def __init__(self, hidden_size: int, in_channels: int = 3):
        super().__init__()
        assert hidden_size % 2 == 0, "Hidden size must be divisible by 2"
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.freq_dim = hidden_size // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000 ** self.freqs)
        
    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        self.freqs = self.freqs.to(indices.device)
        phases = torch.outer(indices, self.freqs)
        phases = torch.polar(torch.ones_like(phases), phases)
        return phases
        
    def _rotary_embedding(self, x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * phases
        x_embed = torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1).to(x.dtype)
        return x_embed
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q (sp.SparseTensor): [..., N, D] tensor of queries
            k (sp.SparseTensor): [..., N, D] tensor of keys
            indices (torch.Tensor): [..., N, C] tensor of spatial positions
        """
        if indices is None:
            indices = torch.arange(q.shape[-2], device=q.device)
            if len(q.shape) > 2:
                indices = indices.unsqueeze(0).expand(q.shape[:-2] + (-1,))
        
        phases = self._get_phases(indices.reshape(-1)).reshape(*indices.shape[:-1], -1)
        if phases.shape[1] < self.hidden_size // 2:
            phases = torch.cat([phases, torch.polar(
                torch.ones(*phases.shape[:-1], self.hidden_size // 2 - phases.shape[1], device=phases.device),
                torch.zeros(*phases.shape[:-1], self.hidden_size // 2 - phases.shape[1], device=phases.device)
            )], dim=-1)
        q_embed = self._rotary_embedding(q, phases)
        k_embed = self._rotary_embedding(k, phases)
        return q_embed, k_embed
    

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int]=None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "windowed"], f"Invalid attention mode: {attn_mode}"
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"
        
        if attn_mode == "windowed":
            raise NotImplementedError("Windowed attention is not yet implemented")
        
        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
            
        if self.qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            
        self.to_out = nn.Linear(channels, channels)

        if use_rope:
            self.rope = RotaryPositionEmbedder(channels)

    def _forward_self_attn(self, x: torch.Tensor, indices: Optional[torch.Tensor]) -> torch.Tensor:
        B, L, _ = x.shape
        qkv = self.to_qkv(x).reshape(B, L, 3, self.num_heads, -1)
        if self.use_rope:
            q, k, v = qkv.unbind(dim=2)
            q, k = self.rope(q, k, indices)
            qkv = torch.stack([q, k, v], dim=2)
        if self.attn_mode == "windowed":
            raise NotImplementedError("Windowed attention is not yet implemented")
        if self.qk_rms_norm:
            q, k, v = qkv.unbind(dim=2)
            q = self.q_rms_norm(q)
            k = self.k_rms_norm(k)
            h = scaled_dot_product_attention(q, k, v)
        else:
            h = scaled_dot_product_attention(qkv)
        return self.to_out(h.reshape(B, L, -1))

    def _forward_cross_attn(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        Lkv = context.shape[1]
        q = self.to_q(x).reshape(B, L, self.num_heads, -1)
        kv = self.to_kv(context).reshape(B, Lkv, 2, self.num_heads, -1)
        if self.qk_rms_norm:
            q = self.q_rms_norm(q)
            k, v = kv.unbind(dim=2)
            k = self.k_rms_norm(k)
            h = scaled_dot_product_attention(q, k, v, attn_bias=attn_bias)
        else:
            h = scaled_dot_product_attention(q, kv, attn_bias=attn_bias)
        return self.to_out(h.reshape(B, L, -1))

    def forward_cross_attn_averaged(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Q-shared multi-view averaged cross-attention.

        Args:
            x: query tensor of shape [B, L, C].
            context: dense context of shape [B, V, Lkv, C_ctx].
            weight: optional per-view token weights of shape [B, V, Lkv]; if given,
                an attention bias log(weight + eps) is applied per view.

        Q and its optional rms-norm are computed once and shared across views.
        K/V projection and attention run per view. Per-view attention outputs are
        accumulated and averaged before the single output projection (valid
        because to_out is linear).
        """
        assert self._type == "cross", "forward_cross_attn_averaged is only supported for cross-attention"
        assert context.ndim == 4, \
            f"forward_cross_attn_averaged expects context of shape [B, V, Lkv, C], got ndim={context.ndim}"
        if weight is not None:
            assert weight.ndim == 3 and weight.shape[1] == context.shape[1], \
                f"weight must be [B, V, Lkv] matching context, got {tuple(weight.shape)} vs {tuple(context.shape)}"
        B, L, _ = x.shape
        V, Lkv = context.shape[1], context.shape[2]

        # Q projection — once, shared across views
        q = self.to_q(x).reshape(B, L, self.num_heads, -1)
        if self.qk_rms_norm:
            q = self.q_rms_norm(q)

        # Per-view K/V + attention, accumulate pre-output-projection
        accum: Optional[torch.Tensor] = None
        for i in range(V):
            kv = self.to_kv(context[:, i]).reshape(B, Lkv, 2, self.num_heads, -1)
            attn_bias = torch.log(weight[:, i, None, None, :] + 1e-7) if weight is not None else None
            if self.qk_rms_norm:
                k, v = kv.unbind(dim=2)
                k = self.k_rms_norm(k)
                h_i = scaled_dot_product_attention(q, k, v, attn_bias=attn_bias)
            else:
                h_i = scaled_dot_product_attention(q, kv, attn_bias=attn_bias)
            h_i = h_i.reshape(B, L, -1)
            accum = h_i if accum is None else accum + h_i

        return self.to_out(accum / V)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: input tensor (self-attn) or query (cross-attn).
            context: context tensor for cross-attention; ignored for self-attn.
            weight: optional per-context-token weights of shape [B, Lkv] for
                cross-attention; if given, an attention bias log(weight + eps)
                is applied. Not supported for self-attention.
            indices: optional indices for rope.
        """
        if self._type == "self":
            assert weight is None, "weight is only supported for cross-attention"
            return self._forward_self_attn(x, indices=indices)
        if context is None:
            raise ValueError("context is required for cross-attention")
        attn_bias = torch.log(weight[:, None, None, :] + 1e-7) if weight is not None else None
        return self._forward_cross_attn(x, context=context, attn_bias=attn_bias)
