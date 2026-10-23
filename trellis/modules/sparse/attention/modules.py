from typing import *
import torch
import torch.nn as nn
from .. import SparseTensor
from .full_attn import sparse_scaled_dot_product_attention
from .serialized_attn import SerializeMode, sparse_serialized_scaled_dot_product_self_attention
from .windowed_attn import sparse_windowed_scaled_dot_product_self_attention
from ...attention import RotaryPositionEmbedder
from ...attention.modules import MultiHeadRMSNorm


class SparseSelfAttention(nn.Module):
    """
    Self-attention for sparse tensors.
    Input x is always a SparseTensor. Operates on raw .feats tensors
    and wraps back to SparseTensor only at boundaries.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        attn_mode: Literal["full", "serialized", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_sequence = shift_sequence
        self.shift_window = shift_window
        self.serialize_mode = serialize_mode
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.to_out = nn.Linear(channels, channels)

        if qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)

        if use_rope:
            self.rope = RotaryPositionEmbedder(channels)

    def forward(self, x: SparseTensor) -> SparseTensor:
        # QKV projection on raw feats
        qkv = self.to_qkv(x.feats)  # [T, 3*C]
        qkv = qkv.reshape(qkv.shape[0], 3, self.num_heads, self.head_dim)  # [T, 3, H, D]
        qkv = x.replace(qkv)  # wrap for attention (needs layout/coords)

        if self.use_rope:
            q, k, v = qkv.feats.unbind(dim=1)  # [T, H, D]
            q, k = self.rope(q, k, qkv.coords[:, 1:])
            qkv = qkv.replace(torch.stack([q, k, v], dim=1))

        if self.qk_rms_norm:
            q, k, v = qkv.feats.unbind(dim=1)  # [T, H, D]
            q = self.q_rms_norm(q)
            k = self.k_rms_norm(k)
            qkv = qkv.replace(torch.stack([q, k, v], dim=1))

        # Attention dispatch
        if self.attn_mode == "full":
            h = sparse_scaled_dot_product_attention(qkv)
        elif self.attn_mode == "serialized":
            h = sparse_serialized_scaled_dot_product_self_attention(
                qkv, self.window_size, serialize_mode=self.serialize_mode,
                shift_sequence=self.shift_sequence, shift_window=self.shift_window
            )
        elif self.attn_mode == "windowed":
            h = sparse_windowed_scaled_dot_product_self_attention(
                qkv, self.window_size, shift_window=self.shift_window
            )

        # Output projection on raw feats
        return x.replace(self.to_out(h.feats.reshape(h.feats.shape[0], -1)))


class SparseCrossAttention(nn.Module):
    """
    Cross-attention for sparse tensors.
    Query x is always a SparseTensor, context is always a dense torch.Tensor.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.ctx_channels = ctx_channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.qk_rms_norm = qk_rms_norm

        self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
        self.to_kv = nn.Linear(ctx_channels, channels * 2, bias=qkv_bias)
        self.to_out = nn.Linear(channels, channels)

        if qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)

    def forward(
        self,
        x: SparseTensor,
        context: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> SparseTensor:
        """
        Cross-attention forward.

        Args:
            x: sparse query tensor.
            context: dense context of shape [B, L, C].
            weight: optional per-context-token weights of shape [B, L]; if given,
                an attention bias log(weight + eps) is applied.
        """
        # Q from sparse feats
        q = self.to_q(x.feats)  # [T, C]
        q = q.reshape(q.shape[0], self.num_heads, self.head_dim)  # [T, H, D]

        # KV from dense context
        kv = self.to_kv(context)  # [B, L, 2*C]
        kv = kv.reshape(*kv.shape[:2], 2, self.num_heads, self.head_dim)  # [B, L, 2, H, D]

        if self.qk_rms_norm:
            q = self.q_rms_norm(q)
            k, v = kv.unbind(dim=2)  # [B, L, H, D] each
            k = self.k_rms_norm(k)
            kv = torch.stack([k, v], dim=2)

        attn_bias = torch.log(weight[:, None, None, :] + 1e-7) if weight is not None else None

        # Wrap q for attention (needs layout/coords)
        q = x.replace(q)
        h = sparse_scaled_dot_product_attention(q, kv, attn_bias=attn_bias)

        # Output projection on raw feats
        return x.replace(self.to_out(h.feats.reshape(h.feats.shape[0], -1)))

    def forward_cross_attn_averaged(
        self,
        x: SparseTensor,
        context: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> SparseTensor:
        """
        Q-shared multi-view averaged cross-attention.

        Args:
            x: sparse query tensor.
            context: dense context of shape [B, V, L, C].
            weight: optional per-view token weights of shape [B, V, L]; if given,
                an attention bias log(weight + eps) is applied per view.

        Q and its optional rms-norm are computed once and shared across views.
        K/V projection and attention run per view. Per-view attention outputs are
        accumulated and averaged before the single output projection (valid
        because to_out is linear).
        """
        assert context.ndim == 4, \
            f"forward_cross_attn_averaged expects context of shape [B, V, L, C], got ndim={context.ndim}"
        if weight is not None:
            assert weight.ndim == 3, \
                f"forward_cross_attn_averaged expects weight of shape [B, V, L], got ndim={weight.ndim}"
            assert weight.shape[1] == context.shape[1], \
                f"context and weight must have matching view dim, got {context.shape[1]} vs {weight.shape[1]}"
        V = context.shape[1]

        # Q from sparse feats — computed once, shared across views
        q = self.to_q(x.feats)  # [T, C]
        q = q.reshape(q.shape[0], self.num_heads, self.head_dim)  # [T, H, D]
        if self.qk_rms_norm:
            q = self.q_rms_norm(q)
        q_sparse = x.replace(q)

        # Per-view: KV projection + attention, accumulate pre-output-projection
        accum = torch.zeros_like(x.feats)  # [T, C]
        for i in range(V):
            kv = self.to_kv(context[:, i])  # [B, L, 2*C]
            kv = kv.reshape(*kv.shape[:2], 2, self.num_heads, self.head_dim)  # [B, L, 2, H, D]
            if self.qk_rms_norm:
                k, v = kv.unbind(dim=2)
                k = self.k_rms_norm(k)
                kv = torch.stack([k, v], dim=2)
            if weight is not None:
                attn_bias = torch.log(weight[:, i, None, None, :] + 1e-7)
                h = sparse_scaled_dot_product_attention(q_sparse, kv, attn_bias=attn_bias)
            else:
                h = sparse_scaled_dot_product_attention(q_sparse, kv)
            accum = accum + h.feats.reshape(h.feats.shape[0], -1)

        return x.replace(self.to_out(accum / V))

