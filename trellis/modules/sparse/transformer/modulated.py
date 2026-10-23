from typing import *
import torch
import torch.nn as nn
from ..basic import SparseTensor
from ..attention import (
    SparseSelfAttention,
    SparseCrossAttention,
    SerializeMode,
)
from ...norm import LayerNorm32
from ...transformer.blocks import FeedForwardNet


class ModulatedSparseTransformerBlock(nn.Module):
    """
    Sparse Transformer block (MSA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.attn = SparseSelfAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_sequence=shift_sequence,
            shift_window=shift_window,
            serialize_mode=serialize_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: SparseTensor, mod: torch.Tensor) -> SparseTensor:
        feats = x.feats
        batch_idx = x.coords[:, 0]  # [T] batch index for broadcasting [B, C] → [T, C]

        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        # Self-attention
        h = self.norm1(feats)
        h = h * (1 + scale_msa[batch_idx]) + shift_msa[batch_idx]
        h = self.attn(x.replace(h)).feats
        h = h * gate_msa[batch_idx]
        feats = feats + h
        # FFN
        h = self.norm2(feats)
        h = h * (1 + scale_mlp[batch_idx]) + shift_mlp[batch_idx]
        h = self.mlp(h)
        h = h * gate_mlp[batch_idx]
        feats = feats + h
        return x.replace(feats)

    def forward(self, x: SparseTensor, mod: torch.Tensor) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, use_reentrant=False)
        else:
            return self._forward(x, mod)


class ModulatedSparseTransformerCrossBlock(nn.Module):
    """
    Sparse Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.

    When multi_view_cross_attn_fusion is True, the cross-attention expects context of shape
    [B, V, L, C] and runs attention independently against each view's K/V; per-view residual
    updates are averaged before the output projection. Self-attention, FFN, and AdaLN modulation
    are unchanged.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        multi_view_cross_attn_fusion: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.multi_view_cross_attn_fusion = multi_view_cross_attn_fusion
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = SparseSelfAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_sequence=shift_sequence,
            shift_window=shift_window,
            serialize_mode=serialize_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = SparseCrossAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor) -> SparseTensor:
        feats = x.feats
        batch_idx = x.coords[:, 0]

        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        # Self-attention
        h = self.norm1(feats)
        h = h * (1 + scale_msa[batch_idx]) + shift_msa[batch_idx]
        h = self.self_attn(x.replace(h)).feats
        h = h * gate_msa[batch_idx]
        feats = feats + h
        # Cross-attention
        h = self.norm2(feats)
        if self.multi_view_cross_attn_fusion:
            h = self.cross_attn.forward_cross_attn_averaged(x.replace(h), context).feats
        else:
            h = self.cross_attn(x.replace(h), context).feats
        feats = feats + h
        # FFN
        h = self.norm3(feats)
        h = h * (1 + scale_mlp[batch_idx]) + shift_mlp[batch_idx]
        h = self.mlp(h)
        h = h * gate_mlp[batch_idx]
        feats = feats + h
        return x.replace(feats)

    def forward(self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, use_reentrant=False)
        else:
            return self._forward(x, mod, context)


class ModulatedSparseTransformerCrossBlockWithWeightedCond(ModulatedSparseTransformerCrossBlock):
    """
    Sparse Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm weighted conditioning.

    Uses the same SparseCrossAttention as the base class — both forward() and
    forward_cross_attn_averaged() accept an optional weight= kwarg, so this subclass
    only exists to thread context_weight through the block's forward signature.

    When multi_view_cross_attn_fusion is True, context is expected as [B, V, L, C] and context_weight
    as [B, V, L]; per-view weighted attention is computed and averaged before the output projection.
    """
    def _forward(self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor, context_weight: torch.Tensor) -> SparseTensor:
        feats = x.feats
        batch_idx = x.coords[:, 0]

        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        # Self-attention
        h = self.norm1(feats)
        h = h * (1 + scale_msa[batch_idx]) + shift_msa[batch_idx]
        h = self.self_attn(x.replace(h)).feats
        h = h * gate_msa[batch_idx]
        feats = feats + h
        # Weighted cross-attention
        h = self.norm2(feats)
        if self.multi_view_cross_attn_fusion:
            h = self.cross_attn.forward_cross_attn_averaged(x.replace(h), context, weight=context_weight).feats
        else:
            h = self.cross_attn(x.replace(h), context, weight=context_weight).feats
        feats = feats + h
        # FFN
        h = self.norm3(feats)
        h = h * (1 + scale_mlp[batch_idx]) + shift_mlp[batch_idx]
        h = self.mlp(h)
        h = h * gate_mlp[batch_idx]
        feats = feats + h
        return x.replace(feats)

    def forward(self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor, context_weight: torch.Tensor) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, context_weight, use_reentrant=False)
        else:
            return self._forward(x, mod, context, context_weight)