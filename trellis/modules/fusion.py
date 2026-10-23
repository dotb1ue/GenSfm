from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .norm import LayerNorm32
from .attention import MultiHeadAttention


def _tokens_to_grid(x: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Reshape token sequence to 3D grid. [B, G^3, C] -> [B, C, G, G, G]"""
    B, _, C = x.shape
    return x.permute(0, 2, 1).reshape(B, C, grid_size, grid_size, grid_size)


def _grid_to_tokens(x: torch.Tensor) -> torch.Tensor:
    """Reshape 3D grid to token sequence. [B, C, G, G, G] -> [B, G^3, C]"""
    B, C = x.shape[:2]
    return x.reshape(B, C, -1).permute(0, 2, 1)


def _extract_pose_neighborhoods(pose_grid: torch.Tensor, k: int) -> torch.Tensor:
    """Extract k x k x k neighborhoods from a 3D pose grid.

    Args:
        pose_grid: [N, P, P, P, C]
        k: neighborhood kernel size (must be odd)
    Returns:
        [N, P^3, k^3, C]
    """
    N, P, _, _, C = pose_grid.shape
    if k == 1:
        return pose_grid.reshape(N, P ** 3, 1, C)
    pad = k // 2
    # Pad spatial dims with replicate
    g = pose_grid.permute(0, 4, 1, 2, 3)                   # [N, C, P, P, P]
    g = F.pad(g, [pad] * 6, mode='replicate')              # [N, C, P+2pad, ...]
    g = g.permute(0, 2, 3, 4, 1)                           # [N, P+2pad, P+2pad, P+2pad, C]
    g = g.unfold(1, k, 1).unfold(2, k, 1).unfold(3, k, 1)  # [N, P, P, P, C, k, k, k]
    g = g.permute(0, 1, 2, 3, 5, 6, 7, 4).contiguous()     # [N, P, P, P, k, k, k, C]
    return g.reshape(N, P ** 3, k ** 3, C)


class BaseFusionLayer(nn.Module):
    """
    Base class for shape-pose fusion layers.

    Provides the shared fusion FFN (norm -> linear -> GELU -> out_proj), spatial
    resampling helpers, and zero-init for ``out_proj``.
    Subclasses implement ``forward()`` to define how shape and pose tokens
    interact with the FFN.
    """
    def __init__(
        self,
        channels: int,
        shape_grid_size: int,
        pose_grid_size: int,
        mlp_ratio: float = 2.0,
        fusion_resolution: Literal["shape", "pose"] = "shape",
    ):
        super().__init__()
        self.channels = channels
        self.shape_grid_size = shape_grid_size
        self.pose_grid_size = pose_grid_size
        self._same_grid = (shape_grid_size == pose_grid_size)
        self._ds_factor = shape_grid_size // pose_grid_size if not self._same_grid else 1
        self.fusion_resolution = fusion_resolution

        # Fusion FFN: cat(shape, pose) [2C] -> hidden -> [2C] -> chunk
        hidden = int(channels * mlp_ratio)
        self.norm = LayerNorm32(2 * channels, elementwise_affine=False, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(2 * channels, hidden),
            nn.GELU(approximate="tanh"),
        )
        self.out_proj = nn.Linear(hidden, 2 * channels)

    def forward(
        self,
        shape: torch.Tensor,
        poses: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            shape: [B, N_shape, C] shape tokens
            poses: [B, N_views, N_pose, C] pose tokens
        Returns:
            shape: [B, N_shape, C] updated shape tokens
            poses: [B, N_views, N_pose, C] updated per-view pose tokens
        """
        raise NotImplementedError

    def _upsample(self, tokens: torch.Tensor) -> torch.Tensor:
        """Resample pose-resolution tokens to shape grid resolution."""
        if self._same_grid:
            return tokens
        grid = _tokens_to_grid(tokens, self.pose_grid_size)
        grid = F.interpolate(grid, size=self.shape_grid_size, mode='nearest')
        return _grid_to_tokens(grid)

    def _downsample(self, tokens: torch.Tensor) -> torch.Tensor:
        """Resample shape-resolution tokens to pose grid resolution."""
        if self._same_grid:
            return tokens
        grid = _tokens_to_grid(tokens, self.shape_grid_size)
        grid = F.avg_pool3d(grid, kernel_size=self._ds_factor, stride=self._ds_factor)
        return _grid_to_tokens(grid)

    @property
    def _fuse_at_pose(self) -> bool:
        return self.fusion_resolution == "pose" and not self._same_grid

    def _shape_to_fuse(self, tokens: torch.Tensor) -> torch.Tensor:
        """Resample shape tokens to fusion resolution."""
        return self._downsample(tokens) if self._fuse_at_pose else tokens

    def _fuse_to_shape(self, tokens: torch.Tensor) -> torch.Tensor:
        """Resample fusion-resolution tokens back to shape resolution."""
        return self._upsample(tokens) if self._fuse_at_pose else tokens

    def _pose_to_fuse(self, tokens: torch.Tensor) -> torch.Tensor:
        """Resample pose tokens to fusion resolution."""
        return tokens if self._fuse_at_pose else self._upsample(tokens)

    def _fuse_to_pose(self, tokens: torch.Tensor) -> torch.Tensor:
        """Resample fusion-resolution tokens back to pose resolution."""
        return tokens if self._fuse_at_pose else self._downsample(tokens)

    def _fuse(self, shape_tokens: torch.Tensor, pose_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Channel-concat -> norm -> FFN -> out_proj -> chunk into shape_delta + pose_delta."""
        h = torch.cat([shape_tokens, pose_tokens], dim=-1)
        h = self.norm(h)
        h = self.ffn(h)
        h = self.out_proj(h)
        return h.chunk(2, dim=-1)

    def initialize_weights(self) -> None:
        """Zero-init output projection so fusion starts as identity. Called after global init."""
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)


class PerViewFusionLayer(BaseFusionLayer):
    """
    Per-view fusion: each view is independently fused with shape through the
    shared FFN, then shape deltas are averaged across views.

    This preserves per-view information through the nonlinear FFN, unlike
    :class:`AggregateFusionLayer` which pre-averages poses before the FFN.
    At V=1 the two are equivalent.
    """
    def forward(
        self,
        shape: torch.Tensor,
        poses: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, V, N_pose, C = poses.shape
        N_fuse = (self.pose_grid_size if self._fuse_at_pose else self.shape_grid_size) ** 3

        # Resample to fusion resolution and batch all views
        shape_f = self._shape_to_fuse(shape)                                        # [B, N_fuse, C]
        shape_rep = shape_f[:, None].expand(-1, V, -1, -1).reshape(B * V, N_fuse, C)
        pose_f = self._pose_to_fuse(poses.reshape(B * V, N_pose, C))               # [B*V, N_fuse, C]

        # Shared FFN per view
        shape_delta, pose_delta = self._fuse(shape_rep, pose_f)

        # Average shape deltas across views and resample back
        shape = shape + self._fuse_to_shape(shape_delta.reshape(B, V, N_fuse, C).mean(dim=1))
        poses = poses + self._fuse_to_pose(pose_delta).reshape(B, V, N_pose, C)

        return shape, poses


class SpatialAggregateFusionLayer(BaseFusionLayer):
    """
    Aggregate fusion with spatial resampling: poses are averaged across views,
    resampled to fusion resolution, fused with shape through the FFN, and the
    pose delta is resampled back and distributed to each view.
    """
    def _aggregate(
        self,
        shape: torch.Tensor,
        poses: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate multi-view poses into fusion-resolution features."""
        return self._pose_to_fuse(poses.mean(dim=1))

    def _distribute(
        self,
        poses: torch.Tensor,
        pose_delta: torch.Tensor,
    ) -> torch.Tensor:
        """Distribute fused pose delta back to each view."""
        return poses + self._fuse_to_pose(pose_delta)[:, None]

    def forward(
        self,
        shape: torch.Tensor,
        poses: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pose_agg = self._aggregate(shape, poses)
        shape_f = self._shape_to_fuse(shape)
        shape_delta, pose_delta = self._fuse(shape_f, pose_agg)
        shape = shape + self._fuse_to_shape(shape_delta)
        poses = self._distribute(poses, pose_delta)
        return shape, poses


class SpatialAttentionAggregateFusionLayer(SpatialAggregateFusionLayer):
    """
    Aggregate fusion with windowed cross-attention for aggregation and
    distribution. Shape tokens in each r x r x r window attend to the
    corresponding pose voxel's k x k x k neighborhood across all views.

    Only supports ``fusion_resolution="shape"`` because the windowed attention
    degenerates at pose resolution (r=1 gives trivial single-token windows).
    """
    def __init__(
        self,
        channels: int,
        shape_grid_size: int,
        pose_grid_size: int,
        num_heads: int = 16,
        mlp_ratio: float = 2.0,
        qk_rms_norm: bool = False,
        neighborhood_size: int = 1,
        fusion_resolution: Literal["shape", "pose"] = "shape",
        **kwargs,
    ):
        if fusion_resolution == "pose":
            raise ValueError(
                "fusion_resolution='pose' is incompatible with spatial_attention modes"
            )
        super().__init__(
            channels, shape_grid_size, pose_grid_size,
            mlp_ratio=mlp_ratio, fusion_resolution=fusion_resolution, **kwargs,
        )
        self.neighborhood_size = neighborhood_size
        assert neighborhood_size % 2 == 1, "neighborhood_size must be odd"

        self.agg_attn = MultiHeadAttention(
            channels, num_heads=num_heads, ctx_channels=channels,
            type="cross", attn_mode="full", qk_rms_norm=qk_rms_norm,
        )
        self.dist_attn = MultiHeadAttention(
            channels, num_heads=num_heads, ctx_channels=channels,
            type="cross", attn_mode="full", qk_rms_norm=qk_rms_norm,
        )

    def _block_partition(self, tokens: torch.Tensor) -> torch.Tensor:
        """Partition shape-resolution tokens into pose-aligned windows.
        [B, S^3, C] -> [B*P^3, r^3, C]
        """
        S = self.shape_grid_size
        P = self.pose_grid_size
        r = self._ds_factor
        B, _, C = tokens.shape
        x = tokens.reshape(B, P, r, P, r, P, r, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7)     # [B, P, P, P, r, r, r, C]
        return x.reshape(B * P ** 3, r ** 3, C)

    def _block_unpartition(self, blocks: torch.Tensor, B: int) -> torch.Tensor:
        """Reverse of _block_partition.
        [B*P^3, r^3, C] -> [B, S^3, C]
        """
        S = self.shape_grid_size
        P = self.pose_grid_size
        r = self._ds_factor
        C = blocks.shape[-1]
        x = blocks.reshape(B, P, P, P, r, r, r, C)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7)     # [B, P, r, P, r, P, r, C]
        return x.reshape(B, S ** 3, C)

    def _aggregate(
        self,
        shape: torch.Tensor,
        poses: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate multi-view poses via windowed cross-attention."""
        B, V, N_pose, C = poses.shape
        P = self.pose_grid_size
        k = self.neighborhood_size

        # Extract neighborhoods from each view's pose grid
        pose_grid = poses.reshape(B * V, P, P, P, C)
        pose_nbrs = _extract_pose_neighborhoods(pose_grid, k)  # [B*V, P^3, k^3, C]

        # Stack views into KV: [B*P^3, V*k^3, C]
        pose_nbrs = pose_nbrs.reshape(B, V, P ** 3, k ** 3, C)
        kv = pose_nbrs.permute(0, 2, 1, 3, 4).reshape(B * P ** 3, V * k ** 3, C)

        # Block-partition shape into windows: [B*P^3, r^3, C]
        q = self._block_partition(shape)

        # Cross attention
        pose_agg = self.agg_attn(q, context=kv)     # [B*P^3, r^3, C]

        return self._block_unpartition(pose_agg, B)

    def _distribute(
        self,
        poses: torch.Tensor,
        pose_delta: torch.Tensor,
    ) -> torch.Tensor:
        """Distribute fused pose delta via attention-based pooling."""
        B, V, N_pose, C = poses.shape
        P = self.pose_grid_size
        r = self._ds_factor

        # Block-partition pose_delta: [B*P^3, r^3, C]
        delta_blocks = self._block_partition(pose_delta)

        # Each view's pose token as query: [B*V*P^3, 1, C]
        q = poses.reshape(B * V * P ** 3, 1, C)

        # Expand KV over views: [B*P^3, r^3, C] -> [B*V*P^3, r^3, C]
        kv = delta_blocks.reshape(B, 1, P ** 3, r ** 3, C).expand(-1, V, -1, -1, -1)
        kv = kv.reshape(B * V * P ** 3, r ** 3, C)

        # Cross attention: each pose token selects from its window's deltas
        updates = self.dist_attn(q, context=kv)      # [B*V*P^3, 1, C]

        return poses + updates.reshape(B, V, P ** 3, C)

    def initialize_weights(self) -> None:
        super().initialize_weights()
        nn.init.zeros_(self.agg_attn.to_out.weight)
        nn.init.zeros_(self.agg_attn.to_out.bias)
        nn.init.zeros_(self.dist_attn.to_out.weight)
        nn.init.zeros_(self.dist_attn.to_out.bias)


FUSION_LAYER_REGISTRY = {
    'PerViewFusionLayer': PerViewFusionLayer,
    'SpatialAggregateFusionLayer': SpatialAggregateFusionLayer,
    'SpatialAttentionAggregateFusionLayer': SpatialAttentionAggregateFusionLayer,
}
