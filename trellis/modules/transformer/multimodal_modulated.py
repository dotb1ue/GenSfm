from typing import *
import torch

from ..fusion import BaseFusionLayer
from .modulated import apply_mod_gate, apply_mod_scale_shift, ModulatedTransformerCrossBlock, ModulatedTransformerCrossBlockWithWeightedCond


class MultiModalModulatedTransformerCrossBlock(ModulatedTransformerCrossBlock):
    """
    Shape/pose transformer cross-attention block operating on two tensors:
      shape: [B, N_shape, C]
      poses: [B, V, N_pose, C]

    ``self_attn_layout`` and ``cross_attn_layout`` are dispatched independently:
      - Self-attention:
          "joint"                — merge shape + all V pose views into one sequence
                                   (shape-shape, shape-pose, and cross-view pose-pose all active).
          "per_modality"         — shape self-attends alone; pose tokens from all V views are
                                   merged into one [B, V*N_pose, C] sequence so cross-view
                                   pose-pose attention is active, but shape-pose cross is NOT.
          "per_modality_per_view"— shape self-attends alone; each pose view self-attends
                                   independently on [B*V, N_pose, C] (no cross-view coupling
                                   via self-attn at all).
      - Cross-attention (context is [B, V*L_per_view, C], V per-view image slices concatenated):
          "full"              — shape and all pose views cross-attend to the full concatenated
                                context (every shape/pose token sees every image view).
          "per_view"          — shape cross-attends to the full context; each pose view
                                cross-attends only to its own image slice (pose_v ↔ image_v).
          "averaged_per_view" — shape cross-attends to each image view separately and averages
                                the per-view attention outputs (shared Q, per-view K/V); pose
                                side is identical to "per_view" (pose_v ↔ image_v only).
      - MLP always runs per-modality (shared MLP is token-wise, results are equivalent).
    """

    def __init__(
        self,
        *args,
        fusion_layer: BaseFusionLayer | None = None,
        fusion_placement: Literal["pre", "post"] = "pre",
        self_attn_layout: Literal["joint", "per_modality", "per_modality_per_view"] = "joint",
        cross_attn_layout: Literal["full", "per_view", "averaged_per_view"] = "full",
        num_register_tokens: int = 0,
        num_pose_register_tokens: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fusion_layer = fusion_layer
        self.fusion_placement = fusion_placement if fusion_layer is not None else None
        self.self_attn_layout = self_attn_layout
        self.cross_attn_layout = cross_attn_layout
        self.num_register_tokens = num_register_tokens
        self.num_pose_register_tokens = num_pose_register_tokens

    def _fuse(
        self,
        shape: torch.Tensor,
        poses: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Register tokens (prepended to their respective streams) bypass the fusion
        # layer — it assumes shape/pose tokens are full 3D grids. Strip regs, fuse
        # the grid cores, then re-prepend.
        R_s, R_p = self.num_register_tokens, self.num_pose_register_tokens
        shape_core = shape[:, R_s:] if R_s > 0 else shape
        pose_core = poses[:, :, R_p:] if R_p > 0 else poses
        shape_core, pose_core = self.fusion_layer(shape_core, pose_core)
        if R_s > 0:
            shape_core = torch.cat([shape[:, :R_s], shape_core], dim=1)
        if R_p > 0:
            pose_core = torch.cat([poses[:, :, :R_p], pose_core], dim=2)
        return shape_core, pose_core

    def _forward_block(
        self,
        shape: torch.Tensor,
        poses: torch.Tensor,
        mod_shape: torch.Tensor,
        mod_poses: torch.Tensor,
        context: torch.Tensor,
        context_weight: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, V, N_pose, C = poses.shape
        N_shape = shape.shape[1]

        # -- Compute modulation --
        if self.share_mod:
            shape_mod, pose_mod = mod_shape, mod_poses
        else:
            shape_mod = self.adaLN_modulation(mod_shape)
            pose_mod = self.adaLN_modulation(mod_poses)
        shape_chunks = shape_mod.chunk(6, dim=-1)
        pose_chunks = [c.reshape(B * V, -1) for c in pose_mod.chunk(6, dim=-1)]
        shifts_msa = [shape_chunks[0], pose_chunks[0]]
        scales_msa = [shape_chunks[1], pose_chunks[1]]
        gates_msa = [shape_chunks[2], pose_chunks[2]]
        shifts_mlp = [shape_chunks[3], pose_chunks[3]]
        scales_mlp = [shape_chunks[4], pose_chunks[4]]
        gates_mlp = [shape_chunks[5], pose_chunks[5]]

        pose_flat = poses.reshape(B * V, N_pose, C)

        # -- Self-attention --
        if self.self_attn_layout == "joint":
            mod_seqlens = [N_shape, *([N_pose] * V)]
            scales_j = torch.cat([scales_msa[0].unsqueeze(1), scales_msa[1].reshape(B, V, -1)], dim=1)
            shifts_j = torch.cat([shifts_msa[0].unsqueeze(1), shifts_msa[1].reshape(B, V, -1)], dim=1)
            gates_j = torch.cat([gates_msa[0].unsqueeze(1), gates_msa[1].reshape(B, V, -1)], dim=1)
            x = torch.cat([shape, pose_flat.reshape(B, V * N_pose, C)], dim=1)
            h = self.norm1(x)
            h = apply_mod_scale_shift(h, scales_j, shifts_j, mod_seqlens)
            h = self.self_attn(h)
            h = apply_mod_gate(h, gates_j, mod_seqlens)
            x = x + h
            shape = x[:, :N_shape]
            pose_flat = x[:, N_shape:].reshape(B * V, N_pose, C)
        else:
            # Shape self-attends alone (no shape↔pose cross in either sub-mode).
            h = self.norm1(shape)
            h = apply_mod_scale_shift(h, scales_msa[0], shifts_msa[0], None)
            h = self.self_attn(h)
            shape = shape + apply_mod_gate(h, gates_msa[0], None)

            # Pose: merged [B, V*N_pose] with per-view mods (per_modality), or flat [B*V, N_pose] (per_modality_per_view).
            if self.self_attn_layout == "per_modality":
                pose_stream = pose_flat.reshape(B, V * N_pose, C)
                pose_scale = scales_msa[1].reshape(B, V, -1)
                pose_shift = shifts_msa[1].reshape(B, V, -1)
                pose_gate = gates_msa[1].reshape(B, V, -1)
                pose_seqlens = [N_pose] * V
            else:  # per_modality_per_view
                pose_stream = pose_flat
                pose_scale = scales_msa[1]
                pose_shift = shifts_msa[1]
                pose_gate = gates_msa[1]
                pose_seqlens = None
            h = self.norm1(pose_stream)
            h = apply_mod_scale_shift(h, pose_scale, pose_shift, pose_seqlens)
            h = self.self_attn(h)
            pose_flat = pose_stream + apply_mod_gate(h, pose_gate, pose_seqlens)
            pose_flat = pose_flat.reshape(B * V, N_pose, C)

        # -- Cross-attention (norm2 has its own affine — no adaLN needed) --
        h_shape = self.norm2(shape)
        h_pose_normed = self.norm2(pose_flat)

        if self.cross_attn_layout == "full":
            cross_kwargs = {"weight": context_weight} if context_weight is not None else {}
            h_pose_cat = h_pose_normed.reshape(B, V * N_pose, C)
            h_joint = self.cross_attn(torch.cat([h_shape, h_pose_cat], dim=1), context, **cross_kwargs)
            h_shape = h_joint[:, :N_shape]
            h_pose = h_joint[:, N_shape:].reshape(B * V, N_pose, C)
        else:  # "per_view" or "averaged_per_view"
            # Common per-view reshape for context and optional weights
            L_per_view = context.shape[1] // V
            ctx_views = context.reshape(B, V, L_per_view, -1)         # [B, V, L_per_view, C]
            ctx_pv = ctx_views.reshape(B * V, L_per_view, -1)         # [B*V, L_per_view, C]
            if context_weight is not None:
                w_views = context_weight.reshape(B, V, L_per_view)    # [B, V, L_per_view]
                w_pv = w_views.reshape(B * V, L_per_view)             # [B*V, L_per_view]
            else:
                w_views = w_pv = None

            # Shape side: the only path that differs between the two layouts
            if self.cross_attn_layout == "per_view":
                shape_kwargs = {"weight": context_weight} if context_weight is not None else {}
                h_shape = self.cross_attn(h_shape, context, **shape_kwargs)
            else:  # averaged_per_view — Q-shared averaging over V views
                h_shape = self.cross_attn.forward_cross_attn_averaged(
                    h_shape, ctx_views, weight=w_views,
                )

            # Pose side: identical in both layouts (per-view routing)
            pose_kwargs = {"weight": w_pv} if w_pv is not None else {}
            h_pose = self.cross_attn(h_pose_normed, ctx_pv, **pose_kwargs)

        shape = shape + h_shape
        pose_flat = pose_flat + h_pose

        # -- MLP (always per-modality — shared MLP is token-wise, equivalent) --
        xs = [shape, pose_flat]
        for i in range(2):
            h = self.norm3(xs[i])
            h = apply_mod_scale_shift(h, scales_mlp[i], shifts_mlp[i], None)
            h = self.mlp(h)
            h = apply_mod_gate(h, gates_mlp[i], None)
            xs[i] = xs[i] + h

        shape, pose_flat = xs
        return shape, pose_flat.reshape(B, V, N_pose, C)

    def _forward_impl(
        self,
        shape: torch.Tensor,
        poses: torch.Tensor,
        mod_shape: torch.Tensor,
        mod_poses: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.fusion_placement == "pre":
            shape, poses = self._fuse(shape, poses)
        shape, poses = self._forward_block(shape, poses, mod_shape, mod_poses, context)
        if self.fusion_placement == "post":
            shape, poses = self._fuse(shape, poses)
        return shape, poses

    def forward(
        self,
        shape: torch.Tensor,
        poses: torch.Tensor,
        mod_shape: torch.Tensor,
        mod_poses: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, shape, poses, mod_shape, mod_poses, context,
                use_reentrant=False,
            )
        return self._forward_impl(shape, poses, mod_shape, mod_poses, context)


class MultiModalModulatedTransformerCrossBlockWithWeightedCond(ModulatedTransformerCrossBlockWithWeightedCond):
    """
    Weighted-conditioning version of MultiModalModulatedTransformerCrossBlock.
    """

    def __init__(
        self,
        *args,
        fusion_layer: BaseFusionLayer | None = None,
        fusion_placement: Literal["pre", "post"] = "pre",
        self_attn_layout: Literal["joint", "per_modality", "per_modality_per_view"] = "joint",
        cross_attn_layout: Literal["full", "per_view", "averaged_per_view"] = "full",
        num_register_tokens: int = 0,
        num_pose_register_tokens: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fusion_layer = fusion_layer
        self.fusion_placement = fusion_placement if fusion_layer is not None else None
        self.self_attn_layout = self_attn_layout
        self.cross_attn_layout = cross_attn_layout
        self.num_register_tokens = num_register_tokens
        self.num_pose_register_tokens = num_pose_register_tokens

    _fuse = MultiModalModulatedTransformerCrossBlock._fuse

    def _forward_impl(
        self,
        shape: torch.Tensor,
        poses: torch.Tensor,
        mod_shape: torch.Tensor,
        mod_poses: torch.Tensor,
        context: torch.Tensor,
        context_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.fusion_placement == "pre":
            shape, poses = self._fuse(shape, poses)
        shape, poses = MultiModalModulatedTransformerCrossBlock._forward_block(
            self, shape, poses, mod_shape, mod_poses, context, context_weight,
        )
        if self.fusion_placement == "post":
            shape, poses = self._fuse(shape, poses)
        return shape, poses

    def forward(
        self,
        shape: torch.Tensor,
        poses: torch.Tensor,
        mod_shape: torch.Tensor,
        mod_poses: torch.Tensor,
        context: torch.Tensor,
        context_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, shape, poses, mod_shape, mod_poses, context, context_weight,
                use_reentrant=False,
            )
        return self._forward_impl(shape, poses, mod_shape, mod_poses, context, context_weight)
