from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..modules.utils import convert_module_to_f16, convert_module_to_f32
from ..modules.transformer import (
    AbsolutePositionEmbedder,
    ModulatedTransformerCrossBlock,
    ModulatedTransformerCrossBlockWithWeightedCond,
    MultiModalModulatedTransformerCrossBlock,
    MultiModalModulatedTransformerCrossBlockWithWeightedCond,
)
from ..modules.fusion import FUSION_LAYER_REGISTRY
from ..modules.spatial import patchify, unpatchify
from .occlusion_helper import OcclusionMaskPatcher


def abs_pos_embed_3d(resolution: int, patch_size: int, channels: int):
    """
    Make absolute position embedding for a spatial input with given resolution, patch size, and channels.
    """
    pos_embedder = AbsolutePositionEmbedder(channels, 3)
    coords = torch.meshgrid(*[torch.arange(res) for res in [resolution // patch_size] * 3], indexing='ij')
    coords = torch.stack(coords, dim=-1).reshape(-1, 3)
    pos_emb = pos_embedder(coords)
    return pos_emb


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: a 1-D Tensor of N indices, one per batch element.
                These may be fractional.
            dim: the dimension of the output.
            max_period: controls the minimum frequency of the embeddings.

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class SparseStructureFlowModel(nn.Module):
    """
    DiT-style flow-matching model over a dense 3D sparse-structure latent.

    Patchifies ``[B, C, R, R, R]`` voxels, runs modulated transformer blocks
    with timestep AdaLN and image cross-attention, and unpatchifies back.
    ``enable_occlusion_mask`` reweights cross-attention by an occlusion mask.
    """

    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        enable_occlusion_mask: bool = False,
        occlusion_mask_patcher_args: Dict[str, Any] = {},
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.enable_occlusion_mask = enable_occlusion_mask
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape":
            self.register_buffer("pos_emb", abs_pos_embed_3d(resolution, patch_size, model_channels))

        self.input_layer = nn.Linear(in_channels * patch_size**3, model_channels)

        if self.enable_occlusion_mask:
            self.occlusion_mask_patcher = OcclusionMaskPatcher(**occlusion_mask_patcher_args)
            transformer_block_cls = ModulatedTransformerCrossBlockWithWeightedCond
        else:
            transformer_block_cls = ModulatedTransformerCrossBlock
            
        self.blocks = nn.ModuleList([
            transformer_block_cls(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])

        self.out_layer = nn.Linear(model_channels, out_channels * patch_size**3)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, occlusion_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], \
                f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], self.in_channels, *[self.resolution] * 3]}"

        h = patchify(x, self.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()

        h = self.input_layer(h)
        h = h + self.pos_emb[None]
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        h = h.type(self.dtype)
        cond = cond.type(self.dtype)

        if self.enable_occlusion_mask:
            assert occlusion_mask is not None, "Occlusion mask is required when enable_occlusion_mask is True"
            occlusion_emb = self.occlusion_mask_patcher.tokenize_mask(occlusion_mask)
            occlusion_emb = occlusion_emb.type(self.dtype)
            for block in self.blocks:
                h = block(h, t_emb, cond, occlusion_emb)
        else:
            for block in self.blocks:
                h = block(h, t_emb, cond)
        h = h.type(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)

        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3)
        h = unpatchify(h, self.patch_size).contiguous()

        return h


class MultiPoseSparseStructureFlowModel(nn.Module):
    """
    Two-stream multi-modal DiT for joint shape + multi-view pose flow.

    Runs parallel token streams for the shape grid ``x`` and per-view pose
    grids ``P``, each with its own timestep AdaLN (``t_x`` / ``t_P``). Self-
    and cross-attention sharing across streams is controlled by
    ``self_attn_layout`` and ``cross_attn_layout``. Optional: fusion layer
    (pre/post), per-view embeddings, shared register tokens, occlusion mask.
    """

    def __init__(
        self,
        resolution: int,
        in_channels: int,
        out_channels: int,
        pose_resolution: int,
        pose_in_channels: int,
        pose_out_channels: int,
        model_channels: int,
        cond_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        pose_patch_size: int = 1,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        fusion_layer_type: Optional[str] = None,
        fusion_layer_args: Dict[str, Any] = {},
        fusion_placement: Literal["pre", "post"] = "pre",
        self_attn_layout: Literal["joint", "per_modality", "per_modality_per_view"] = "joint",
        cross_attn_layout: Literal["full", "per_view", "averaged_per_view"] = "full",
        num_view_embeds: int = 0,
        view_embed_random_perm: bool = False,
        num_shared_register_tokens: int = 0,
        num_pose_register_tokens: int = 0,
        enable_occlusion_mask: bool = False,
        occlusion_mask_patcher_args: Dict[str, Any] = {},
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.shape_in_channels = in_channels
        self.out_channels = out_channels
        self.shape_out_channels = out_channels
        self.pose_resolution = pose_resolution
        self.pose_in_channels = pose_in_channels
        self.pose_out_channels = pose_out_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.pose_patch_size = pose_patch_size
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.self_attn_layout = self_attn_layout
        self.cross_attn_layout = cross_attn_layout
        self.num_view_embeds = num_view_embeds
        self.view_embed_random_perm = view_embed_random_perm
        self.num_shared_register_tokens = num_shared_register_tokens
        self.num_pose_register_tokens = num_pose_register_tokens
        self.enable_occlusion_mask = enable_occlusion_mask
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape":
            self.register_buffer("pos_emb", abs_pos_embed_3d(resolution, patch_size, model_channels))
            self.register_buffer("pose_pos_emb", abs_pos_embed_3d(pose_resolution, pose_patch_size, model_channels))

        if num_view_embeds > 0:
            self.view_embed_pose = nn.Parameter(torch.empty(num_view_embeds, model_channels))
            self.view_embed_cond = nn.Parameter(torch.empty(num_view_embeds, cond_channels))

        if num_shared_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.empty(num_shared_register_tokens, model_channels))

        if num_pose_register_tokens > 0:
            self.pose_register_tokens = nn.Parameter(torch.empty(num_pose_register_tokens, model_channels))

        self.input_layer = nn.Linear(in_channels * patch_size**3, model_channels)
        self.pose_input_layer = nn.Linear(pose_in_channels * pose_patch_size**3, model_channels)

        # Resolve fusion layer class (None = no fusion)
        fusion_layer_cls = FUSION_LAYER_REGISTRY[fusion_layer_type] if fusion_layer_type else None

        if self.enable_occlusion_mask:
            self.occlusion_mask_patcher = OcclusionMaskPatcher(**occlusion_mask_patcher_args)
            transformer_block_cls = MultiModalModulatedTransformerCrossBlockWithWeightedCond
        else:
            transformer_block_cls = MultiModalModulatedTransformerCrossBlock

        def _make_fusion_layer():
            if fusion_layer_cls is None:
                return None
            return fusion_layer_cls(
                channels=model_channels,
                shape_grid_size=resolution // patch_size,
                pose_grid_size=pose_resolution // pose_patch_size,
                **fusion_layer_args,
            )

        self.blocks = nn.ModuleList([
            transformer_block_cls(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
                fusion_layer=_make_fusion_layer(),
                fusion_placement=fusion_placement,
                self_attn_layout=self.self_attn_layout,
                cross_attn_layout=self.cross_attn_layout,
                num_register_tokens=num_shared_register_tokens,
                num_pose_register_tokens=num_pose_register_tokens,
            )
            for _ in range(num_blocks)
        ])

        self.out_layer = nn.Linear(model_channels, out_channels * patch_size**3)
        self.pose_out_layer = nn.Linear(model_channels, pose_out_channels * pose_patch_size**3)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Initialize view-index embeddings:
        if self.num_view_embeds > 0:
            nn.init.normal_(self.view_embed_pose, std=0.02)
            nn.init.normal_(self.view_embed_cond, std=0.02)

        # Initialize shared register tokens:
        if self.num_shared_register_tokens > 0:
            nn.init.normal_(self.register_tokens, std=0.02)
        if self.num_pose_register_tokens > 0:
            nn.init.normal_(self.pose_register_tokens, std=0.02)

        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)
        nn.init.constant_(self.pose_out_layer.weight, 0)
        nn.init.constant_(self.pose_out_layer.bias, 0)

        # Zero-out fusion layer output heads:
        for block in self.blocks:
            if hasattr(block, 'fusion_layer') and block.fusion_layer is not None:
                block.fusion_layer.initialize_weights()

    def _sample_view_perm(self, V: int, device: torch.device) -> torch.LongTensor:
        K = self.num_view_embeds
        assert V <= K, f"Number of views ({V}) exceeds num_view_embeds ({K})"
        if self.view_embed_random_perm:
            return torch.randperm(K, device=device)[:V]
        return torch.arange(V, device=device)

    def forward(self,
        x: torch.Tensor,
        P: torch.Tensor,
        t_x: torch.Tensor,
        t_P: torch.Tensor,
        cond: torch.Tensor,
        occlusion_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W, D], the input sparse structure
            P: [B, N_views, C_pose, H_pose, W_pose, D_pose], the input multi-view poses
            t_x: [B], the timestep for x
            t_P: [B, N_views], the timestep for P
            cond: [B, N_views, L_cond, C_cond], the conditioning
            occlusion_mask: [B, N_views, H, W, D], the occlusion mask
            **kwargs: additional arguments

        Returns:
            x_pred: [B, C, H, W, D], the output sparse structure
            P_pred: [B, N_views, C_pose, H_pose, W_pose, D_pose], the output multi-view poses
        """
        assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], \
                f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], self.in_channels, *[self.resolution] * 3]}"
        assert [*P.shape] == [x.shape[0], P.shape[1], self.pose_in_channels, *[self.pose_resolution] * 3], \
                f"Pose shape mismatch, got {P.shape}, expected {[x.shape[0], P.shape[1], self.pose_in_channels, *[self.pose_resolution] * 3]}"
        assert [*cond.shape] == [x.shape[0], P.shape[1], cond.shape[2], self.cond_channels], \
                f"Condition shape mismatch, got {cond.shape}, expected {[x.shape[0], P.shape[1], cond.shape[2], self.cond_channels]}"

        hx = patchify(x, self.patch_size)
        hx = hx.view(*hx.shape[:2], -1).permute(0, 2, 1).contiguous()
        hP = patchify(P.flatten(end_dim=1), self.pose_patch_size)
        hP = hP.view(*P.shape[:2], hP.shape[1], -1).permute(0, 1, 3, 2).contiguous()

        hx = self.input_layer(hx) + self.pos_emb[None]
        hP = self.pose_input_layer(hP) + self.pose_pos_emb[None, None]
        if self.num_pose_register_tokens > 0:
            # Prepend before view_embed so regs inherit view identity but not pose_pos_emb.
            reg = self.pose_register_tokens[None, None].expand(hP.shape[0], hP.shape[1], -1, -1)
            hP = torch.cat([reg, hP], dim=2)
        if self.num_view_embeds > 0:
            V = hP.shape[1]
            idx = self._sample_view_perm(V, hP.device)
            hP = hP + self.view_embed_pose[idx][None, :, None, :]
            cond = cond + self.view_embed_cond[idx][None, :, None, :]
        t_x_emb = self.t_embedder(t_x.reshape(-1))
        t_P_emb = self.t_embedder(t_P.reshape(-1)).view(t_P.shape[0], t_P.shape[1], -1)
        if self.share_mod:
            t_x_emb = self.adaLN_modulation(t_x_emb)
            t_P_emb = self.adaLN_modulation(t_P_emb)
        t_x_emb = t_x_emb.type(self.dtype)
        t_P_emb = t_P_emb.type(self.dtype)
        hx = hx.type(self.dtype)
        hP = hP.type(self.dtype)
        cond = cond.type(self.dtype).flatten(start_dim=1, end_dim=2)

        if self.num_shared_register_tokens > 0:
            reg_tokens = self.register_tokens.to(dtype=self.dtype).unsqueeze(0).expand(hx.shape[0], -1, -1)
            hx = torch.cat([reg_tokens, hx], dim=1)

        if self.enable_occlusion_mask:
            assert occlusion_mask is not None, "Occlusion mask is required when enable_occlusion_mask is True"
            occlusion_emb = self.occlusion_mask_patcher.tokenize_mask(occlusion_mask)
            occlusion_emb = occlusion_emb.type(self.dtype)
            for block in self.blocks:
                hx, hP = block(hx, hP, t_x_emb, t_P_emb, cond, occlusion_emb)
        else:
            for block in self.blocks:
                hx, hP = block(hx, hP, t_x_emb, t_P_emb, cond)

        if self.num_shared_register_tokens > 0:
            hx = hx[:, self.num_shared_register_tokens:]
        if self.num_pose_register_tokens > 0:
            hP = hP[:, :, self.num_pose_register_tokens:]

        hx = F.layer_norm(hx.type(x.dtype), hx.shape[-1:])
        hP = F.layer_norm(hP.type(x.dtype), hP.shape[-1:])
        ox = self.out_layer(hx)
        oP = self.pose_out_layer(hP)

        ox = ox.permute(0, 2, 1).view(ox.shape[0], ox.shape[2], *[self.resolution // self.patch_size] * 3)
        ox = unpatchify(ox, self.patch_size).contiguous()
        pose_token_channels = oP.shape[3]
        oP = oP.permute(0, 1, 3, 2).reshape(
            oP.shape[0] * oP.shape[1],
            pose_token_channels,
            *[self.pose_resolution // self.pose_patch_size] * 3
        )
        oP = unpatchify(oP, self.pose_patch_size)
        oP = oP.view(x.shape[0], P.shape[1], -1, *oP.shape[2:]).contiguous()

        return ox, oP


class VisualMultiPoseSparseStructureFlowModel(MultiPoseSparseStructureFlowModel):
    """
    MultiPose DiT that fuses DINOv2 patch tokens with per-view multi-scale
    visual CNN features.

    ``cond`` is a dict ``{'dino_cond': [B,V,L,C_dino], 'visual_cond':
    [B,V,3,H,W]}``. Visual features are encoded per view through a CNN pyramid,
    each level is patchified to ``visual_patch_resolution**2`` tokens and
    channel-concatenated onto the last ``L_patch`` DINO tokens; the ``L-L_patch``
    prefix tokens (CLS + register) zero-pad their visual half. Optional trailing
    zero-pad channels let ``cond_channels`` land on a hardware-friendly value.
    """

    def __init__(
        self,
        *args,
        cond_channels: int,
        dino_cond_channels: int = 1024,
        visual_input_channels: int = 3,
        visual_feat_resolution: int = 296,
        visual_patch_resolution: int = 37,
        visual_feat_channels: List[int] = [6, 6, 6, 8],
        visual_conv_kernel_size: int = 3,
        **kwargs,
    ):
        import warnings

        assert visual_feat_resolution % visual_patch_resolution == 0, \
            f"visual_feat_resolution ({visual_feat_resolution}) must be divisible by visual_patch_resolution ({visual_patch_resolution})"
        ratio = visual_feat_resolution // visual_patch_resolution
        assert ratio > 0 and (ratio & (ratio - 1)) == 0, \
            f"visual_feat_resolution / visual_patch_resolution ({ratio}) must be a power of 2"
        expected_levels = int(np.log2(ratio)) + 1
        assert len(visual_feat_channels) == expected_levels, \
            f"visual_feat_channels has {len(visual_feat_channels)} levels, expected {expected_levels} for ratio {ratio}"

        # Compute the natural visual channel count from the pyramid.
        visual_cond_channels = 0
        r = visual_feat_resolution
        for c in visual_feat_channels:
            visual_cond_channels += c * (r // visual_patch_resolution) ** 2
            r //= 2

        required = dino_cond_channels + visual_cond_channels
        if cond_channels < required:
            raise ValueError(
                f"cond_channels ({cond_channels}) is smaller than required dino_cond_channels ({dino_cond_channels}) "
                f"+ visual_cond_channels ({visual_cond_channels}) = {required}. Increase cond_channels or reduce visual_feat_channels."
            )
        visual_pad_channels = cond_channels - required
        if visual_pad_channels > 0:
            warnings.warn(
                f"VisualMultiPoseSparseStructureFlowModel: zero-padding {visual_pad_channels} visual cond channels "
                f"to reach cond_channels={cond_channels} (dino={dino_cond_channels} + visual={visual_cond_channels} + pad={visual_pad_channels}).",
                stacklevel=2,
            )

        super().__init__(*args, cond_channels=cond_channels, **kwargs)

        self.dino_cond_channels = dino_cond_channels
        self.visual_cond_channels = visual_cond_channels
        self.visual_pad_channels = visual_pad_channels
        self.visual_input_channels = visual_input_channels
        self.visual_feat_resolution = visual_feat_resolution
        self.visual_patch_resolution = visual_patch_resolution
        self.visual_feat_channels = visual_feat_channels
        self.visual_conv_kernel_size = visual_conv_kernel_size

        # Build the per-level Conv2d pyramid (first level at stride 1, rest at stride 2).
        in_res = out_res = visual_feat_resolution
        in_ch = visual_input_channels
        self.visual_conv_blocks = nn.ModuleList()
        for out_ch in visual_feat_channels:
            is_downsample = out_res < in_res
            self.visual_conv_blocks.append(nn.Conv2d(
                in_ch + int(self.enable_occlusion_mask),
                out_ch,
                kernel_size=visual_conv_kernel_size,
                padding=visual_conv_kernel_size // 2,
                stride=2 if is_downsample else 1,
            ))
            in_ch = out_ch
            in_res = out_res
            out_res //= 2

    def fuse_cond(
        self,
        dino_cond: torch.Tensor,
        visual_cond: torch.Tensor,
        occlusion_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            dino_cond: [B, V, L, C_dino]
            visual_cond: [B, V, 3, H, W]
            occlusion_mask: [B, V, H_occ, W_occ] (optional)
        Returns:
            [B, V, L, cond_channels] — DINO tokens with per-view multi-scale
            visual features concatenated onto the last L_patch tokens; prefix
            tokens zero-pad their visual half.
        """
        B, V, L, _ = dino_cond.shape
        L_patch = self.visual_patch_resolution ** 2
        assert L >= L_patch, f"dino_cond has {L} tokens per view, need >= L_patch={L_patch} for patch alignment"

        # Fold V into batch so the CNN stack runs per-view.
        vc = visual_cond.reshape(B * V, *visual_cond.shape[2:])
        om_flat = None
        if self.enable_occlusion_mask:
            assert occlusion_mask is not None, "occlusion_mask is required when enable_occlusion_mask is True"
            om_flat = occlusion_mask.reshape(B * V, *occlusion_mask.shape[2:])

        vc = F.interpolate(
            vc,
            size=(self.visual_feat_resolution, self.visual_feat_resolution),
            mode='bilinear', align_corners=False, antialias=True,
        )

        visual_feats: List[torch.Tensor] = []
        if self.enable_occlusion_mask:
            for conv in self.visual_conv_blocks:
                om_i = F.interpolate(
                    om_flat[:, None].float(), size=vc.shape[-2:],
                    mode='bilinear', align_corners=False, antialias=True,
                )
                vc = conv(torch.cat([vc, om_i], dim=1))
                visual_feats.append(vc)
        else:
            for conv in self.visual_conv_blocks:
                vc = conv(vc)
                visual_feats.append(vc)

        # Patchify each level down to visual_patch_resolution**2 tokens and
        # channel-concatenate across levels (plus optional zero-pad channels).
        out_res = self.visual_feat_resolution
        tokens: List[torch.Tensor] = []
        for vf in visual_feats:
            p = out_res // self.visual_patch_resolution
            vf = patchify(vf, p)
            vf = vf.flatten(start_dim=2, end_dim=3).permute(0, 2, 1)
            tokens.append(vf)
            out_res //= 2
        visual_tokens = torch.cat(tokens, dim=2)  # [B*V, L_patch, visual_cond_channels]
        if self.visual_pad_channels > 0:
            pad = torch.zeros(
                *visual_tokens.shape[:2], self.visual_pad_channels,
                device=visual_tokens.device, dtype=visual_tokens.dtype,
            )
            visual_tokens = torch.cat([visual_tokens, pad], dim=2)
        visual_tokens = visual_tokens.reshape(B, V, L_patch, -1)

        # Split DINO tokens into prefix + patch halves; zero-pad prefix's visual side.
        dino_prefix = dino_cond[:, :, :-L_patch]
        dino_patches = dino_cond[:, :, -L_patch:]
        C_vis_total = visual_tokens.shape[-1]
        prefix_pad = torch.zeros(
            *dino_prefix.shape[:3], C_vis_total,
            device=dino_prefix.device, dtype=dino_prefix.dtype,
        )
        fused_prefix = torch.cat([dino_prefix, prefix_pad], dim=-1)
        fused_patches = torch.cat([dino_patches, visual_tokens], dim=-1)
        return torch.cat([fused_prefix, fused_patches], dim=2)

    def forward(
        self,
        x: torch.Tensor,
        P: torch.Tensor,
        t_x: torch.Tensor,
        t_P: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        occlusion_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fused = self.fuse_cond(
            dino_cond=cond['dino_cond'],
            visual_cond=cond['visual_cond'],
            occlusion_mask=occlusion_mask,
        )
        return super().forward(x, P, t_x, t_P, fused, occlusion_mask=occlusion_mask, **kwargs)
