from typing import *
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32
from ..modules.transformer import AbsolutePositionEmbedder
from ..modules.norm import LayerNorm32
from ..modules import sparse as sp
from ..modules.sparse.transformer import (
    ModulatedSparseTransformerCrossBlock,
    ModulatedSparseTransformerCrossBlockWithWeightedCond,
)
from ..modules.spatial import patchify
from .sparse_structure_flow import TimestepEmbedder
from .. import models
from ..utils import dist_utils
from .occlusion_helper import OcclusionMaskPatcher


class SparseResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        out_channels: Optional[int] = None,
        downsample: bool = False,
        upsample: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.downsample = downsample
        self.upsample = upsample
        
        assert not (downsample and upsample), "Cannot downsample and upsample at the same time"

        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels, bias=True),
        )
        self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        self.updown = None
        if self.downsample:
            self.updown = sp.SparseDownsample(2)
        elif self.upsample:
            self.updown = sp.SparseUpsample(2)

    def _updown(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.updown is not None:
            x = self.updown(x)
        return x

    @torch.compiler.disable
    def forward(self, x: sp.SparseTensor, emb: torch.Tensor) -> sp.SparseTensor:
        emb_out = self.emb_layers(emb).type(x.dtype)
        scale, shift = torch.chunk(emb_out, 2, dim=1)

        x = self._updown(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h)
        h = h.replace(self.norm2(h.feats)) * (1 + scale) + shift
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)

        return h
    

class SLatFlowModel(nn.Module):
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
        num_io_res_blocks: int = 2,
        io_block_channels: List[int] = None,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        use_skip_connection: bool = True,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        enable_occlusion_mask: bool = False,
        occlusion_mask_patcher_args: Dict[str, Any] = {},
        multi_view_cross_attn_fusion: bool = False,
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
        self.num_io_res_blocks = num_io_res_blocks
        self.io_block_channels = io_block_channels
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.use_skip_connection = use_skip_connection
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.enable_occlusion_mask = enable_occlusion_mask
        self.multi_view_cross_attn_fusion = multi_view_cross_attn_fusion
        self.dtype = torch.float16 if use_fp16 else torch.float32

        if self.io_block_channels is not None:
            assert int(np.log2(patch_size)) == np.log2(patch_size), "Patch size must be a power of 2"
            assert np.log2(patch_size) == len(io_block_channels), "Number of IO ResBlocks must match the number of stages"

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        self.input_layer = sp.SparseLinear(in_channels, model_channels if io_block_channels is None else io_block_channels[0])
        
        self.input_blocks = nn.ModuleList([])
        if io_block_channels is not None:
            for chs, next_chs in zip(io_block_channels, io_block_channels[1:] + [model_channels]):
                self.input_blocks.extend([
                    SparseResBlock3d(
                        chs,
                        model_channels,
                        out_channels=chs,
                    )
                    for _ in range(num_io_res_blocks-1)
                ])
                self.input_blocks.append(
                    SparseResBlock3d(
                        chs,
                        model_channels,
                        out_channels=next_chs,
                        downsample=True,
                    )
                )
            
        if self.enable_occlusion_mask:
            self.occlusion_mask_patcher = OcclusionMaskPatcher(**occlusion_mask_patcher_args)
            transformer_block_cls = ModulatedSparseTransformerCrossBlockWithWeightedCond
        else:
            transformer_block_cls = ModulatedSparseTransformerCrossBlock

        self.blocks = nn.ModuleList([
            transformer_block_cls(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=self.share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
                multi_view_cross_attn_fusion=self.multi_view_cross_attn_fusion,
            )
            for _ in range(num_blocks)
        ])

        self.out_blocks = nn.ModuleList([])
        if io_block_channels is not None:
            for chs, prev_chs in zip(reversed(io_block_channels), [model_channels] + list(reversed(io_block_channels[1:]))):
                self.out_blocks.append(
                    SparseResBlock3d(
                        prev_chs * 2 if self.use_skip_connection else prev_chs,
                        model_channels,
                        out_channels=chs,
                        upsample=True,
                    )
                )
                self.out_blocks.extend([
                    SparseResBlock3d(
                        chs * 2 if self.use_skip_connection else chs,
                        model_channels,
                        out_channels=chs,
                    )
                    for _ in range(num_io_res_blocks-1)
                ])
            
        self.out_layer = sp.SparseLinear(model_channels if io_block_channels is None else io_block_channels[0], out_channels)

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
        self.input_blocks.apply(convert_module_to_f16)
        self.blocks.apply(convert_module_to_f16)
        self.out_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.blocks.apply(convert_module_to_f32)
        self.out_blocks.apply(convert_module_to_f32)

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

    def forward(self, x: sp.SparseTensor, t: torch.Tensor, cond: torch.Tensor, occlusion_mask: Optional[torch.Tensor] = None, **kwargs) -> sp.SparseTensor:
        h = self.input_layer(x).type(self.dtype)
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        cond = cond.type(self.dtype)
        if self.enable_occlusion_mask:
            assert occlusion_mask is not None, "Occlusion mask is required when enable_occlusion_mask is True"
            # Patcher handles arbitrary leading dims: [B, H, W] -> [B, L], [B, V, H, W] -> [B, V, L]
            occlusion_emb = self.occlusion_mask_patcher.tokenize_mask(occlusion_mask)
            # Concat-mode upstream passes mask through as [B, V, H, W]; flatten per-view
            # tokens so the weight length matches the V-flattened cond. cross_per_view
            # keeps V via multi_view_cross_attn_fusion=True and consumes [B, V, L] natively.
            if occlusion_emb.ndim == 3 and not self.multi_view_cross_attn_fusion:
                occlusion_emb = occlusion_emb.flatten(start_dim=1, end_dim=2)
            occlusion_emb = occlusion_emb.type(self.dtype)

        skips = []
        # pack with input blocks
        for block in self.input_blocks:
            h = block(h, t_emb)
            skips.append(h.feats)
        
        if self.pe_mode == "ape":
            h = h.replace(h.feats + self.pos_embedder(h.coords[:, 1:]).type(self.dtype))
        
        if self.enable_occlusion_mask:
            for block in self.blocks:
                h = block(h, t_emb, cond, occlusion_emb)
        else:
            for block in self.blocks:
                h = block(h, t_emb, cond)

        # unpack with output blocks
        for block, skip in zip(self.out_blocks, reversed(skips)):
            if self.use_skip_connection:
                h = block(h.replace(torch.cat([h.feats, skip], dim=1)), t_emb)
            else:
                h = block(h, t_emb)

        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h.type(x.dtype))
        return h


class LatentConditioningSLatFlowModel(SLatFlowModel):
    """
    SLatFlowModel with view-conditioned latent.

    ``view_aggregation`` selects how the view dim V is fed to the cross-attention
    block. Shapes reaching the block (per mode, with ``enable_occlusion_mask=True``):

    | mode           | cond         | mask tokens | block path                  |
    |----------------|--------------|-------------|-----------------------------|
    | mean           | [B, L, C]    | [B, L]      | standard cross-attn         |
    | concat         | [B, V*L, C]  | [B, V*L]    | standard cross-attn         |
    | cross_per_view | [B, V, L, C] | [B, V, L]   | forward_cross_attn_averaged |

    ``cross_per_view`` sets ``multi_view_cross_attn_fusion=True`` on the base
    class; the other two leave it ``False``. When ``num_view_embeds > 0`` a
    learnable ``view_embed_cond[:V]`` is added to cond tokens in both
    ``concat`` and ``cross_per_view``.

    Args:
        cond_latent_channels (int): number of channels for the latent conditioning
        pretrained_slat_enc (str): name of the pretrained structured latent encoder
        slat_enc_path (str): path to the structured latent encoder, if given, will override the pretrained_slat_enc
        slat_enc_ckpt (str): name of the structured latent encoder checkpoint
        view_aggregation (str): one of ``"mean"`` / ``"concat"`` / ``"cross_per_view"``.
        num_view_embeds (int): size of the learnable per-view PE table; 0 disables it.
    """
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        cond_latent_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        num_io_res_blocks: int = 2,
        io_block_channels: List[int] = None,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        use_skip_connection: bool = True,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        enable_occlusion_mask: bool = False,
        occlusion_mask_patcher_args: Dict[str, Any] = {},
        pretrained_slat_enc: str = 'microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16',
        slat_enc_path: Optional[str] = None,
        slat_enc_ckpt: Optional[str] = None,
        view_aggregation: Literal["mean", "concat", "cross_per_view"] = "mean",
        num_view_embeds: int = 0,
        view_embed_random_perm: bool = False,
    ):
        assert view_aggregation in ("mean", "concat", "cross_per_view"), \
            f"view_aggregation must be 'mean' / 'concat' / 'cross_per_view', got {view_aggregation!r}"
        # cross_per_view keeps V intact and delegates fusion to the cross-attention block.
        super().__init__(
            resolution,
            in_channels + cond_latent_channels,
            model_channels,
            cond_channels,
            out_channels,
            num_blocks,
            num_heads,
            num_head_channels,
            mlp_ratio,
            patch_size,
            num_io_res_blocks,
            io_block_channels,
            pe_mode,
            use_fp16,
            use_checkpoint,
            use_skip_connection,
            share_mod,
            qk_rms_norm,
            qk_rms_norm_cross,
            enable_occlusion_mask,
            occlusion_mask_patcher_args,
            multi_view_cross_attn_fusion=(view_aggregation == "cross_per_view"),
        )
        self.cond_latent_channels = cond_latent_channels
        self.pretrained_slat_enc = pretrained_slat_enc
        self.slat_enc_path = slat_enc_path
        self.slat_enc_ckpt = slat_enc_ckpt
        self.slat_enc = [None]  # Use a list to avoid register the slat_enc as a sub nn.Module

        self.view_aggregation = view_aggregation
        self.num_view_embeds = num_view_embeds
        self.view_embed_random_perm = view_embed_random_perm
        if view_aggregation in ("concat", "cross_per_view") and num_view_embeds > 0:
            self.view_embed_cond = nn.Parameter(torch.empty(num_view_embeds, cond_channels))
            nn.init.normal_(self.view_embed_cond, std=0.02)

    def _sample_view_perm(self, V: int, device: torch.device) -> torch.LongTensor:
        K = self.num_view_embeds
        assert V <= K, f"V ({V}) exceeds num_view_embeds ({K})"
        if self.view_embed_random_perm:
            return torch.randperm(K, device=device)[:V]
        return torch.arange(V, device=device)

    def _loading_slat_enc(self):
        if self.slat_enc[0] is not None:
            return
        if self.slat_enc_path is not None:
            cfg = json.load(open(os.path.join(self.slat_enc_path, 'config.json'), 'r'))
            encoder = getattr(models, cfg['models']['encoder']['name'])(**cfg['models']['encoder']['args'])
            ckpt_path = os.path.join(self.slat_enc_path, 'ckpts', f'encoder_{self.slat_enc_ckpt}.pt')
            encoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        else:
            encoder = models.from_pretrained(self.pretrained_slat_enc)
        self.slat_enc[0] = encoder.eval().cuda()

    def _delete_slat_enc(self):
        del self.slat_enc[0]
        self.slat_enc = [None]

    def _sample_at_uvs(self, feat_maps: List[torch.Tensor], uvs: sp.SparseTensor) -> torch.Tensor:
        """
        Grid-sample 2D feature maps at per-item (optionally per-view) UV coords
        and mean across views.

        Shapes:
        - feat_maps[s]: ``[B*V, C_s, H_s, W_s]`` (single-view is the V=1 case,
          shape ``[B, C_s, H_s, W_s]``).
        - uvs.feats:    ``[N, V, 2]`` (multi-view) or ``[N, 2]`` (single-view).

        For each batch item ``i`` and each of its V views, samples every feature
        map at ``uvs_i[:, v]`` (scaled from [0,1] to [-1,1]), averages over V,
        and channel-concatenates across scales. Returns ``[sum_i N_i, sum_s C_s]``
        aligned with ``uvs.coords``.
        """
        is_multiview = uvs.feats.ndim == 3
        V = uvs.feats.shape[1] if is_multiview else 1
        out: List[torch.Tensor] = []
        for i, uvs_i in enumerate(uvs.unbind(0)):
            uv_i = uvs_i.feats if is_multiview else uvs_i.feats.unsqueeze(1)    # [N_i, V, 2]
            grid = uv_i.permute(1, 0, 2).unsqueeze(2) * 2 - 1                   # [V, N_i, 1, 2]
            sampled: List[torch.Tensor] = []
            for fm in feat_maps:
                fm_i = fm[i * V : (i + 1) * V]                                  # [V, C, H, W]
                f = F.grid_sample(
                    fm_i, grid,
                    mode='bilinear', padding_mode='zeros', align_corners=False,
                )                                                               # [V, C, N_i, 1]
                sampled.append(f.squeeze(-1).mean(dim=0).permute(1, 0))         # [N_i, C]
            out.append(torch.cat(sampled, dim=1))
        return torch.cat(out, dim=0)

    @torch.compiler.disable
    def _wrap_sparse_taps(self, feat_maps: List[torch.Tensor], uvs: sp.SparseTensor) -> sp.SparseTensor:
        # Eager tail: ``_sample_at_uvs`` iterates over ``uvs.unbind(0)`` with
        # per-item shapes, and ``sp.SparseTensor.__init__`` uses ``.item()``
        # in ``__cal_shape`` — both hostile to tracing. Keeping just this
        # wrap out of the compiled graph lets the dense pyramid above get
        # folded into the outer compile while the sampling/construction
        # runs eager.
        return sp.SparseTensor(feats=self._sample_at_uvs(feat_maps, uvs), coords=uvs.coords)

    @torch.compiler.disable
    @torch.no_grad()
    def encode_latent(self, uvs: sp.SparseTensor, cond: torch.Tensor, occlusion_mask: Optional[torch.Tensor] = None) -> sp.SparseTensor:
        # Per-batch-item null-condition flag: computed on the original cond
        # shape so it always has length B, matching ``latent``'s batch dim
        # (independent of V). ``cond[..., :5, :]`` selects the first 5 tokens
        # across all views; ``all(dim>=1)`` reduces to [B].
        null_condition_mask = (cond[..., :5, :] == 0).flatten(start_dim=1).all(dim=1).float()

        # Multi-view: fold V into batch so downstream ops see a flat ``B*V`` dim.
        # ``uvs.feats`` carries the V signal for the sampler (shape [N, V, 2]).
        if cond.ndim == 4:                                   # [B, V, L, C_dino]
            cond = cond.flatten(0, 1)                        # [B*V, L, C_dino]
        if occlusion_mask is not None and occlusion_mask.ndim == 4:
            occlusion_mask = occlusion_mask.flatten(0, 1)    # [B*V, H, W]
        patchtokens = cond[:, 5:].permute(0, 2, 1).reshape(cond.shape[0], 1024, 37, 37)

        if self.enable_occlusion_mask:
            assert occlusion_mask is not None, "Occlusion mask is required when enable_occlusion_mask is True"
            patch_weights = self.occlusion_mask_patcher.patchify_mask(occlusion_mask)
            patch_weights = patch_weights.type(patchtokens.dtype)
            patchtokens = patchtokens * patch_weights[:, None, :, :]

        cond_feats = sp.SparseTensor(feats=self._sample_at_uvs([patchtokens], uvs), coords=uvs.coords)

        with dist_utils.local_master_first():
            self._loading_slat_enc()
        latent = self.slat_enc[0](cond_feats, sample_posterior=False)

        latent = latent * (1 - null_condition_mask[:, None])

        return latent

    def _aggregate_views(
        self,
        cond: torch.Tensor,
        occlusion_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Reshape the view dim of ``cond`` (and optionally the occlusion mask) for
        the cross-attention block. ``None`` / non-4D inputs pass through.
        See ``LatentConditioningSLatFlowModel`` docstring for the shape table.
        """
        # Cond token path
        if cond.ndim == 4:
            V = cond.shape[1]
            if self.view_aggregation == "mean":
                cond = cond.mean(dim=1)
            elif self.view_aggregation in ("concat", "cross_per_view"):
                if self.num_view_embeds > 0:
                    idx = self._sample_view_perm(V, cond.device)
                    cond = cond + self.view_embed_cond[idx][None, :, None, :]
                if self.view_aggregation == "concat":
                    cond = cond.flatten(start_dim=1, end_dim=2)
                # cross_per_view: keep [B, V, L, C]
            else:
                raise ValueError(f"unknown view_aggregation: {self.view_aggregation!r}")

        # Mask path — only mean reduces here; concat / cross_per_view pass V through
        # so the patcher downstream can emit per-view tokens.
        if occlusion_mask is not None and occlusion_mask.ndim == 4:
            if self.view_aggregation == "mean":
                occlusion_mask = occlusion_mask.mean(dim=1)

        return cond, occlusion_mask

    def forward(self, x: sp.SparseTensor, t: torch.Tensor, cond: torch.Tensor, uvs: sp.SparseTensor, occlusion_mask: Optional[torch.Tensor] = None, **kwargs) -> sp.SparseTensor:
        latent_cond = self.encode_latent(uvs, cond, occlusion_mask)
        x = x.replace(torch.cat([x.feats, latent_cond.feats], dim=1))
        cond, occlusion_mask = self._aggregate_views(cond, occlusion_mask)
        h = super().forward(x, t, cond, occlusion_mask, **kwargs)
        return h


class VisualLatentConditioningSLatFlowModel(LatentConditioningSLatFlowModel):
    """
    SLat Flow Model with visual latent conditioning.

    Args:
        visual_input_channels (int): number of channels for the visual input. Defaults to 3.
        visual_feat_resolution (int): resolution for the first layer of visual feature. Defaults to 256.
        visual_feat_channels (int): number of channels for each layer of visual feature.
        visual_conv_kernel_size (int): kernel size for the visual convolution. Defaults to 3.
    """
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        cond_latent_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        num_io_res_blocks: int = 2,
        io_block_channels: List[int] = None,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        use_skip_connection: bool = True,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        enable_occlusion_mask: bool = False,
        occlusion_mask_patcher_args: Dict[str, Any] = {},
        pretrained_slat_enc: str = 'microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16', 
        slat_enc_path: Optional[str] = None, 
        slat_enc_ckpt: Optional[str] = None, 
        visual_input_channels: int = 3,
        visual_feat_resolution: int = 256,
        visual_feat_channels: List[int] = [4, 4, 4, 4],
        visual_conv_kernel_size: int = 3,
        view_aggregation: Literal["mean", "concat", "cross_per_view"] = "mean",
        num_view_embeds: int = 0,
        view_embed_random_perm: bool = False,
    ):
        super().__init__(
            resolution,
            in_channels + sum(visual_feat_channels),
            model_channels,
            cond_channels,
            cond_latent_channels,
            out_channels,
            num_blocks,
            num_heads,
            num_head_channels,
            mlp_ratio,
            patch_size,
            num_io_res_blocks,
            io_block_channels,
            pe_mode,
            use_fp16,
            use_checkpoint,
            use_skip_connection,
            share_mod,
            qk_rms_norm,
            qk_rms_norm_cross,
            enable_occlusion_mask,
            occlusion_mask_patcher_args,
            pretrained_slat_enc=pretrained_slat_enc,
            slat_enc_path=slat_enc_path,
            slat_enc_ckpt=slat_enc_ckpt,
            view_aggregation=view_aggregation,
            num_view_embeds=num_view_embeds,
            view_embed_random_perm=view_embed_random_perm,
        )
        self.visual_input_channels = visual_input_channels
        self.visual_feat_resolution = visual_feat_resolution
        self.visual_feat_channels = visual_feat_channels
        self.visual_conv_kernel_size = visual_conv_kernel_size

        self.visual_conv_blocks = self._build_pyramid()

    def _build_pyramid(self) -> nn.Module:
        """
        Build the pyramid module assigned to ``self.visual_conv_blocks``.

        The base version returns an ``nn.ModuleList`` of per-level ``Conv2d``s
        that ``encode_visual_feat`` iterates over. Subclasses may return any
        ``nn.Module`` (e.g. a single composite pyramid) as long as they also
        override ``encode_visual_feat`` to match the new structure.
        """
        in_ch = self.visual_input_channels
        blocks = nn.ModuleList()
        for lvl, out_ch in enumerate(self.visual_feat_channels):
            blocks.append(nn.Conv2d(
                in_ch + int(self.enable_occlusion_mask),
                out_ch,
                kernel_size=self.visual_conv_kernel_size,
                padding=self.visual_conv_kernel_size // 2,
                stride=1 if lvl == 0 else 2,
            ))
            in_ch = out_ch
        return blocks

    def encode_visual_feat(self, uvs: sp.SparseTensor, visual_cond: torch.Tensor, occlusion_mask: Optional[torch.Tensor] = None) -> sp.SparseTensor:
        # Multi-view: fold V into batch for the CNN / interpolate. ``uvs.feats``
        # shape ``[N, V, 2]`` carries the V signal for the sampler.
        if visual_cond.ndim == 5:                                       # [B, V, 3, H, W]
            visual_cond = visual_cond.flatten(0, 1)                     # [B*V, 3, H, W]
        if occlusion_mask is not None and occlusion_mask.ndim == 4:
            occlusion_mask = occlusion_mask.flatten(0, 1)               # [B*V, H, W]
        visual_cond = F.interpolate(
            visual_cond,
            size=(self.visual_feat_resolution, self.visual_feat_resolution),
            mode='bilinear',
            align_corners=False,
            antialias=True,
        )

        visual_feats: List[torch.Tensor] = []
        for conv_block in self.visual_conv_blocks:
            if self.enable_occlusion_mask:
                occlusion_mask_i = F.interpolate(
                    occlusion_mask[:, None, :, :].float(),
                    size=visual_cond.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                    antialias=True,
                )
                visual_cond = conv_block(torch.cat([visual_cond, occlusion_mask_i], dim=1))
            else:
                visual_cond = conv_block(visual_cond)
            visual_feats.append(visual_cond)

        return self._wrap_sparse_taps(visual_feats, uvs)

    def forward(self, x: sp.SparseTensor, t: torch.Tensor, cond: Dict[str, torch.Tensor], uvs: sp.SparseTensor, occlusion_mask: Optional[torch.Tensor] = None, **kwargs) -> sp.SparseTensor:
        latent_cond = self.encode_latent(uvs, cond['dino_cond'], occlusion_mask)
        visual_cond = self.encode_visual_feat(uvs, cond['visual_cond'], occlusion_mask)
        x = x.replace(torch.cat([x.feats, latent_cond.feats, visual_cond.feats], dim=1))
        dino_cond, occlusion_mask = self._aggregate_views(cond['dino_cond'], occlusion_mask)
        h = SLatFlowModel.forward(self, x, t, dino_cond, occlusion_mask, **kwargs)
        return h


# ---------------------------------------------------------------------------
# ConvNeXt-style visual pyramid primitives
# ---------------------------------------------------------------------------


class _LayerNorm2d(nn.Module):
    """``nn.LayerNorm`` over the channel dim of a channels-first 4D tensor."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, H, W, C] -> LN -> [B, C, H, W]
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class _ConvNeXtBlock(nn.Module):
    """
    ConvNeXt V1 block — faithful to the paper's recipe (LN + GELU, fixed):

        DwConv(k, groups=C) → permute CL → LN → Linear(C→mC) → GELU →
        Linear(mC→C) → LayerScale → permute CF → + x.
    """
    def __init__(
        self,
        channels: int,
        mlp_ratio: float = 4.0,
        dwconv_kernel_size: int = 7,
        layer_scale_init: float = 1e-6,
    ):
        super().__init__()
        hidden = int(round(channels * mlp_ratio))
        pad = dwconv_kernel_size // 2
        self.dwconv = nn.Conv2d(channels, channels, dwconv_kernel_size,
                                padding=pad, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.pw1 = nn.Linear(channels, hidden)
        self.pw2 = nn.Linear(hidden, channels)
        self.gamma = (
            nn.Parameter(layer_scale_init * torch.ones(channels))
            if layer_scale_init > 0 else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).contiguous()        # (B,H,W,C)
        x = self.norm(x)
        x = self.pw1(x)
        x = F.gelu(x)
        x = self.pw2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2).contiguous()        # (B,C,H,W)
        return shortcut + x


class _DownsampleLayer(nn.Module):
    """Inter-stage downsampler (ConvNeXt recipe): LN → Conv(k=2, stride=2)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm = _LayerNorm2d(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.norm(x))


class _VisualPyramid(nn.Module):
    """
    ConvNeXt-style multi-scale visual encoder.

    Stem(Conv+LN) → [stage_i: ConvNeXt blocks × N_i + Tap_i + Downsample_{i→i+1}].
    Output is a list of per-stage tap feature maps (channels-first) that the
    caller grid-samples at UV coords.

    Occlusion mask is concatenated either at the stem (``mask_injection="stem"``)
    or at every downsample + stem input (``"per_stage"``). The latter mirrors
    the baseline's "fresh mask at each scale" pattern.
    """
    def __init__(
        self,
        visual_input_channels: int,
        tap_channels: List[int],
        hidden_channels: List[int],
        num_blocks: List[int],
        mlp_ratio: float,
        dwconv_kernel_size: int,
        stem_kernel_size: int,
        stem_stride: int,
        layer_scale_init: float,
        enable_occlusion_mask: bool,
        mask_injection: str,
    ):
        super().__init__()
        assert len(tap_channels) == len(hidden_channels) == len(num_blocks), (
            f"list lengths must match: taps={len(tap_channels)} "
            f"hidden={len(hidden_channels)} blocks={len(num_blocks)}"
        )
        self.num_stages = len(tap_channels)
        self.enable_occlusion_mask = enable_occlusion_mask
        self.mask_injection = mask_injection

        mask_at_stem = enable_occlusion_mask and mask_injection in ("stem", "per_stage")
        self._mask_at_downsample = enable_occlusion_mask and mask_injection == "per_stage"

        # Stem: Conv → LN (channels-first).
        stem_in = visual_input_channels + int(mask_at_stem)
        self.stem = nn.Sequential(
            nn.Conv2d(stem_in, hidden_channels[0], kernel_size=stem_kernel_size,
                      stride=stem_stride,
                      padding=stem_kernel_size // 2),
            _LayerNorm2d(hidden_channels[0]),
        )

        self.stages = nn.ModuleList()
        self.taps = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(self.num_stages):
            C = hidden_channels[i]
            body = nn.Sequential(*[
                _ConvNeXtBlock(
                    channels=C, mlp_ratio=mlp_ratio,
                    dwconv_kernel_size=dwconv_kernel_size,
                    layer_scale_init=layer_scale_init,
                ) for _ in range(num_blocks[i])
            ])
            self.stages.append(body)

            tap_conv = nn.Conv2d(C, tap_channels[i], kernel_size=1)
            self.taps.append(nn.Sequential(_LayerNorm2d(C), tap_conv))

            if i < self.num_stages - 1:
                ds_in = C + int(self._mask_at_downsample)
                self.downsamples.append(_DownsampleLayer(ds_in, hidden_channels[i + 1]))

    def forward(self, visual_cond: torch.Tensor,
                occlusion_mask: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        x = visual_cond
        if self.enable_occlusion_mask and self.mask_injection in ("stem", "per_stage"):
            assert occlusion_mask is not None
            om = F.interpolate(
                occlusion_mask[:, None, :, :].float(), size=x.shape[-2:],
                mode='bilinear', align_corners=False, antialias=True,
            )
            x = torch.cat([x, om], dim=1)
        x = self.stem(x)

        taps: List[torch.Tensor] = []
        for i in range(self.num_stages):
            x = self.stages[i](x)
            taps.append(self.taps[i](x))
            if i < self.num_stages - 1:
                if self._mask_at_downsample:
                    om_i = F.interpolate(
                        occlusion_mask[:, None, :, :].float(), size=x.shape[-2:],
                        mode='bilinear', align_corners=False, antialias=True,
                    )
                    x = torch.cat([x, om_i], dim=1)
                x = self.downsamples[i](x)
        return taps


def _broadcast_to_list(x: Union[int, List[int]], length: int, name: str) -> List[int]:
    if isinstance(x, int):
        return [x] * length
    if isinstance(x, (list, tuple)):
        if len(x) != length:
            raise ValueError(f"{name} has length {len(x)}, expected {length}")
        return list(x)
    raise TypeError(f"{name} must be int or list[int], got {type(x).__name__}")


class ConvNeXtVisualLatentConditioningSLatFlowModel(VisualLatentConditioningSLatFlowModel):
    """
    Visual latent conditioning with a ConvNeXt-style multi-scale encoder.

    The per-stage tap width (``visual_feat_channels``, inherited) still sets the
    low-dim output that gets grid-sampled and concatenated onto ``x.feats``; the
    internal working width (``hidden_channels``) is decoupled so the pyramid can
    be wide without bloating the DiT input. Occlusion mask injection, norm, and
    activation are all configurable.
    """

    def __init__(
        self,
        *args,
        hidden_channels: Union[int, List[int]] = 32,
        num_pyramid_blocks: Union[int, List[int]] = 1,
        pyramid_mlp_ratio: float = 4.0,
        dwconv_kernel_size: int = 7,
        stem_kernel_size: int = 3,
        stem_stride: int = 1,
        layer_scale_init: float = 1e-6,
        mask_injection: Literal["stem", "per_stage"] = "stem",
        **kwargs,
    ):
        # Stash cfg BEFORE super().__init__ so _build_pyramid (invoked by
        # parent via dynamic dispatch) can read it. NB: ``num_pyramid_blocks``
        # is distinct from the DiT's ``num_blocks`` which is forwarded through
        # *args/**kwargs to the parent.
        self._convnext_cfg = dict(
            hidden_channels=hidden_channels,
            num_pyramid_blocks=num_pyramid_blocks,
            pyramid_mlp_ratio=pyramid_mlp_ratio,
            dwconv_kernel_size=dwconv_kernel_size,
            stem_kernel_size=stem_kernel_size,
            stem_stride=stem_stride,
            layer_scale_init=layer_scale_init,
            mask_injection=mask_injection,
        )
        super().__init__(*args, **kwargs)

    def _build_pyramid(self) -> nn.Module:
        cfg = self._convnext_cfg
        L = len(self.visual_feat_channels)
        hidden = _broadcast_to_list(cfg['hidden_channels'], L, "hidden_channels")
        num_pyramid_blocks = _broadcast_to_list(
            cfg['num_pyramid_blocks'], L, "num_pyramid_blocks")
        return _VisualPyramid(
            visual_input_channels=self.visual_input_channels,
            tap_channels=list(self.visual_feat_channels),
            hidden_channels=hidden,
            num_blocks=num_pyramid_blocks,
            mlp_ratio=cfg['pyramid_mlp_ratio'],
            dwconv_kernel_size=cfg['dwconv_kernel_size'],
            stem_kernel_size=cfg['stem_kernel_size'],
            stem_stride=cfg['stem_stride'],
            layer_scale_init=cfg['layer_scale_init'],
            enable_occlusion_mask=self.enable_occlusion_mask,
            mask_injection=cfg['mask_injection'],
        )

    def encode_visual_feat(self, uvs: sp.SparseTensor, visual_cond: torch.Tensor,
                           occlusion_mask: Optional[torch.Tensor] = None) -> sp.SparseTensor:
        if visual_cond.ndim == 5:                                       # [B, V, 3, H, W]
            visual_cond = visual_cond.flatten(0, 1)                     # [B*V, 3, H, W]
        if occlusion_mask is not None and occlusion_mask.ndim == 4:
            occlusion_mask = occlusion_mask.flatten(0, 1)               # [B*V, H, W]
        visual_cond = F.interpolate(
            visual_cond,
            size=(self.visual_feat_resolution, self.visual_feat_resolution),
            mode='bilinear', align_corners=False, antialias=True,
        )
        taps = self.visual_conv_blocks(visual_cond, occlusion_mask if self.enable_occlusion_mask else None)
        return self._wrap_sparse_taps(taps, uvs)
