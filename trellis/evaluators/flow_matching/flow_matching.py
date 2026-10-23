from typing import *
import json
import re
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from ..base import Evaluator
from ..alignment_mixin import AlignmentMixin
from ...utils.pose_utils import *
from ...utils.align_utils import align_extrinsics_by_similarity
from ...utils.voxel_utils import dilate_binary_volume, expand_binary_volume
from ...trainers.flow_matching.mixins.flow_matching_core import FlowMatchingCoreMixin
from ...trainers.flow_matching.mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from ...pipelines import samplers

_PER_VIEW_COL_RE = re.compile(r'_v\d+$')
_PER_VIEW_COL_BASE_RE = re.compile(r'(.+)_v\d+$')
_PER_STEP_COL_RE = re.compile(r'_step\d+$')
_PER_STEP_COL_BASE_RE = re.compile(r'(.+)_step(\d+)$')
_T_STEP_COL_RE = re.compile(r'^t_step\d+$')


class FlowMatchingEvaluator(FlowMatchingCoreMixin, AlignmentMixin, Evaluator):
    """
    Base evaluator for Flow Matching latent model with Sparse UV structure decoder.

    Mirrors the metric computations from the VAE evaluator, while the latent can be obtained by two methods:
    1. Timestep Sampling and diffuse from GT latent:
      - sampling a timestep t, diffusing z_0 -> z_t,
      - predicting velocity v̂ with the denoiser,
      - reconstructing ẑ_0 using the flow formula,
      - decoding ẑ_0 to logits.
    2. Single-view Guidance:
      - Denoise from pure noise by stepping t from 1 to 0 with a noise schedule.
      - decoding the final clean latent to logits.

    Args:
        models (dict[str, nn.Module]): Expected to contain 'denoiser' and 'decoder'.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        load_step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        fp16_mode (str): FP16 mode.
        i_sample (int): Sample interval.

        predict_ss_from_ssuv (bool): Whether to predict the sparse 3D structure from the sparse UV structure. Default is False.
        predict_ssuv_from_ss (bool): Whether to predict the sparse UV structure from the sparse 3D structure. Default is False.
        evaluate_pose (bool): Whether to evaluate the pose related metrics. Default is True.
        dilation_sigma (float): Standard deviation of Gaussian kernel for post-processing dilation. Default is 0.
        dilation_min_weight (float): Minimum weight to include in the dilated grid. Default is 0.05.
        ssuv_expand_min_num_activated (int): Minimum number of activated voxels for the ssuv_recon. Default is 10.
        ssuv_expand_threshold_step (float): Step size for the ssuv_recon expansion threshold. Default is 0.05.

        t_schedule (dict): Time schedule for t sampling. Matches the trainer.
        sigma_min (float): Minimum noise level used in flow diffusion.
        num_t_buckets (int): Number of t buckets. Default is 10.
        sample_ssuv (bool | 'gt_latent'): Sampling mode for ssuv predictions. False → teacher-forced from sampled t (default); True → full sampling from pure noise; 'gt_latent' → decode GT x_0 directly without any denoising.
        ssuv_sampler_args (dict): Arguments for the ssuv sampler (only used when sample_ssuv is True).
    """

    def __init__(
        self,
        *args,
        predict_ss_from_ssuv: bool = False,
        predict_ssuv_from_ss: bool = False,
        evaluate_pose: bool = True,
        dilation_sigma: float = 0.0,
        dilation_min_weight: float = 0.05,
        ssuv_expand_min_num_activated: int = 10,
        ssuv_expand_threshold_step: float = 0.05,
        t_schedule: dict = {'name': 'uniform', 'args': {}},
        sigma_min: float = 1e-5,
        num_t_buckets: int = 10,
        sample_ssuv: Union[bool, Literal['gt_latent']] = False,
        ssuv_sampler_args: Optional[dict] = None,
        save_pose: bool = False,
        use_gt_intrinsics: bool = False,
        eval_intermediate_steps: Union[bool, List[int]] = False,
        save_debug: bool = False,
        save_debug_topk: int = 128,
        alignment_method: Literal["none", "trajectory", "input_pointmap", "full_pointcloud"] = "none",
        pointmap_keep_count: Optional[int] = None,
        pose_ambiguity: bool = False,
        denoise_test_view_poses: bool = False,
        num_input_views_in_joint: int = 0,
        test_view_chunk_size: Optional[int] = None,
        pose_only_sampler_args: Optional[dict] = None,
        **kwargs
    ):
        # Sparse-only / rendering kwargs from SparseFlowMatchingEvaluator configs
        # flow through **kwargs and are absorbed by Evaluator.__init__(**kwargs).
        if save_debug:
            # DebugSaver assembles per-sample bundles and recompute attn weights
            # in forward hooks; B>1 would interleave samples in capture buffers.
            kwargs['batch_size_per_gpu'] = 1
            kwargs.pop('batch_size', None)
        super().__init__(*args, t_schedule=t_schedule, sigma_min=sigma_min, **kwargs)
        assert alignment_method in {"none", "trajectory", "input_pointmap", "full_pointcloud"}, \
            f"unknown alignment_method={alignment_method!r}"
        self.alignment_method = alignment_method
        self.pointmap_keep_count = (
            None if pointmap_keep_count is None else int(pointmap_keep_count))
        self.pose_ambiguity = bool(pose_ambiguity)
        self.predict_ss_from_ssuv = predict_ss_from_ssuv
        self.predict_ssuv_from_ss = predict_ssuv_from_ss
        self.evaluate_pose = evaluate_pose
        self.dilation_sigma = dilation_sigma
        self.dilation_min_weight = dilation_min_weight
        self.ssuv_expand_min_num_activated = ssuv_expand_min_num_activated
        self.ssuv_expand_threshold_step = ssuv_expand_threshold_step
        self.num_t_buckets = num_t_buckets
        self._xyz_voxels = None
        self.sample_ssuv = sample_ssuv
        self.ssuv_sampler = getattr(samplers, ssuv_sampler_args['name'])(**ssuv_sampler_args['args']) if sample_ssuv is True else None
        self.ssuv_sampler_params = ssuv_sampler_args['params'] if sample_ssuv is True else {}
        self.save_pose = save_pose
        self.use_gt_intrinsics = use_gt_intrinsics
        self.eval_intermediate_steps = eval_intermediate_steps

        # --- Debug saver (lazy: only built when save_debug=True) ---
        # Lives in debug_saver.py; subclasses pick the right class via
        # ``_debug_saver_cls`` (sparse override returns SparseDebugSaver).
        self.save_debug = save_debug
        self.debug_saver = None
        if save_debug:
            self.debug_saver = self._debug_saver_cls()(self, topk=save_debug_topk)
            self.debug_saver.validate()
            self.debug_saver.setup_hooks()

        # --- Pose-only test-view denoising (lazy: only built when enabled) ---
        # When True, after sampling shape latent x_pred, additionally denoise
        # pose tokens for test views with x pinned clean (t_x=0). The resulting
        # extrinsics live in pred-world by construction, so test-view rendering
        # / metrics need NO alignment fit. See ``_sample_test_view_poses``.
        # ``num_input_views_in_joint`` (K): how many input-view pose latents to
        # pin clean alongside the test slots in each pose-only sampler call.
        # K=0 ≡ test-only; K=V_input ≡ full joint; intermediate K mixes some.
        assert num_input_views_in_joint >= 0, \
            f"num_input_views_in_joint must be >= 0, got {num_input_views_in_joint}"
        self.denoise_test_view_poses = denoise_test_view_poses
        self.num_input_views_in_joint = num_input_views_in_joint
        self.test_view_chunk_size = test_view_chunk_size
        if denoise_test_view_poses:
            assert pose_only_sampler_args is not None, \
                "pose_only_sampler_args is required when denoise_test_view_poses=True"
            self.pose_only_sampler = getattr(samplers, pose_only_sampler_args['name'])(
                **pose_only_sampler_args['args'])
            self.pose_only_sampler_params = pose_only_sampler_args.get('params', {})
        else:
            self.pose_only_sampler = None
            self.pose_only_sampler_params = {}

    @staticmethod
    def _debug_saver_cls():
        """Subclass hook so SparseFlowMatchingEvaluator returns SparseDebugSaver."""
        from .debug_saver import DebugSaver
        return DebugSaver

    def evaluation_metrics(self, indices, **data):
        """
        Compute evaluation metrics using flow reconstruction pipeline.

        Args:
            indices: [N] tensor of indices for this batch.
            **data: May contain keys identical to the VAE evaluator:
                - 'x_0', 'cond', 'ss', 'ssuv', 'uv_volume', 'uv_weight', 'depth_volume', 'intrinsics', 'extrinsics'.
        Returns:
            Tuple[metrics_dict, samples_dict]
        """
        metrics, samples = {}, {}
        step_payload = None

        if self.sample_ssuv == 'gt_latent':
            preds = self._decode_ssuv(data['x_0'])
        elif self.sample_ssuv is True:
            preds, step_payload = self._sample_ssuv(**data)
        else:
            t = self.sample_t(self.batch_size_per_gpu).to(self.device).float()
            preds = self._sample_ssuv_from_t(t, **data)
            metrics['t'] = t

        m, s = self._compute_metrics_from_preds(preds, data, indices, suffix='')
        metrics.update(m)
        samples.update(s)

        if step_payload is not None:
            N = data['x_0'].shape[0]
            for step_i, t_i, x0_i in step_payload:
                preds_i = self._decode_ssuv(x0_i)
                m_i, _ = self._compute_metrics_from_preds(preds_i, data, indices, suffix=f'_step{step_i}')
                metrics.update(m_i)
                metrics[f't_step{step_i}'] = torch.full((N,), float(t_i), device=self.device)

        return metrics, samples

    def _compute_metrics_from_preds(self, preds, data, indices, suffix=''):
        """Compute occupancy/UV/pose metrics. With non-empty ``suffix`` every
        metric key is suffixed, pose files are merged, samples are suppressed.
        """
        metrics, samples = {}, {}
        has_suv_decoder = 'suv_decoder' in self.models
        emit_samples = (suffix == '')

        ss_metrics, ss_samples = self._evaluate_occupancy('ss', data['ss'], preds['ss_logits'], preds['ss_recon'])
        metrics.update({k + suffix: v for k, v in ss_metrics.items()})
        if emit_samples:
            samples.update(ss_samples)

        if has_suv_decoder and not self.predict_ssuv_from_ss:
            ssuv_metrics, ssuv_samples = self._evaluate_occupancy('ssuv', data['ssuv'], preds['ssuv_logits'], preds['ssuv_recon'])
            metrics.update({k + suffix: v for k, v in ssuv_metrics.items()})
            if emit_samples:
                samples.update(ssuv_samples)

        if has_suv_decoder:
            uv_metrics, uv_samples = self._evaluate_uv_volume(**data, **preds)
            metrics.update({k + suffix: v for k, v in uv_metrics.items()})
            if emit_samples:
                samples.update(uv_samples)

        if self.evaluate_pose:
            pose_metrics, pose_samples = self._evaluate_poses(indices=indices, pose_suffix=suffix, **data, **preds)
            metrics.update({k + suffix: v for k, v in pose_metrics.items()})
            if emit_samples:
                samples.update(pose_samples)

        return metrics, samples

    def _decode_ssuv(self, pred_x_0: torch.Tensor):
        # Decode to prediction logits. Layout of pred_x_0 channels:
        #   [0:ss_latent_dim)                        → ss_decoder latent
        #   [ss_latent_dim:ss_latent_dim+suv_latent) → suv_decoder latent (if present)
        ss_latent_dim = self.models['ss_decoder'].latent_channels
        has_suv_decoder = 'suv_decoder' in self.models

        if has_suv_decoder:
            suv_latent_dim = self.models['suv_decoder'].latent_channels
            suv_logits = self.models['suv_decoder'](pred_x_0[:, ss_latent_dim:ss_latent_dim+suv_latent_dim, :, :, :])

        if self.predict_ss_from_ssuv:
            assert has_suv_decoder, 'predict_ss_from_ssuv requires suv_decoder'
            ss_logits = suv_logits[:, :1, :, :, :]
        else:
            ss_logits = self.models['ss_decoder'](pred_x_0[:, :ss_latent_dim, :, :, :])
        ss_recon = (ss_logits > 0).int()

        if not has_suv_decoder:
            return {
                'ss_logits': ss_logits,
                'ss_recon': ss_recon,
                'ssuv_logits': None,
                'ssuv_recon': None,
                'uv_volume_pred': None,
            }

        ssuv_logits, suv_logits = suv_logits[:, :1, :, :, :], suv_logits[:, 1:, :, :, :]
        if self.predict_ssuv_from_ss:
            ssuv_logits = ss_logits
        # Always expand ssuv_recon to guarantee enough active voxels for DLT
        # pose solving, regardless of where ssuv_logits came from.
        ssuv_recon = expand_binary_volume(
            logits=ssuv_logits,
            min_num_activated=self.ssuv_expand_min_num_activated,
            threshold_step=self.ssuv_expand_threshold_step,
        )

        uv_volume_pred = torch.sigmoid(suv_logits[:, :2, :, :, :])

        return {
            'ss_logits': ss_logits,
            'ss_recon': ss_recon,
            'ssuv_logits': ssuv_logits,
            'ssuv_recon': ssuv_recon,
            'uv_volume_pred': uv_volume_pred,
        }

    def _sample_ssuv_from_t(self, t: torch.Tensor, x_0: torch.Tensor, cond: torch.Tensor, **kwargs):
        """
        Obtain predictions by reconstructing latent with the flow denoiser and decoding.

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the outputs of the decoder.
        """
        # Flow: diffuse x_0 -> x_t, predict v, reconstruct x̂_0
        noise = torch.randn_like(x_0)
        x_t = self.diffuse(x_0, t, noise)
        
        # Predict velocity v̂ with denoiser
        args = self.get_inference_cond(cond, **kwargs)
        pred_v = self.models['denoiser'](x_t, t * 1000, **args)
        assert pred_v.shape == noise.shape == x_0.shape
        
        # Reconstruct original latent using flow formula: x̂_0 = x_t - t * v̂
        pred_x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t[:, None, None, None, None]) * pred_v

        return self._decode_ssuv(pred_x_0)

    def _sample_ssuv(self, x_0: torch.Tensor, cond: torch.Tensor, **kwargs):
        """Full-sampling predictions. Returns ``(final_preds, step_payload)``;
        ``step_payload`` is ``None`` when ``eval_intermediate_steps`` is off.
        """
        args = self.get_inference_cond(cond, **kwargs)
        sampling_noise = torch.randn_like(x_0)
        result = self.ssuv_sampler.sample(
            model=self.models['denoiser'],
            noise=sampling_noise,
            **args,
            **self.ssuv_sampler_params,
            verbose=False,
        )
        pred_x_0 = result.samples
        final_preds = self._decode_ssuv(pred_x_0)
        step_payload = self._build_step_payload(result, ('pred_x_0',))
        return final_preds, step_payload

    def _sampler_t_seq(self) -> np.ndarray:
        """Reconstruct the sampler's ``t`` schedule (length ``steps+1``)."""
        params = self.ssuv_sampler_params or {}
        steps = int(params.get('steps', 50))
        rescale_t = float(params.get('rescale_t', 1.0))
        t_seq = np.linspace(1.0, 0.0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        return t_seq

    def _build_step_payload(self, result, keys):
        """Gather per-step ``(step_idx, t_value, *tensors)`` from sampler result
        for the keys named in ``keys`` (e.g. ``('pred_x_0',)``). ``None`` when off.
        """
        flag = self.eval_intermediate_steps
        if not flag:
            return None
        lst = result.get(keys[0], None)
        if lst is None:
            return None
        num_steps = len(lst)
        if flag is True:
            indices = list(range(num_steps))
        else:
            indices = sorted({int(i) for i in flag if 0 <= int(i) < num_steps})
        t_seq = self._sampler_t_seq()
        payload = []
        for i in indices:
            tensors = tuple(result[k][i] for k in keys)
            t_value = float(t_seq[i]) if i < len(t_seq) else float('nan')
            payload.append((i, t_value, *tensors))
        return payload

    def _evaluate_occupancy(self, prefix, occupancy, occupancy_logits, occupancy_recon):
        """Evaluate the occupancy of sparse structure metrics."""
        metrics = {}
        samples = {}
        
        # BCE loss for occupancy
        bce_loss = F.binary_cross_entropy_with_logits(occupancy_logits, occupancy.float(), reduction='none')
        bce_loss = bce_loss.mean(dim=[2, 3, 4])  # Average over spatial dimensions
        metrics[f'{prefix}_bce_loss'] = bce_loss.squeeze(1)  # [batch_size]
        
        # Binary accuracy and IoU (Intersection over Union)
        accuracy = (occupancy_recon == occupancy).float().mean(dim=[2, 3, 4])
        metrics[f'{prefix}_accuracy'] = accuracy.squeeze(1)  # [batch_size]
        intersection = (occupancy_recon * occupancy).sum(dim=[2, 3, 4])
        union = (occupancy_recon + occupancy).clamp(0, 1).sum(dim=[2, 3, 4])
        iou = (intersection + 1e-8) / (union + 1e-8)  # Add small epsilon to avoid division by zero
        metrics[f'{prefix}_iou'] = iou.squeeze(1)  # [batch_size]
        
        # Ratio of activated occupancy voxels (occupancy_recon = True)
        activated_count = occupancy_recon.sum(dim=[2, 3, 4]).squeeze(1)  # [N]
        activated_count_gt = occupancy.sum(dim=[2, 3, 4]).squeeze(1)  # [N]
        total_voxels = occupancy.numel() // occupancy.shape[0]  # Total voxels per sample
        activate_ratio = activated_count.float() / total_voxels
        activate_ratio_gt = activated_count_gt.float() / total_voxels
        metrics[f'{prefix}_activate_ratio'] = activate_ratio  # [batch_size]
        metrics[f'{prefix}_activate_ratio_gt'] = activate_ratio_gt  # [batch_size]
        metrics[f'{prefix}_activate_ratio_diff'] = (activate_ratio - activate_ratio_gt).abs()  # [batch_size]
        
        # Samples for visualization
        samples[f'{prefix}_gt'] = {'value': occupancy, 'type': 'sample'}
        samples[f'{prefix}_recon'] = {'value': occupancy_recon, 'type': 'sample'}
        samples[f'{prefix}_diff'] = {'value': (occupancy_recon - occupancy).abs(), 'type': 'sample'}
        
        return metrics, samples

    def _evaluate_uv_volume(self, uv_volume, ssuv, uv_volume_pred, **kwargs):
        """Evaluate UV volume metrics."""
        metrics = {}
        samples = {}
        
        ssuv_sum = ssuv.sum(dim=[1, 2, 3, 4])  # [N]
        
        # L1 and MSE loss for UV volume (weighted by ground truth uv weight)
        uv_l1_loss = F.l1_loss(uv_volume_pred, uv_volume, reduction='none')
        uv_mse_loss = F.mse_loss(uv_volume_pred, uv_volume, reduction='none')
        metrics['uv_l1_loss'] = (uv_l1_loss * ssuv).sum(dim=[1, 2, 3, 4]) / ssuv_sum
        metrics['uv_mse_loss'] = (uv_mse_loss * ssuv).sum(dim=[1, 2, 3, 4]) / ssuv_sum

        # UV distance (weighted by ground truth uv weight)
        uv_distance = torch.norm(uv_volume - uv_volume_pred, dim=1, keepdim=True)
        metrics['uv_distance'] = (uv_distance * ssuv).sum(dim=[1, 2, 3, 4]) / ssuv_sum
        
        # Samples for visualization
        samples['uv_volume_gt'] = {'value': {'uv_volume': uv_volume, 'ssuv': ssuv}, 'type': 'sample'}
        samples['uv_volume_pred'] = {'value': {'uv_volume': uv_volume_pred, 'ssuv': ssuv}, 'type': 'sample'}
        
        # UV volume difference visualization
        uv_volume_diff = (uv_volume - uv_volume_pred).norm(dim=1, keepdim=True)
        uv_volume_diff = uv_volume_diff / uv_volume_diff.max().clamp_min_(1e-2)
        samples['uv_volume_diff'] = {'value': {'uv_volume': uv_volume_diff, 'ssuv': ssuv}, 'type': 'sample'}
        
        return metrics, samples 

    def _get_xyz_voxel_grid(self, shape: Tuple[int, int, int], device: torch.device) -> torch.Tensor:
        """Lazily build and cache the centered voxel xyz grid [1, 3, R*R*R] in [-0.5, 0.5)."""
        if self._xyz_voxels is None:
            D, H, W = shape
            x = (torch.arange(W, dtype=torch.float32, device=device) + 0.5) / W - 0.5
            y = (torch.arange(H, dtype=torch.float32, device=device) + 0.5) / H - 0.5
            z = (torch.arange(D, dtype=torch.float32, device=device) + 0.5) / D - 0.5
            xyz_voxels = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=0)  # [3, D, H, W]
            self._xyz_voxels = xyz_voxels.view(1, 3, -1)  # [1, 3, D*H*W]
        return self._xyz_voxels

    def _solve_pose_from_ssuv(self, ssuv_recon, uv_volume_pred, intrinsics_gt=None, **_) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve per-sample camera pose from uv_volume + ssuv mask.

        Uses linear DLT by default; when `self.use_gt_intrinsics` is True and
        `intrinsics_gt` is provided, solves PnP with known K instead, bypassing
        the fx/t_z near-ambiguity in full DLT + RQ decomposition.

        Args:
            ssuv_recon: [N, 1, D, H, W] binary mask.
            uv_volume_pred: [N, 2, D, H, W] uv volume.
            intrinsics_gt: Optional [N, 3, 3] GT intrinsics (cropped frame,
                matching uv_volume_pred) used when `self.use_gt_intrinsics`.

        Returns:
            extrinsics: [N, 4, 4] homogeneous RT matrices.
            intrinsics: [N, 3, 3] K matrices.
            reproj: [N] reprojection errors.
        """
        D, H, W = uv_volume_pred.shape[-3:]
        xyz_voxels = self._get_xyz_voxel_grid((D, H, W), uv_volume_pred.device)

        uv_voxels = uv_volume_pred.flatten(start_dim=2)        # [N, 2, D*H*W]
        voxel_weights = ssuv_recon.flatten(start_dim=1)        # [N, D*H*W]

        use_known_k = self.use_gt_intrinsics and intrinsics_gt is not None

        extrinsics_list = []
        intrinsics_list = []
        reproj_list = []
        for i in range(uv_voxels.shape[0]):
            valid_voxel = (voxel_weights[i] > 0)
            if self.dilation_sigma > 0:
                valid_voxel = dilate_binary_volume(
                    valid_voxel.view(D, H, W),
                    sigma=self.dilation_sigma,
                    min_weight=self.dilation_min_weight,
                ).flatten()
            xyzs = xyz_voxels[:, :, valid_voxel].permute(0, 2, 1).contiguous()
            uvs = uv_voxels[i:i+1, :, valid_voxel].permute(0, 2, 1).contiguous()
            if use_known_k:
                K = intrinsics_gt[i:i+1]
                RT, RE = solve_pnp_known_k(xyzs, uvs, K)
            else:
                K, RT, RE = calibrate_camera_dlt(xyzs, uvs)
            intrinsics_list.append(K)
            extrinsics_list.append(RT)
            reproj_list.append(RE)

        extrinsics = torch.concat(extrinsics_list, dim=0)
        intrinsics = torch.concat(intrinsics_list, dim=0)
        reproj = torch.stack(reproj_list)
        # Extend to 4x4 homogeneous extrinsics
        pad_row = extrinsics.new_zeros(extrinsics.shape[0], 1, 4)
        pad_row[:, 0, 3] = 1.0
        extrinsics = torch.cat([extrinsics, pad_row], dim=1)
        return extrinsics, intrinsics, reproj

    def _adjust_raw_intrinsics(self, intrinsics: torch.Tensor, crop_bbox: Optional[torch.Tensor]) -> torch.Tensor:
        """Map normalized intrinsics from a cropped image back to the raw image."""
        if crop_bbox is None:
            return intrinsics
        fov_scale = crop_bbox[:, 1] - crop_bbox[:, 0]
        raw_intrinsics = intrinsics.clone()
        raw_intrinsics[:, 0, 0] = raw_intrinsics[:, 0, 0] * fov_scale[:, 0]
        raw_intrinsics[:, 0, 2] = raw_intrinsics[:, 0, 2] * fov_scale[:, 0] + crop_bbox[:, 0, 0]
        raw_intrinsics[:, 1, 1] = raw_intrinsics[:, 1, 1] * fov_scale[:, 1]
        raw_intrinsics[:, 1, 2] = raw_intrinsics[:, 1, 2] * fov_scale[:, 1] + crop_bbox[:, 0, 1]
        return raw_intrinsics

    def _evaluate_poses(self, uv_volume, ss, ssuv, uv_volume_pred, ssuv_recon, intrinsics, extrinsics, indices=None, pose_suffix='', **kwargs):
        """Evaluate transforms metrics."""
        metrics = {}
        samples = {}

        ext_pred, K_pred, reproj_pred = self._solve_pose_from_ssuv(ssuv_recon, uv_volume_pred, intrinsics_gt=intrinsics)
        pose_blocks = [('', ext_pred, K_pred, reproj_pred)]
        # Oracle must mirror the pred path's mask source; under predict_ssuv_from_ss
        # the pred uses ss_logits, and GT `ssuv` (frustum mask when const_ssuv=True)
        # would include out-of-crop voxels whose uv_volume got clamped — DLT garbage.
        # Oracle is skipped when GT 3D data is absent.
        has_oracle_gt = uv_volume is not None and ss is not None and ssuv is not None
        if has_oracle_gt:
            oracle_mask = ss if self.predict_ssuv_from_ss else ssuv
            ext_oracle, K_oracle, reproj_oracle = self._solve_pose_from_ssuv(oracle_mask, uv_volume, intrinsics_gt=intrinsics)
            pose_blocks.append(('oracle_', ext_oracle, K_oracle, reproj_oracle))

        for prefix, ext, K, reproj in pose_blocks:
            for k, v in compute_pose_diff_metrics(
                ext, K, extrinsics, intrinsics, ambiguity=self.pose_ambiguity,
            ).items():
                metrics[f'{prefix}{k}'] = v
            metrics[f'{prefix}reproj'] = reproj.reshape(-1).clamp_max(1.0)

        if self.save_pose:
            kw = {f'extrinsics_pred{pose_suffix}': ext_pred,
                  f'intrinsics_pred{pose_suffix}': K_pred}
            if pose_suffix == '':
                kw['extrinsics_gt'] = extrinsics
                kw['intrinsics_gt'] = intrinsics
            self.save_poses('poses', indices, **kw)

        return metrics, samples

    def get_extra_metrics_statistics(self, df):
        """
        Get extra metrics statistics: t-bucket breakdown and per-step breakdown.
        """
        result = {}

        # T-bucket stats (teacher-forced mode only)
        if self.sample_ssuv is not True and self.sample_ssuv != 'gt_latent' and 't' in df.columns:
            # Exclude positional per-view columns (_v0, _v1, ...) from t-bucket stats
            metric_columns = [c for c in df.columns
                              if c not in self._METADATA_COLUMNS and not _PER_VIEW_COL_RE.search(c)]

            t_bins = np.linspace(0, 1, self.num_t_buckets + 1)
            t_labels = [f"t_{t_bins[i]:.2f}-{t_bins[i+1]:.2f}" for i in range(self.num_t_buckets)]
            df['t_bucket'] = pd.cut(df['t'], bins=t_bins, labels=t_labels, include_lowest=True)

            statistics_by_t_bucket = []
            for i, bucket_label in enumerate(t_labels):
                bucket_df = df[df['t_bucket'] == bucket_label]
                if len(bucket_df) == 0:
                    continue
                statistics_by_t_bucket.append({
                    't_bucket': bucket_label,
                    't_range': [float(t_bins[i]), float(t_bins[i+1])],
                    'sample_count': len(bucket_df),
                    'statistics': {m: self._compute_column_stats(bucket_df[m]) for m in metric_columns},
                })

            if self.is_master:
                self._create_t_bucket_visualization(statistics_by_t_bucket)
            result['statistics_by_t_bucket'] = statistics_by_t_bucket

        # Per-step stats (only when intermediate-step eval was enabled)
        stats_by_step = self._compute_statistics_by_step(df)
        if stats_by_step:
            if self.is_master:
                self._create_t_bucket_visualization(
                    stats_by_step,
                    output_filename='stats_by_steps.png',
                    title='Metrics Visualization with Quantile Error Bars Across Sampling Steps',
                    xlabel='Sampling step (t range)',
                )
            result['statistics_by_step'] = stats_by_step

        return result

    def _compute_statistics_by_step(self, df):
        """Group ``_step{i}`` columns into a stats_by_t_bucket-shaped list;
        ``[]`` when no per-step columns are present (flag-off path).
        """
        bases_per_step = {}  # step_i -> {base -> col_name}
        for c in df.columns:
            if _T_STEP_COL_RE.match(c):
                continue  # t_step{i} is the step's t value, not a metric to aggregate
            m = _PER_STEP_COL_BASE_RE.match(c)
            if not m:
                continue
            base, idx = m.group(1), int(m.group(2))
            # Skip per-view step columns (base ends with _v\d+); they're not aggregated here.
            if _PER_VIEW_COL_RE.search(base):
                continue
            bases_per_step.setdefault(idx, {})[base] = c

        if not bases_per_step:
            return []

        # Try to recover the sampler's t schedule for t_range. Fall back to [t, t].
        try:
            t_seq = self._sampler_t_seq()
        except Exception:
            t_seq = None

        statistics_by_step = []
        for step_i in sorted(bases_per_step.keys()):
            col_map = bases_per_step[step_i]
            t_col = f't_step{step_i}'
            t_value = float(df[t_col].iloc[0]) if t_col in df.columns and len(df[t_col]) > 0 else float('nan')

            if t_seq is not None and 0 <= step_i + 1 < len(t_seq):
                t_range = [float(t_seq[step_i + 1]), float(t_seq[step_i])]
            else:
                t_range = [t_value, t_value]

            stats = {base: self._compute_column_stats(df[col]) for base, col in col_map.items()}
            statistics_by_step.append({
                't_bucket': f'step_{step_i}',
                'step': step_i,
                't_range': t_range,
                't': t_value,
                'sample_count': int(len(df[next(iter(col_map.values()))].dropna())),
                'statistics': stats,
            })
        return statistics_by_step

    def postprocess_metrics_statistics(self, metrics_stats):
        """Drop per-step and per-step-t columns from the top-level statistics."""
        return {k: v for k, v in metrics_stats.items()
                if not _PER_STEP_COL_RE.search(k) and not _T_STEP_COL_RE.match(k)}

    def _create_t_bucket_visualization(
        self,
        statistics_by_t_bucket,
        output_filename: str = 'stats_by_t_buckets.png',
        title: str = 'Metrics Visualization with Quantile Error Bars Across T Buckets',
        xlabel: str = 'T Bucket Range',
    ):
        """
        Create visualization for each metric across t buckets.
        """
        if not statistics_by_t_bucket:
            return
        
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        
        # Get metric names from the first bucket
        first_bucket = statistics_by_t_bucket[0]
        metric_names = list(first_bucket['statistics'].keys())
        
        # Extract t bucket information
        t_buckets = []
        t_ranges = []
        for bucket in statistics_by_t_bucket:
            t_buckets.append(bucket['t_bucket'])
            t_ranges.append(bucket['t_range'])
        
        # Calculate grid dimensions
        n_metrics = len(metric_names)
        n_cols = 4  # You can adjust this
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        # Create figure and subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        fig.suptitle(title, fontsize=18, y=0.95)
        
        # Flatten axes if it's a 2D array
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Plot each metric
        for i, metric_name in enumerate(metric_names):
            ax = axes[i]
            
            # Extract data for this metric
            means = []
            quantiles_25 = []
            quantiles_50 = []
            quantiles_75 = []
            quantiles_95 = []
            
            for bucket in statistics_by_t_bucket:
                stats = bucket['statistics'][metric_name]
                means.append(stats['mean'])
                quantiles_25.append(stats['25%'])
                quantiles_50.append(stats['50%'])
                quantiles_75.append(stats['75%'])
                quantiles_95.append(stats['95%'])
            
            # Create x-axis labels (t bucket ranges)
            x_labels = [f"{bucket['t_range'][0]:.2f}-{bucket['t_range'][1]:.2f}" 
                       for bucket in statistics_by_t_bucket]
            x_positions = np.arange(len(x_labels))
            
            # Plot the main line with mean values
            line, = ax.plot(x_positions, means, 'o-', linewidth=2, markersize=6, 
                           color='blue', label='Mean', zorder=3)
            
            # Add quantile error bars as three separate colored bars
            for j, (x, y_mean) in enumerate(zip(x_positions, means)):
                q25 = quantiles_25[j]
                q50 = quantiles_50[j]
                q75 = quantiles_75[j]
                q95 = quantiles_95[j]
                
                # Plot 25%-50% range (light green)
                ax.vlines(x, q25, q50, color='lightgreen', alpha=0.8, linewidth=6, 
                         label='25%-50% range' if j == 0 else "")
                
                # Plot 50%-75% range (orange)
                ax.vlines(x, q50, q75, color='orange', alpha=0.8, linewidth=6,
                         label='50%-75% range' if j == 0 else "")
                
                # Plot 75%-95% range (red)
                ax.vlines(x, q75, q95, color='red', alpha=0.8, linewidth=6,
                         label='75%-95% range' if j == 0 else "")
                
                # Plot median as a point
                ax.plot(x, q50, 's', color='darkblue', markersize=4, 
                       label='Median' if j == 0 else "", zorder=4)
            
            ax.set_title(metric_name, fontsize=12, pad=15)
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel('Value (log scale)', fontsize=11)
            ax.set_yscale('log')  # Set logarithmic scale for y-axis
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=10)
            
            # Set x-axis labels
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
            
            # Add legend for the first few plots to avoid clutter
            if i < 4:
                ax.legend(fontsize=9, loc='upper right')
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        # Adjust layout with more space for title
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save the plot
        output_path = os.path.join(self.metrics_root, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Metrics visualization saved to: {output_path}")
        plt.close()

    def save_poses(self, prefix, indices, **pose_dict):
        """Save camera poses (extrinsics/intrinsics) as per-instance JSON files.

        Each value in ``pose_dict`` must be a tensor with leading batch dim B;
        either ``(B, ...)`` or ``(B, V, ...)`` layouts are supported. The
        ``i``-th slice goes to ``<prefix>/<idx_i>.json`` as a recursive list.

        Extrinsics must be ``(..., 4, 4)``; intrinsics must be ``(..., 3, 3)``.
        ``None`` values are skipped. Merges into an existing file if present —
        racy under parallel writers to the same ``<idx>.json``.
        """
        pose_dir = os.path.join(self.results_root, prefix)
        os.makedirs(pose_dir, exist_ok=True)

        normalized = {}
        for k, v in pose_dict.items():
            if v is None:
                continue
            assert isinstance(v, torch.Tensor), f"{k} must be a torch.Tensor, got {type(v)}"
            if 'extrinsics' in k:
                assert v.shape[-2:] == (4, 4), f"{k} must be (4, 4), got {tuple(v.shape)}"
            elif 'intrinsics' in k:
                assert v.shape[-2:] == (3, 3), f"{k} must be (3, 3), got {tuple(v.shape)}"
            normalized[k] = v

        for i, idx in enumerate(indices):
            idx_int = int(idx)
            path = os.path.join(pose_dir, f'{idx_int}.json')
            existing = {}
            if os.path.exists(path):
                with open(path) as f:
                    existing = json.load(f)
            existing.update({k: v[i].detach().cpu().tolist() for k, v in normalized.items()})
            with open(path, 'w') as f:
                json.dump(existing, f, indent=2)


class FlowMatchingCFGEvaluator(ClassifierFreeGuidanceMixin, FlowMatchingEvaluator):
    """
    Evaluator for Flow Matching latent model with Sparse UV structure decoder and classifier-free guidance.
    """
    pass


class MultiPoseFlowMatchingEvaluator(FlowMatchingEvaluator):
    """
    Evaluator for multi-pose Flow Matching that jointly denoises structure (x_0) and
    per-view poses (P_0). Evaluates occupancy, UV volume, and pose metrics per-view,
    then averages across views.

    The ``alignment_method`` switch (output-frame → GT-frame similarity) lives
    on ``FlowMatchingEvaluator`` and is shared with the single-view evaluator.
    Multi-pose-specific behavior here is the joint x/P sampling and per-view
    pose metric expansion.
    """

    def _decode_ssuv(self, x_pred: torch.Tensor, P_pred: torch.Tensor):
        """Decode structure from x_pred and per-view SUV from P_pred."""
        ss_latent_dim = self.models['ss_decoder'].latent_channels
        ss_logits = self.models['ss_decoder'](x_pred[:, :ss_latent_dim])
        ss_recon = (ss_logits > 0).int()

        if 'suv_decoder' not in self.models:
            return {
                'ss_logits': ss_logits,
                'ss_recon': ss_recon,
                'ssuv_logits': None,
                'ssuv_recon': None,
                'uv_volume_pred': None,
                # Raw pre-decoder latents — pose-only test-view denoising reads
                # x_pred_latent as the "clean" shape latent to condition on, and
                # P_pred_latent for joint-mode initialization. Cheap to retain
                # since they're already in memory at this point.
                'x_pred_latent': x_pred,
                'P_pred_latent': P_pred,
            }

        N, V = P_pred.shape[:2]
        suv_decoder = self.models['suv_decoder']
        suv_latent_dim = suv_decoder.latent_channels
        P_flat = P_pred.reshape(N * V, *P_pred.shape[2:])
        suv_logits_all = suv_decoder(P_flat[:, :suv_latent_dim])
        suv_logits_all = suv_logits_all.reshape(N, V, *suv_logits_all.shape[1:])

        ssuv_logits = suv_logits_all[:, :, :1]  # [N, V, 1, R, R, R]
        uv_logits = suv_logits_all[:, :, 1:]    # [N, V, >=2, R, R, R]

        if self.predict_ssuv_from_ss:
            ssuv_logits = ss_logits.unsqueeze(1).expand_as(ssuv_logits)

        ssuv_logits_flat = ssuv_logits.reshape(N * V, *ssuv_logits.shape[2:])
        ssuv_recon_flat = expand_binary_volume(
            logits=ssuv_logits_flat,
            min_num_activated=self.ssuv_expand_min_num_activated,
            threshold_step=self.ssuv_expand_threshold_step,
        )
        ssuv_recon = ssuv_recon_flat.reshape(N, V, *ssuv_recon_flat.shape[1:])
        uv_volume_pred = torch.sigmoid(uv_logits[:, :, :2])  # [N, V, 2, R, R, R]

        return {
            'ss_logits': ss_logits,
            'ss_recon': ss_recon,
            'ssuv_logits': ssuv_logits,
            'ssuv_recon': ssuv_recon,
            'uv_volume_pred': uv_volume_pred,
            # See note above — used by ``_sample_test_view_poses``.
            'x_pred_latent': x_pred,
            'P_pred_latent': P_pred,
        }

    def _sample_ssuv_from_t(self, t: torch.Tensor, x_0: torch.Tensor, P_0: torch.Tensor, cond: torch.Tensor, **kwargs):
        V = P_0.shape[1]
        x_noise = torch.randn_like(x_0)
        P_noise = torch.randn_like(P_0)
        t_x = t
        t_P = t[:, None].expand(-1, V)

        x_t = self.diffuse(x_0, t_x, x_noise)
        P_t = self.diffuse(P_0, t_P, P_noise)

        args = self.get_inference_cond(cond, **kwargs)
        x_pred_v, P_pred_v = self.models['denoiser'](x_t, P_t, t_x * 1000, t_P * 1000, **args)

        # Reconstruct x̂_0 and P̂_0
        t_x_expand = t_x[:, None, None, None, None]
        x_pred_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t_x_expand) * x_pred_v
        t_P_expand = t_P[:, :, None, None, None, None]
        P_pred_0 = (1 - self.sigma_min) * P_t - (self.sigma_min + (1 - self.sigma_min) * t_P_expand) * P_pred_v

        return self._decode_ssuv(x_pred_0, P_pred_0)

    def _sample_ssuv(self, x_0: torch.Tensor, P_0: torch.Tensor, cond: torch.Tensor, **kwargs):
        """Returns ``(final_preds, step_payload)``; step entries are
        ``(step_idx, t_value, pred_x_0, pred_P_0)``.
        """
        args = self.get_inference_cond(cond, **kwargs)
        x_noise = torch.randn_like(x_0)
        P_noise = torch.randn_like(P_0)
        result = self.ssuv_sampler.sample(
            model=self.models['denoiser'],
            noise=(x_noise, P_noise),
            **args,
            **self.ssuv_sampler_params,
            verbose=False,
        )
        x_pred, P_pred = result.samples
        final_preds = self._decode_ssuv(x_pred, P_pred)
        step_payload = self._build_step_payload(result, ('pred_x_0', 'pred_P_0'))
        return final_preds, step_payload

    _WARNED_V_LT_3 = False  # class-level; warn once per process

    def _evaluate_poses(
        self,
        uv_volume,     # [N, V, 2, D, H, W]
        ss,            # [N, 1, D, H, W] or [N, V, 1, D, H, W]
        ssuv,          # [N, V, 1, D, H, W]
        uv_volume_pred,
        ssuv_recon,
        intrinsics,    # [N, V, 3, 3]
        extrinsics,    # [N, V, 4, 4]
        indices=None,
        pose_suffix='',
        **kwargs,
    ):
        """MultiPose override: computes per-view pose metrics, fits a
        similarity transform from camera centers, and emits aligned per-view
        metrics plus alignment-quality metrics."""
        import warnings

        # 1. Shape assertion — catches accidental caller-contract drift.
        assert extrinsics.ndim == 4 and extrinsics.shape[-2:] == (4, 4), \
            f"expected [N, V, 4, 4] extrinsics, got {tuple(extrinsics.shape)}"
        assert intrinsics.shape[-2:] == (3, 3)
        N, V = extrinsics.shape[:2]

        # 2. Fallback check.
        skip_alignment = False
        if V < 3:
            if not MultiPoseFlowMatchingEvaluator._WARNED_V_LT_3:
                warnings.warn(
                    "Skipping similarity alignment for V<3: rank-deficient "
                    "camera-centers fit. Reporting raw per-view pose metrics "
                    "only.", RuntimeWarning)
                MultiPoseFlowMatchingEvaluator._WARNED_V_LT_3 = True
            skip_alignment = True

        # 3. Flatten and run DLT for pred (always) and oracle (only when GT 3D
        # data is present).
        uv_pred_flat = uv_volume_pred.reshape(N * V, *uv_volume_pred.shape[2:])
        ssuv_recon_flat = ssuv_recon.reshape(N * V, *ssuv_recon.shape[2:])
        ext_flat = extrinsics.reshape(N * V, 4, 4)
        intrinsics_flat = intrinsics.reshape(N * V, *intrinsics.shape[2:])
        extrinisc_pred, intrinsic_pred, reproj_pred = self._solve_pose_from_ssuv(ssuv_recon_flat, uv_pred_flat, intrinsics_gt=intrinsics_flat)

        has_oracle_gt = uv_volume is not None and ss is not None and ssuv is not None
        if has_oracle_gt:
            uv_flat = uv_volume.reshape(N * V, *uv_volume.shape[2:])
            ssuv_flat = ssuv.reshape(N * V, *ssuv.shape[2:])
            # Oracle must mirror the pred path's mask source; under predict_ssuv_from_ss
            # the pred uses ss_logits, and GT `ssuv` (frustum mask when const_ssuv=True)
            # would include out-of-crop voxels whose uv_volume got clamped — DLT garbage.
            if self.predict_ssuv_from_ss:
                if ss.ndim == 5:  # [N, 1, D, H, W] — shared across views
                    ss_nv = ss.unsqueeze(1).expand(N, V, *ss.shape[1:])
                else:             # [N, V, 1, D, H, W]
                    ss_nv = ss
                oracle_mask_flat = ss_nv.reshape(N * V, *ss_nv.shape[2:])
            else:
                oracle_mask_flat = ssuv_flat
            extrinisc_best, intrinsic_best, reproj_best = self._solve_pose_from_ssuv(oracle_mask_flat, uv_flat, intrinsics_gt=intrinsics_flat)

        # 4. Pred + oracle pose-fidelity blocks. Compute the standard 4-tuple
        # via compute_pose_diff_metrics, attach DLT reproj as a 5th, then
        # spread each across views with emit_per_view_metrics.
        K_gt_flat = intrinsics.reshape(N * V, *intrinsics.shape[2:])
        metrics = {}
        pose_blocks = [('', extrinisc_pred, intrinsic_pred, reproj_pred)]
        if has_oracle_gt:
            pose_blocks.append(('oracle_', extrinisc_best, intrinsic_best, reproj_best))
        for prefix, ext, K, reproj in pose_blocks:
            pose = compute_pose_diff_metrics(
                ext, K, ext_flat, K_gt_flat, ambiguity=self.pose_ambiguity,
            )
            pose['reproj'] = reproj.reshape(-1).clamp_max(1.0)
            for k, v in pose.items():
                emit_per_view_metrics(metrics, f'{prefix}{k}', v, N, V)

        # Pairwise relative-pose metrics use raw predicted extrinsics and
        # require at least two views; no alignment or oracle pose is used.
        if V >= 2:
            ext_pred_pair_nv = extrinisc_pred.reshape(N, V, 4, 4)
            metrics.update(pairwise_pose_diff_metrics(
                ext_pred_pair_nv, extrinsics, ambiguity=self.pose_ambiguity,
            ))

        if skip_alignment:
            if self.save_pose:
                kw = {f'extrinsics_pred{pose_suffix}': extrinisc_pred.reshape(N, V, 4, 4),
                      f'intrinsics_pred{pose_suffix}': intrinsic_pred.reshape(N, V, 3, 3)}
                if pose_suffix == '':
                    kw['extrinsics_gt'] = extrinsics
                    kw['intrinsics_gt'] = intrinsics
                self.save_poses('poses', indices, **kw)
            return metrics, {}

        # 5-6. Fit similarity (camera-centers + look-direction augmentation)
        # and apply it to every pred extrinsic. Pose metrics always use the
        # trajectory similarity (§5.3), regardless of self.alignment_method,
        # since they measure pose-rig fidelity specifically.
        ext_pred_nv = extrinisc_pred.reshape(N, V, 4, 4)

        align_result = self._fit_trajectory_alignment(ext_pred_nv, extrinsics)
        s = align_result["s"]
        R_pa = align_result["R_pa"]
        T_pa = align_result["T_pa"]
        residual = align_result["residual"]
        valid = align_result["valid"]
        ext_aln_nv = align_result["extrinsics_pred_aligned"]                     # [N,V,4,4]
        ext_aln_flat = ext_aln_nv.reshape(N * V, 4, 4)

        # 7. Aligned per-view metrics, masked by validity.
        # Alignment is a similarity transform — it doesn't touch K, so we pass
        # K_pred for both args (rel_fov_deg would be 0) and skip that key.
        pose_aln = compute_pose_diff_metrics(
            ext_aln_flat, intrinsic_pred, ext_flat, intrinsic_pred,
            ambiguity=self.pose_ambiguity,
        )
        valid_flat = valid[:, None].expand(N, V).reshape(-1)
        nan = torch.full_like(pose_aln['rel_rangle_deg'], float("nan"))
        for k in ('rel_rangle_deg', 'rel_tangle_deg', 'cxcy_dist',
                  'rel_rangle_acc_5deg', 'rel_rangle_acc_15deg'):
            v = torch.where(valid_flat, pose_aln[k], nan)
            emit_per_view_metrics(metrics, f'aligned_{k}', v, N, V, nan_safe=True)

        # 8. Alignment-quality metrics (per-instance).
        eye = torch.eye(3, device=R_pa.device, dtype=R_pa.dtype).expand(N, 3, 3)
        # R_pa is row-vector form; column-form rotation from identity is R_pa^T.
        R_pa_col = R_pa.transpose(-1, -2)
        align_rangle = rotation_angle(R_pa_col, eye, ambiguity=False)
        align_ttrans = torch.norm(T_pa, dim=-1)

        nan1 = torch.full((N,), float("nan"), device=s.device, dtype=s.dtype)
        metrics['align_scale'] = torch.where(valid, s, nan1)
        metrics['align_rangle_deg'] = torch.where(valid, align_rangle, nan1)
        metrics['align_ttrans'] = torch.where(valid, align_ttrans, nan1)
        metrics['align_residual'] = torch.where(valid, residual, nan1)

        if self.save_pose:
            intr_pred_nv = intrinsic_pred.reshape(N, V, 3, 3)
            kw = {f'extrinsics_pred{pose_suffix}': ext_pred_nv,
                  f'intrinsics_pred{pose_suffix}': intr_pred_nv,
                  f'extrinsics_pred_aligned{pose_suffix}': ext_aln_nv,
                  f'intrinsics_pred_aligned{pose_suffix}': intr_pred_nv}
            if pose_suffix == '':
                kw['extrinsics_gt'] = extrinsics
                kw['intrinsics_gt'] = intrinsics
            self.save_poses('poses', indices, **kw)

        return metrics, {}

    def postprocess_metrics_statistics(self, metrics_stats):
        """Remove per-view metrics that are grouped by index in the batch,
        and also drop per-step columns (handled by base class).
        """
        metrics_stats = super().postprocess_metrics_statistics(metrics_stats)
        return {k: v for k, v in metrics_stats.items() if not _PER_VIEW_COL_RE.search(k)}

    def get_extra_metrics_statistics(self, df):
        """Add per-view statistics grouped by the dataset's actual view indices."""
        result = super().get_extra_metrics_statistics(df)

        dataset = self.dataset
        if not (hasattr(dataset, '_select_eval_views') and hasattr(dataset, 'num_pose_views')
                and dataset.num_pose_views > 1):
            return result

        metric_bases = sorted({m.group(1) for c in df.columns
                               for m in [_PER_VIEW_COL_BASE_RE.match(c)] if m})
        if not metric_bases:
            return result

        per_real_view = {}  # real_view_idx -> {metric_base -> Series}
        for dataset_idx, group in df.groupby('dataset_index'):
            _, viewset_idx = divmod(int(dataset_idx), dataset.eval_num_viewsets)
            view_indices = dataset._select_eval_views(viewset_idx)
            for vi, real_idx in enumerate(view_indices):
                bucket = per_real_view.setdefault(real_idx, {})
                for base in metric_bases:
                    col = f'{base}_v{vi}'
                    if col not in df.columns:
                        continue
                    values = group[col].dropna()
                    if base not in bucket:
                        bucket[base] = values
                    else:
                        bucket[base] = pd.concat([bucket[base], values])

        stats_by_view = {}
        for real_idx in sorted(per_real_view):
            view_stats = {}
            for base, values in per_real_view[real_idx].items():
                if len(values) > 0:
                    view_stats[base] = self._compute_column_stats(values)
            stats_by_view[f'view_{real_idx}'] = view_stats
        result['statistics_by_view'] = stats_by_view

        return result

    def _get_test_chunk_size(self, V_input_for_joint: int = 0) -> int:
        """Resolve the per-call test-view chunk size for pose-only sampling.

        The hard cap is the model's ``num_view_embeds`` (K), minus any
        slots reserved for pinned-clean input views (``V_input_for_joint``).
        Caller may further reduce via ``self.test_view_chunk_size``; ``None``
        defaults to the cap.
        """
        denoiser = self.models['denoiser']
        # Walk through torch.compile wrapper if present.
        K = getattr(denoiser, 'num_view_embeds', None)
        if K is None and hasattr(denoiser, '_orig_mod'):
            K = getattr(denoiser._orig_mod, 'num_view_embeds', None)
        assert K is not None and K > 0, \
            "denoiser must expose num_view_embeds > 0 for pose-only sampling"
        cap = K - V_input_for_joint
        assert cap > 0, (
            f"num_view_embeds={K} ≤ V_input={V_input_for_joint}; "
            f"no view-embed slots left for test views in joint mode")
        chunk = self.test_view_chunk_size if self.test_view_chunk_size is not None else cap
        assert chunk > 0, f"test_view_chunk_size must be positive, got {chunk}"
        if chunk > cap:
            import warnings
            warnings.warn(
                f"test_view_chunk_size={chunk} exceeds cap {cap} (num_view_embeds={K}, "
                f"V_input_for_joint={V_input_for_joint}); clamping to {cap}.",
                RuntimeWarning,
            )
            chunk = cap
        return chunk

    def _sample_test_view_poses(self, x_pred_latent, P_input_pred_latent,
                                input_cond, test_cond, test_intrinsics):
        """Pose-only denoising for test views, returning extrinsics in pred frame.

        Args:
            x_pred_latent: ``[N, C, R, R, R]`` clean shape latent (already
                sampled or GT). Pinned at ``t_x = 0`` for the whole loop.
            P_input_pred_latent: ``[N, V_in, C_pose, R_p, R_p, R_p]`` clean
                input-view pose latents. Required when
                ``num_input_views_in_joint > 0`` (the first K of these are
                concatenated as pinned-clean slots); ignored otherwise.
            input_cond: ``[N, V_in, 3, H, W]`` input-view conditioning images.
                Required when ``num_input_views_in_joint > 0``; ignored
                otherwise.
            test_cond: ``[N, M, 3, H, W]`` per-test-view conditioning images.
            test_intrinsics: ``[N, M, 3, 3]`` GT intrinsics for the test
                views (used by the DLT/PnP pose solver — we don't trust the
                model's intrinsic prediction for novel views).

        Returns:
            extrinsics_test_pred: ``[N, M, 4, 4]`` in pred frame.
            intrinsics_test_pred: ``[N, M, 3, 3]`` (GT K when use_gt_intrinsics).
        """
        assert self.denoise_test_view_poses, \
            "_sample_test_view_poses requires denoise_test_view_poses=True"
        assert self.pose_only_sampler is not None, "pose_only_sampler not constructed"
        assert test_cond.ndim == 5, f"test_cond must be [N,M,C,H,W], got {tuple(test_cond.shape)}"
        N, M = test_cond.shape[:2]

        # Resolve K (input-view slots pinned clean per chunk). Clamp to the
        # number of input views actually available; K=0 ≡ test-only.
        V_in_total = P_input_pred_latent.shape[1] if P_input_pred_latent is not None else 0
        K = min(self.num_input_views_in_joint, V_in_total)
        if K > 0:
            assert P_input_pred_latent is not None, \
                "num_input_views_in_joint>0 requires P_input_pred_latent"
            assert input_cond is not None and input_cond.ndim == 5, \
                "num_input_views_in_joint>0 requires input_cond [N, V_in, 3, H, W]"
            assert input_cond.shape[1] >= K, \
                f"input_cond has {input_cond.shape[1]} views < K={K}"
        chunk = self._get_test_chunk_size(V_input_for_joint=K)

        device = x_pred_latent.device
        dtype = x_pred_latent.dtype
        # Per-view pose latent shape (C_pose, R_p, R_p, R_p). With K>0 we
        # inherit it from the input pose latent; otherwise we read from the
        # model spec.
        if K > 0:
            P_shape = P_input_pred_latent.shape[2:]
        else:
            denoiser = self.models['denoiser']
            d = denoiser._orig_mod if hasattr(denoiser, '_orig_mod') else denoiser
            P_shape = (d.pose_in_channels, d.pose_resolution,
                       d.pose_resolution, d.pose_resolution)

        ext_chunks: List[torch.Tensor] = []
        K_chunks: List[torch.Tensor] = []

        for s in range(0, M, chunk):
            e = min(s + chunk, M)
            Vc = e - s
            cond_c = test_cond[:, s:e]                                  # [N, Vc, 3, H, W]
            P_test_noise = torch.randn(N, Vc, *P_shape, device=device, dtype=dtype)

            if K > 0:
                # Concatenate the first K input slots with the test chunk
                # along V. Sampler pins K input slots clean via
                # ``pose_denoise_mask`` (False = clean, True = denoise).
                P_in_used    = P_input_pred_latent[:, :K]                # [N, K, ...]
                cond_in_used = input_cond[:, :K]                         # [N, K, 3, H, W]
                P_init    = torch.cat([P_in_used,    P_test_noise], dim=1)
                cond_full = torch.cat([cond_in_used, cond_c],       dim=1)
                pose_mask = torch.zeros(K + Vc, dtype=torch.bool, device=device)
                pose_mask[K:] = True
            else:
                P_init = P_test_noise
                cond_full = cond_c
                pose_mask = None  # all-True via sampler default

            # Mirror _sample_ssuv: pose-only sampling drives the joint x+P
            # denoiser (the ssuv branch), so request type='ssuv'.
            args = self.get_inference_cond(cond_full, type='ssuv')
            ssuv_denoiser = self.models['ssuv_denoiser'] if 'ssuv_denoiser' in self.models else self.models['denoiser']
            result = self.pose_only_sampler.sample(
                model=ssuv_denoiser,
                noise=(x_pred_latent, P_init),
                **args,
                **self.pose_only_sampler_params,
                pose_denoise_mask=pose_mask,
                verbose=False,
            )
            _, P_full = result.samples
            P_test_chunk = P_full[:, K:]                                 # drop input slots if any

            # Decode the pose latent to (ssuv, uv_volume) and DLT/PnP-solve.
            # Reuse _decode_ssuv: it expects [N, V_chunk, ...]. ssuv comes from
            # ss_logits (always under predict_ssuv_from_ss), or from the suv
            # decoder otherwise.
            preds_chunk = self._decode_ssuv(x_pred_latent, P_test_chunk)
            assert preds_chunk['uv_volume_pred'] is not None, \
                "_sample_test_view_poses requires a suv_decoder"
            ssuv_recon_flat = preds_chunk['ssuv_recon'].reshape(N * Vc, *preds_chunk['ssuv_recon'].shape[2:])
            uv_pred_flat = preds_chunk['uv_volume_pred'].reshape(N * Vc, *preds_chunk['uv_volume_pred'].shape[2:])
            K_gt_flat = test_intrinsics[:, s:e].reshape(N * Vc, *test_intrinsics.shape[2:])
            ext_flat, K_flat, _ = self._solve_pose_from_ssuv(
                ssuv_recon_flat, uv_pred_flat, intrinsics_gt=K_gt_flat)

            ext_chunks.append(ext_flat.reshape(N, Vc, 4, 4))
            K_chunks.append(K_flat.reshape(N, Vc, 3, 3))

        return torch.cat(ext_chunks, dim=1), torch.cat(K_chunks, dim=1)

    def _compute_metrics_from_preds(self, preds, data, indices, suffix=''):
        """MultiPose variant — per-view flatten/average for occupancy+UV,
        unflattened pose eval. Non-empty ``suffix`` suffixes metric keys,
        merges per-step pose files, and suppresses samples.
        """
        metrics, samples = {}, {}
        has_suv_decoder = 'suv_decoder' in self.models
        emit_samples = (suffix == '')
        N = data['x_0'].shape[0]
        V = data['P_0'].shape[1]

        ss_gt = data['ss'][:, 0] if data['ss'].ndim == 6 else data['ss']
        ss_metrics, ss_samples = self._evaluate_occupancy('ss', ss_gt, preds['ss_logits'], preds['ss_recon'])
        metrics.update({k + suffix: v for k, v in ss_metrics.items()})
        if emit_samples:
            samples.update(ss_samples)

        if has_suv_decoder and not self.predict_ssuv_from_ss:
            ssuv_gt_flat = data['ssuv'].reshape(N * V, *data['ssuv'].shape[2:])
            ssuv_logits_flat = preds['ssuv_logits'].reshape(N * V, *preds['ssuv_logits'].shape[2:])
            ssuv_recon_flat = preds['ssuv_recon'].reshape(N * V, *preds['ssuv_recon'].shape[2:])
            ssuv_metrics, _ = self._evaluate_occupancy('ssuv', ssuv_gt_flat, ssuv_logits_flat, ssuv_recon_flat)
            for k, v in ssuv_metrics.items():
                metrics[k + suffix] = v.reshape(N, V).mean(dim=1)

        if has_suv_decoder:
            uv_gt_flat = data['uv_volume'].reshape(N * V, *data['uv_volume'].shape[2:])
            ssuv_gt_flat = data['ssuv'].reshape(N * V, *data['ssuv'].shape[2:])
            uv_pred_flat = preds['uv_volume_pred'].reshape(N * V, *preds['uv_volume_pred'].shape[2:])
            uv_metrics, _ = self._evaluate_uv_volume(uv_volume=uv_gt_flat, ssuv=ssuv_gt_flat, uv_volume_pred=uv_pred_flat)
            for k, v in uv_metrics.items():
                metrics[k + suffix] = v.reshape(N, V).mean(dim=1)

        if self.evaluate_pose:
            pose_metrics, _ = self._evaluate_poses(
                indices=indices,
                uv_volume=data.get('uv_volume'),
                ss=data.get('ss'),
                ssuv=data.get('ssuv'),
                uv_volume_pred=preds['uv_volume_pred'],
                ssuv_recon=preds['ssuv_recon'],
                intrinsics=data['intrinsics'],
                extrinsics=data['extrinsics'],
                pose_suffix=suffix,
            )
            metrics.update({k + suffix: v for k, v in pose_metrics.items()})

            # Test-view pose metrics via pose-only denoising. The resulting
            # extrinsics live in pred frame, so compare directly to GT test
            # extrinsics — no alignment fit involved.
            if self.denoise_test_view_poses and 'test_cond' in data:
                test_metrics = self._evaluate_test_view_poses(preds, data)
                metrics.update({k + suffix: v for k, v in test_metrics.items()})

        return metrics, samples

    def _evaluate_test_view_poses(
        self,
        preds,
        data,
        ext_test: Optional[torch.Tensor] = None,
        K_test: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Emit per-view ``nv_*`` pose metrics for test views.

        If ``ext_test`` / ``K_test`` are provided, reuse them; otherwise run
        pose-only denoising via ``_sample_test_view_poses``. Sparse-multipose
        callers already sample for NV rendering — pass those in to avoid
        denoising twice.
        """
        if ext_test is None or K_test is None:
            ext_test, K_test = self._sample_test_view_poses(
                x_pred_latent=preds['x_pred_latent'],
                P_input_pred_latent=(preds['P_pred_latent']
                                     if self.num_input_views_in_joint > 0 else None),
                input_cond=(data['cond'] if self.num_input_views_in_joint > 0 else None),
                test_cond=data['test_cond'],
                test_intrinsics=data['test_intrinsics'],
            )
        ext_gt = data['test_extrinsics']
        K_gt = data['test_intrinsics']
        N, M = ext_test.shape[:2]
        ext_test_f = ext_test.reshape(N * M, 4, 4)
        K_test_f = K_test.reshape(N * M, 3, 3)
        ext_gt_f = ext_gt.reshape(N * M, 4, 4)
        K_gt_f = K_gt.reshape(N * M, 3, 3)

        out: Dict[str, torch.Tensor] = {}
        for k, v in compute_pose_diff_metrics(
            ext_test_f, K_test_f, ext_gt_f, K_gt_f, ambiguity=self.pose_ambiguity,
        ).items():
            emit_per_view_metrics(out, f'nv_{k}', v, N, M)
        return out

    def evaluation_metrics(self, indices, **data):
        metrics, samples = {}, {}
        N = data['x_0'].shape[0]
        step_payload = None
        if self.save_debug:
            self.debug_saver.begin_sample()
        try:
            if self.sample_ssuv == 'gt_latent':
                preds = self._decode_ssuv(data['x_0'], data['P_0'])
            elif self.sample_ssuv is True:
                preds, step_payload = self._sample_ssuv(**data)
            else:
                t = self.sample_t(N).to(self.device).float()
                preds = self._sample_ssuv_from_t(t, **data)
                metrics['t'] = t

            m, s = self._compute_metrics_from_preds(preds, data, indices, suffix='')
            metrics.update(m)
            samples.update(s)

            debug_steps = [] if self.save_debug and step_payload is not None else None
            if step_payload is not None:
                for step_i, t_i, x0_i, P0_i in step_payload:
                    preds_i = self._decode_ssuv(x0_i, P0_i)
                    m_i, _ = self._compute_metrics_from_preds(preds_i, data, indices, suffix=f'_step{step_i}')
                    metrics.update(m_i)
                    metrics[f't_step{step_i}'] = torch.full((N,), float(t_i), device=self.device)
                    if debug_steps is not None:
                        debug_steps.append({'step_i': step_i, 't_i': float(t_i),
                                            'preds_i': preds_i, 'metrics_i': m_i})

            if debug_steps:
                self.debug_saver.write(indices, data, debug_steps)

            return metrics, samples
        finally:
            if self.save_debug:
                self.debug_saver.end_sample()


class MultiPoseFlowMatchingCFGEvaluator(ClassifierFreeGuidanceMixin, MultiPoseFlowMatchingEvaluator):
    """
    Multi-pose evaluator with classifier-free guidance.
    """
    pass
