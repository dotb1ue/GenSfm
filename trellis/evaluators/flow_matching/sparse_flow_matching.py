from typing import *
import os
import torch
import torch.nn.functional as F

from ..basic_single_view import BasicSingleViewEvaluator
from ...trainers.flow_matching.mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from ...trainers.flow_matching.mixins.image_conditioned import VisualImageConditionedMixin
from ...pipelines import samplers
from ...modules.sparse.basic import SparseTensor, init_SparseTensor
from ...utils.pose_utils import *
from ...utils.voxel_utils import project_cv
from ...utils.align_utils import transform_points_by_similarity
from ...utils.sample_utils import save_outputs_for_instance
from .flow_matching import FlowMatchingEvaluator, MultiPoseFlowMatchingEvaluator


class SparseFlowMatchingEvaluator(FlowMatchingEvaluator, BasicSingleViewEvaluator):
    """
    Evaluator for Sparse Flow Matching latent model with SLat decoder.

    Two-stage evaluation pipeline:
      1. SSUV stage (inherited from FlowMatchingEvaluator): sample or decode the
         sparse UV structure, optionally solve camera pose from it.
      2. SLat stage: sample the structured latent, decode to 3D representations
         (mesh, gaussian), render, and compute photometric/geometric metrics.

    Args:
        metrics_on_raw_cond (bool): Whether to compute metrics on the raw uncropped condition. Default is True.
        normalization (dict): Optional slat latent normalization stats.
        enable_uv (bool): Whether to use the UV volume for solving pose and sampling slat. Default is True.
        slat_sampler_args (dict): Arguments for the structured latent's sampler.
        save_mesh (bool): Whether to save the mesh. Default is False.

    Inherited from FlowMatchingEvaluator (passed via **kwargs):
        sample_ssuv (bool | 'gt_latent'): SSUV sampling mode. False → skip SSUV stage, use GT pose;
            True → full sampling from pure noise; 'gt_latent' → decode GT latent directly. Default is False.
        ssuv_sampler_args (dict): Sampler config for SSUV sampling (only used when sample_ssuv is True).
        predict_ss_from_ssuv (bool): Predict sparse 3D structure from sparse UV structure. Default is False.
        predict_ssuv_from_ss (bool): Predict sparse UV structure from sparse 3D structure. Default is False.
        evaluate_pose (bool): Whether to evaluate pose metrics from SSUV predictions.
            Only effective when sample_ssuv is truthy. Default is True.
        save_pose (bool): Whether to save predicted/GT poses as JSON files. Default is False.
        use_gt_intrinsics (bool): Use GT intrinsics (PnP) instead of full DLT for pose solving. Default is False.
        dilation_sigma (float): Gaussian kernel sigma for post-processing dilation in pose solver. Default is 0.
        dilation_min_weight (float): Minimum weight for dilated grid in pose solver. Default is 0.05.
        ssuv_expand_min_num_activated (int): Minimum activated voxels for ssuv_recon expansion. Default is 10.
        ssuv_expand_threshold_step (float): Step size for ssuv_recon expansion threshold. Default is 0.05.
    """

    def __init__(
        self,
        *args,
        metrics_on_raw_cond: bool = True,
        normalization: Optional[dict] = None,
        enable_uv: bool = True,
        slat_sampler_args: dict = {"name": "FlowEulerGuidanceIntervalSampler", "args": {"sigma_min": 1e-5}, "params": {"steps": 25, "cfg_strength": 3.0, "cfg_interval": [0.5, 1.0], "rescale_t": 3.0}},
        save_mesh: bool = False,
        # Alignment-related (consumed by _fit_pointmap_alignment / _fit_pointcloud_alignment).
        # Used here regardless of single-view vs multi-pose.
        num_3d_samples: int = 20_000,
        gt_mesh_path_template: str = "{root}/renders/{sha256}/mesh.ply",
        icp_n_inits: int = 4,
        icp_n_iters: int = 50,
        mesh_sample_method: str = "area",
        n_3d_render_views: int = 64,
        n_3d_render_resolution: int = 512,
        evaluate_3d_geometry: bool = False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.evaluate_3d_geometry = evaluate_3d_geometry
        self.metrics_on_raw_cond = metrics_on_raw_cond
        self.normalization = normalization
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(1, -1).cuda()
            self.std = torch.tensor(self.normalization['std']).reshape(1, -1).cuda()
        self.enable_uv = enable_uv
        self.slat_sampler = getattr(samplers, slat_sampler_args['name'])(**slat_sampler_args['args'])
        self.slat_sampler_params = slat_sampler_args['params']
        self.save_mesh = save_mesh
        self.num_3d_samples = int(num_3d_samples)
        self.gt_mesh_path_template = gt_mesh_path_template
        self.icp_n_inits = int(icp_n_inits)
        self.icp_n_iters = int(icp_n_iters)
        valid_methods = {"area", "fps", "render"}
        assert mesh_sample_method in valid_methods, \
            f"mesh_sample_method must be one of {sorted(valid_methods)}, got {mesh_sample_method!r}"
        self.mesh_sample_method = mesh_sample_method
        self.n_3d_render_views = int(n_3d_render_views)
        self.n_3d_render_resolution = int(n_3d_render_resolution)
        self._stage2_pcd: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        init_SparseTensor()  # Init sparse tensor explicitly
        
    def evaluation_metrics(self, indices, **data):
        """
        Compute evaluation metrics using flow reconstruction pipeline.

        Args:
            indices: [N] tensor of indices for this batch.
            **data: May contain keys identical to the VAE evaluator:
                - 'x_0', 'cond', 'ss', 'ssuv', 'uv_volume', 'depth_volume', 'intrinsics', 'extrinsics'.
        Returns:
            Tuple[metrics_dict, samples_dict]
        """
        if self.sample_ssuv:  # use sampled pose
            if self.sample_ssuv == 'gt_latent':
                ssuv_preds = self._decode_ssuv(data['x_0'])
            else:
                ssuv_preds = self._sample_ssuv(**data)
            if self.enable_uv:
                extrinsics_pred, intrinsics_pred, _ = self._solve_pose_from_ssuv(ssuv_preds['ssuv_recon'], ssuv_preds['uv_volume_pred'])
                raw_intrinsics_pred = self._adjust_raw_intrinsics(intrinsics_pred, data.get('crop_bbox'))
            else:
                extrinsics_pred, intrinsics_pred, raw_intrinsics_pred = data['extrinsics'], data['intrinsics'], data['raw_intrinsics']
            noise = self._make_randn_sparse_latent_from_ss(ssuv_preds['ss_recon'])
            if self.enable_uv:
                data['uvs'] = self._make_sparse_uvs_from_pose(noise, extrinsics_pred, intrinsics_pred)
            preds = self._sample_slat(**data, noise=noise)
        else:  # use GT pose
            extrinsics_pred, intrinsics_pred, raw_intrinsics_pred = data['extrinsics'], data['intrinsics'], data['raw_intrinsics']
            # The slat denoiser (e.g. VisualLatent / ConvNeXtVisualLatent variants)
            # requires per-voxel ``uvs``. Pre-build the noise from GT occupancy
            # and project voxel centers via GT pose so we reuse the same coords
            # for both noise and uvs (mirrors the sample_ssuv=True branch).
            if self.enable_uv:
                if isinstance(data.get('x_0'), SparseTensor):
                    noise = data['x_0'].replace(torch.randn_like(data['x_0'].feats))
                else:
                    ss_for_noise = data['ss'][:, 0] if data['ss'].ndim == 6 else data['ss']
                    noise = self._make_randn_sparse_latent_from_ss(ss_for_noise)
                data['uvs'] = self._make_sparse_uvs_from_pose(noise, extrinsics_pred, intrinsics_pred)
                preds = self._sample_slat(**data, noise=noise)
            else:
                preds = self._sample_slat(**data)

        # --- Stage 2: pre-sample 3D point clouds when either C2 alignment or
        # 3D-geom metric is on. Both paths consume the same point sets, so we
        # sample once per batch.
        needs_3d_pcd = self.evaluate_3d_geometry or self.alignment_method == "full_pointcloud"
        if needs_3d_pcd:
            self._stage2_pcd = self._sample_3d_pointclouds(preds['mesh'], indices)
        else:
            self._stage2_pcd = None

        # --- Alignment: fit similarity (output → GT frame) once. Under
        # ``alignment_method='none'`` (default) this returns None and all
        # subsequent renders use raw pred extrinsics — bit-identical to prior
        # behavior. Under aligned modes, GT cams are inverse-mapped into
        # pred-world for rendering pred outputs.
        align = self._fit_alignment(
            extrinsics_pred=extrinsics_pred,
            intrinsics_pred=intrinsics_pred,
            data=data,
            preds=preds,
        )

        cond_prefix = 'raw_' if self.metrics_on_raw_cond else ''

        # Effective input-view (E, K): see multi-pose evaluator for rationale.
        if align is None:
            ext_input_eff = extrinsics_pred
            cond_intrinsics = raw_intrinsics_pred if self.metrics_on_raw_cond else intrinsics_pred
        else:
            ext_input_eff = self._apply_alignment_inverse(data['extrinsics'], align)
            cond_intrinsics = data[f'{cond_prefix}intrinsics']

        renders = self._render_reps(preds['gaussian'], ext_input_eff, cond_intrinsics)
        cond_renders_vis = self._render_color_for_vis(preds['gaussian'], ext_input_eff, cond_intrinsics)
        cond_gt_vis = self._compose_gt_for_vis(data.get(f'{cond_prefix}cond_rgba'))
        metrics, samples = self.evaluate_photometric(
            'cond', data[f'{cond_prefix}cond'], renders['color'],
            images_vis=cond_gt_vis, renders_vis=cond_renders_vis,
        )
        self.save_images('cond_color', indices,
            cond_gt_vis if cond_gt_vis is not None else data[f'{cond_prefix}cond'],
            cond_renders_vis if cond_renders_vis is not None else renders['color'])

        renders = self._render_reps(preds['mesh'], ext_input_eff, cond_intrinsics)
        renders['depth'] *= (renders['mask'] == 1.0)  # mask out alpha-transition regions
        depth_metrics, depth_samples, cond_pointmaps = self.evaluate_geometry(
            'cond',
            data[f'{cond_prefix}cond_depth'], renders['depth'], data['extrinsics'], data[f'{cond_prefix}intrinsics'])
        metrics.update(depth_metrics)
        samples.update(depth_samples)
        self.save_images('cond_depth', indices,
            depth_samples['cond_depth_gt']['value'], depth_samples['cond_depth_render']['value'])
        self.save_pointmaps('cond_pointmap', indices, cond_pointmaps['gt'], cond_pointmaps['rd'])

        if self.evaluate_novel_view:
            assert 'test_extrinsics' in data, (
                "evaluate_novel_view=True but the dataset batch contains no test_* keys; "
                "set eval_test_views=[...] on the eval_dataset (MultiViewMixin) so "
                "_gather_test_views populates test_extrinsics / test_intrinsics / test_cond."
            )
            nv_images = data[f'test_{cond_prefix}cond']
            nv_intrinsics = data[f'test_{cond_prefix}intrinsics']
            # Under alignment, pull GT test cams into pred-world via inverse
            # similarity; under 'none' use the GT test extrinsics verbatim
            # (the historical assumption that pred frame == GT frame).
            nv_extrinsics = self._apply_alignment_inverse(data['test_extrinsics'], align)
            nv_renders = self._render_reps(preds['gaussian'], nv_extrinsics, nv_intrinsics)
            nv_renders_vis = self._render_color_for_vis(preds['gaussian'], nv_extrinsics, nv_intrinsics)
            nv_gt_vis = self._compose_gt_for_vis(data.get(f'test_{cond_prefix}cond_rgba'))
            # Metric path is 4D-only; flatten any [N, M, ...] to [N*M, ...].
            def _flat4(t):
                return t.reshape(-1, *t.shape[-3:]) if (t is not None and t.ndim == 5) else t
            nv_metrics, nv_samples = self.evaluate_photometric(
                'nv', _flat4(nv_images), _flat4(nv_renders['color']),
                images_vis=_flat4(nv_gt_vis), renders_vis=_flat4(nv_renders_vis),
            )
            metrics.update(nv_metrics)
            samples.update(nv_samples)
            # Save with natural shape so save_images uses its multi-view branch.
            self.save_images('nv_color', indices,
                nv_gt_vis if nv_gt_vis is not None else nv_images,
                nv_renders_vis if nv_renders_vis is not None else nv_renders['color'])

            if self.evaluate_clip_metric:
                nv_clip_metrics, nv_clip_samples = self.evaluate_novel_view_clip(data[f'{cond_prefix}cond'], nv_images, nv_renders['color'])
                metrics.update(nv_clip_metrics)
                samples.update(nv_clip_samples)

        if self.evaluate_cross_view:
            cv_colors, cv_colors_vis = [], []
            for theta in self.cross_view_angles:
                cv_extrinsics = rotate_extrinsics_world_to_cam_batch(ext_input_eff, self.cross_view_axis, theta, self.cross_view_pivot)
                cv_renders = self._render_reps(preds['gaussian'], cv_extrinsics, cond_intrinsics)
                cv_colors.append(cv_renders['color'])
                cv_vis_i = self._render_color_for_vis(preds['gaussian'], cv_extrinsics, cond_intrinsics)
                if cv_vis_i is not None:
                    cv_colors_vis.append(cv_vis_i)
            cv_colors = torch.stack(cv_colors, dim=1)
            cv_colors_vis = torch.stack(cv_colors_vis, dim=1) if cv_colors_vis else cv_colors

            if self.evaluate_clip_metric:
                cv_metrics, cv_samples = self.evaluate_cross_view_clip(data[f'{cond_prefix}cond'], cv_colors)
                metrics.update(cv_metrics)
                samples.update(cv_samples)

            cv_gt_vis = self._compose_gt_for_vis(data.get(f'{cond_prefix}cond_rgba'))
            cv_gt = cv_gt_vis if cv_gt_vis is not None else data[f'{cond_prefix}cond']
            self.save_images('cv', indices, cv_gt, cv_colors_vis)

        if self.save_mesh:
            # Save raw pred extrinsic alongside the eff K (matches multi-pose).
            self._save_mesh(indices, preds, extrinsics_pred, cond_intrinsics)

        if 'occlusion_mask' in data:
            mask = 1.0 - data[f'{cond_prefix}occlusion_mask'][:, None, :, :].float()
            soft_mask = mask * 0.5 + 0.5
            self.save_images('cond_masked', indices, data[f'{cond_prefix}cond'] * mask, data[f'{cond_prefix}cond'] * soft_mask)

        # --- Stage 2: 3D-CD / 3D-F@τ on full point clouds (protocol §4.2). Pred
        # points live in pred frame; transform to GT frame via the forward
        # similarity (no-op when alignment_method='none').
        if self.evaluate_3d_geometry:
            N = data['cond'].shape[0]
            pred_pts, gt_pts = self._stage2_pcd
            if align is not None:
                pred_pts_gt = transform_points_by_similarity(
                    pred_pts, align['s'], align['R_pa'], align['T_pa'])
            else:
                pred_pts_gt = pred_pts
            pred_list = [pred_pts_gt[n] for n in range(N)]
            gt_list = [gt_pts[n] for n in range(N)]
            geom3d_metrics, _ = self._evaluate_3d_geometry(
                'cond', indices, pred_list, gt_list, (0.01, 0.05, 0.10))
            metrics.update(geom3d_metrics)

        # Free the per-batch point cache before pose eval / next batch.
        self._stage2_pcd = None

        if self.evaluate_pose and self.sample_ssuv:
            pose_metrics, _ = self._evaluate_poses(
                indices=indices,
                uv_volume=data.get('uv_volume'), ss=data.get('ss'), ssuv=data.get('ssuv'),
                uv_volume_pred=ssuv_preds['uv_volume_pred'], ssuv_recon=ssuv_preds['ssuv_recon'],
                intrinsics=data['intrinsics'], extrinsics=data['extrinsics'],
            )
            metrics.update(pose_metrics)

        return metrics, samples

    def _decode_slat(self, pred_x_0: SparseTensor):
        """
        Decode the latent to representations

        Args:
            pred_x_0: The predicted latent tensor.
            
        Returns:
            A dict with the outputs of the decoder.
        """
        if self.normalization is not None:
            pred_x_0 = pred_x_0.replace(pred_x_0.feats * self.std + self.mean)

        format_to_postfix = {'mesh': 'mesh', 'gaussian': 'gs', 'radiance_field': 'rf'}
        ret = {}
        for format in ['mesh', 'gaussian', 'radiance_field']:
            ret[format] = []
            format_key = f'slat_decoder_{format_to_postfix[format]}'
            if format_key not in self.models:
                continue
            slat_dec = self.models[format_key]
            for i in range(0, pred_x_0.shape[0]):
                ret[format].append(slat_dec(pred_x_0[i:i+1]))
            ret[format] = sum(ret[format], [])
        return ret

    def _render_color_for_vis(self, reps, extrinsics, intrinsics):
        """Re-render ``reps`` with ``self.render_bg_color`` for visualization.
        Returns the ``'color'`` tensor only, or ``None`` if vis bg is black."""
        if not any(c != 0.0 for c in self.render_bg_color):
            return None
        return self._render_reps(reps, extrinsics, intrinsics, bg_color=self.render_bg_color)['color']

    def _compose_gt_for_vis(self, rgba):
        """Composite un-premultiplied RGBA [..., 4, H, W] onto the vis bg.
        Returns ``None`` when vis bg is black or rgba is missing — caller then
        falls back to the original premultiplied-on-black GT tensor."""
        if rgba is None or not any(c != 0.0 for c in self.render_bg_color):
            return None
        bg = rgba.new_tensor(self.render_bg_color).view(*([1] * (rgba.ndim - 3)), 3, 1, 1)
        rgb, alpha = rgba[..., :3, :, :], rgba[..., 3:4, :, :]
        return rgb * alpha + bg * (1.0 - alpha)

    def _sample_slat(self, cond: torch.Tensor, noise: Optional[SparseTensor] = None, **kwargs):
        """
        Obtain slat predictions by full sampling.
        """
        args = self.get_inference_cond(cond, type='slat', **kwargs)
        slat_denoiser = self.models['slat_denoiser'] if 'slat_denoiser' in self.models else self.models['denoiser']

        if noise is None:
            if 'x_0' in kwargs and isinstance(kwargs['x_0'], SparseTensor):
                sampling_noise = kwargs['x_0'].replace(torch.randn_like(kwargs['x_0'].feats))
            else:
                coords = kwargs['ss'].squeeze(1).nonzero().int().contiguous()
                sampling_noise = SparseTensor(
                    coords=coords, 
                    feats=torch.randn((coords.shape[0], slat_denoiser.in_channels), dtype=torch.float32, device=coords.device),
                )
        else:
            sampling_noise = noise
            
        result = self.slat_sampler.sample(
            model=slat_denoiser, 
            noise=sampling_noise, 
            **args,
            **self.slat_sampler_params, 
            verbose=False,
        )
        pred_x_0 = result.samples
        return self._decode_slat(pred_x_0)

    def _make_randn_sparse_latent_from_ss(self, ss: torch.Tensor) -> SparseTensor:
        coords = []
        layout = []
        start = 0
        for i, ss_i in enumerate(ss):
            coords_i = ss_i.nonzero()
            coords_i[:, 0] += i
            coords.append(coords_i.int())
            layout.append(slice(start, start + coords_i.shape[0]))
            start += coords_i.shape[0]
        coords = torch.cat(coords)

        slat_denoiser = self.models['slat_denoiser'] if 'slat_denoiser' in self.models else self.models['denoiser']
        latent_dim = slat_denoiser.out_channels

        randn_x_0 = SparseTensor(
            coords=coords,
            feats=torch.randn((coords.shape[0], latent_dim), dtype=torch.float32, device=slat_denoiser.device),
        )
        randn_x_0.register_spatial_cache('layout', layout)

        return randn_x_0

    def _make_sparse_uvs_from_pose(self, x: SparseTensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> SparseTensor:
        """Project voxel centers onto camera poses. Accepts single-view [B, 4, 4] or
        multi-view [B, V, 4, 4] poses; returns feats [N, 2] or [N, V, 2] accordingly."""
        slat_denoiser = self.models['slat_denoiser'] if 'slat_denoiser' in self.models else self.models['denoiser']
        resolution = slat_denoiser.resolution
        is_multiview = extrinsics.ndim == 4

        uvs = []
        for i, x_i in enumerate(x.unbind(0)):
            xyzs_i = (x_i.coords[:, 1:].float() + 0.5) / resolution - 0.5
            uvs_i, _ = project_cv(xyzs_i, intrinsics[i], extrinsics[i])
            if is_multiview:
                uvs_i = uvs_i.permute(1, 0, 2).contiguous()
            uvs.append(uvs_i)
        uvs = torch.cat(uvs, dim=0).clamp(0.0, 1.0)
        return SparseTensor(feats=uvs, coords=x.coords)

    def _sample_ssuv(self, cond: torch.Tensor, **kwargs):
        """
        Obtain ssuv predictions by full sampling from pure noise. Overrides
        FlowMatchingEvaluator._sample_ssuv because SFM:
        - has its own `ssuv_denoiser` model key (falls back to `denoiser`),
        - cannot use `torch.randn_like(x_0)` since SFM's `x_0` is a SparseTensor (slat).
        """
        args = self.get_inference_cond(cond, type='ssuv', **kwargs)

        ssuv_denoiser = self.models['ssuv_denoiser'] if 'ssuv_denoiser' in self.models else self.models['denoiser']
        in_channels, res = ssuv_denoiser.in_channels, ssuv_denoiser.resolution
        sampling_noise = torch.randn((cond.shape[0], in_channels, res, res, res), dtype=cond.dtype, device=cond.device)

        result = self.ssuv_sampler.sample(
            model=ssuv_denoiser,
            noise=sampling_noise,
            **args,
            **self.ssuv_sampler_params,
            verbose=False,
        )
        pred_x_0 = result.samples
        return self._decode_ssuv(pred_x_0)

    @staticmethod
    def _debug_saver_cls():
        from .debug_saver import SparseDebugSaver
        return SparseDebugSaver

    def _save_mesh(self, indices, preds, extrinsics, intrinsics):
        batch_size = indices.shape[0]
        mesh_dir = os.path.join(self.results_root, 'mesh')
        instance_indices, _ = self.get_instance_view_indices(indices.cpu().numpy())
        sha256s = self.instances[instance_indices][:, 1]
        preds['pose'] = [{'extrinsic': extrinsics[i], 'intrinsic': intrinsics[i]} for i in range(batch_size)]
        for i in range(batch_size):
            mesh_dir_i = os.path.join(mesh_dir, f'{indices[i]}_{sha256s[i]}')
            os.makedirs(mesh_dir_i, exist_ok=True)
            save_outputs_for_instance(preds, mesh_dir_i, i, verbose=False)


class SparseFlowMatchingCFGEvaluator(ClassifierFreeGuidanceMixin, SparseFlowMatchingEvaluator):
    """
    Evaluator for Sparse Flow Matching latent model with SLat decoder and classifier-free guidance.
    """
    def get_inference_cond(self, cond, type: Literal['ssuv', 'slat'], **kwargs):
        """
        Get the conditioning data for inference.
        """
        return super().get_inference_cond(cond, **kwargs)


class MultiPoseSparseFlowMatchingEvaluator(MultiPoseFlowMatchingEvaluator, SparseFlowMatchingEvaluator):
    """
    Evaluator for multi-pose Sparse Flow Matching that jointly denoises structure and poses.
    Supports both single-view (V=1) and multi-view (V>1) evaluation.

    Inherits _decode_ssuv from MultiPoseFlowMatchingEvaluator and all of
    SparseFlowMatchingEvaluator's slat decoding + rendering pipeline. Overrides
    the full-sampling noise construction (uses ssuv_denoiser attrs instead of
    x_0 shapes, since SFM's x_0 is a SparseTensor slat latent).

    See SparseFlowMatchingEvaluator for inherited args (alignment helpers,
    point-cloud sampling kwargs, evaluate_3d_geometry).
    """

    def _sample_ssuv(self, cond: torch.Tensor, **kwargs):
        """Override: multi-pose SSUV sampling uses `ssuv_denoiser` attrs to build noise.

        Returns ``(final_preds, step_payload)``; ``step_payload`` is ``None``
        when ``eval_intermediate_steps`` is off.
        """
        args = self.get_inference_cond(cond, type='ssuv', **kwargs)

        ssuv_denoiser = self.models['ssuv_denoiser'] if 'ssuv_denoiser' in self.models else self.models['denoiser']
        N = cond.shape[0]
        N_views = cond.shape[1] if cond.ndim == 5 else 1

        shape_in_channels, res = ssuv_denoiser.shape_in_channels, ssuv_denoiser.resolution
        x_noise = torch.randn((N, shape_in_channels, res, res, res), dtype=cond.dtype, device=cond.device)
        pose_in_channels = ssuv_denoiser.pose_in_channels
        pose_res = ssuv_denoiser.pose_resolution
        P_noise = torch.randn((N, N_views, pose_in_channels, pose_res, pose_res, pose_res), dtype=cond.dtype, device=cond.device)

        result = self.ssuv_sampler.sample(
            model=ssuv_denoiser,
            noise=(x_noise, P_noise),
            **args,
            **self.ssuv_sampler_params,
            verbose=False,
        )
        x_pred, P_pred = result.samples
        final_preds = self._decode_ssuv(x_pred, P_pred)
        step_payload = self._build_step_payload(result, ('pred_x_0', 'pred_P_0'))
        return final_preds, step_payload

    def evaluation_metrics(self, indices, **data):
        assert data['cond'].ndim == 5, "Multi-pose evaluation requires [N, V, ...] cond"
        N, V = data['cond'].shape[0:2]

        if self.save_debug:
            self.debug_saver.begin_sample()
        try:
            return self._evaluation_metrics_inner(indices, N, V, **data)
        finally:
            if self.save_debug:
                self.debug_saver.end_sample()

    def _evaluation_metrics_inner(self, indices, N, V, **data):
        debug_extras = {} if self.save_debug else None
        ssuv_step_payload = None
        if self.sample_ssuv:
            if self.sample_ssuv == 'gt_latent':
                ssuv_preds = self._decode_ssuv(data['x_0'], data['P_0'])
                extrinsics_pred = data['extrinsics']
                intrinsics_pred = data['intrinsics']
                raw_intrinsics_pred = data['raw_intrinsics']
            else:
                ssuv_preds, ssuv_step_payload = self._sample_ssuv(**data)
                if self.enable_uv:
                    ssuv_flat = ssuv_preds['ssuv_recon'].flatten(0, 1)
                    uv_flat = ssuv_preds['uv_volume_pred'].flatten(0, 1)
                    ext_flat, int_flat, _ = self._solve_pose_from_ssuv(ssuv_flat, uv_flat)
                    extrinsics_pred = ext_flat.unflatten(0, (N, V))
                    intrinsics_pred = int_flat.unflatten(0, (N, V))
                    crop_bbox = data.get('crop_bbox')
                    if crop_bbox is not None:
                        raw_intrinsics_pred = self._adjust_raw_intrinsics(int_flat, crop_bbox.flatten(0, 1)).unflatten(0, (N, V))
                    else:
                        raw_intrinsics_pred = intrinsics_pred
                else:
                    extrinsics_pred = data['extrinsics']
                    intrinsics_pred = data['intrinsics']
                    raw_intrinsics_pred = data['raw_intrinsics']
            noise = self._make_randn_sparse_latent_from_ss(ssuv_preds['ss_recon'])
            if self.enable_uv:
                data['uvs'] = self._make_sparse_uvs_from_pose(noise, extrinsics_pred, intrinsics_pred)
            preds = self._sample_slat(**data, noise=noise)
        else:
            extrinsics_pred = data['extrinsics']
            intrinsics_pred = data['intrinsics']
            raw_intrinsics_pred = data['raw_intrinsics']
            # No ssuv sampling pass in this branch — pose-only test-view
            # denoising would have no x_pred_latent to condition on. We still
            # define ssuv_preds so the NV branch can guard cleanly below.
            ssuv_preds = None
            # The slat denoiser (e.g. VisualLatent / ConvNeXtVisualLatent variants)
            # requires per-voxel ``uvs``. Pre-build the noise from GT occupancy
            # and project voxel centers via GT pose so we reuse the same coords
            # for both noise and uvs (mirrors the sample_ssuv=True branch).
            if self.enable_uv:
                if isinstance(data.get('x_0'), SparseTensor):
                    noise = data['x_0'].replace(torch.randn_like(data['x_0'].feats))
                else:
                    ss_for_noise = data['ss'][:, 0] if data['ss'].ndim == 6 else data['ss']
                    noise = self._make_randn_sparse_latent_from_ss(ss_for_noise)
                data['uvs'] = self._make_sparse_uvs_from_pose(noise, extrinsics_pred, intrinsics_pred)
                preds = self._sample_slat(**data, noise=noise)
            else:
                preds = self._sample_slat(**data)

        # --- Stage 2: pre-sample 3D point clouds (pred surface + GT mesh)
        # when either C2 alignment or 3D-geom metric is on. Both paths consume
        # the same point sets, so we sample once per batch.
        needs_3d_pcd = self.evaluate_3d_geometry or self.alignment_method == "full_pointcloud"
        if needs_3d_pcd:
            self._stage2_pcd = self._sample_3d_pointclouds(preds['mesh'], indices)
        else:
            self._stage2_pcd = None

        # --- Alignment: fit similarity (output → GT frame) once. Under
        # ``alignment_method='none'`` (default) this returns None and all
        # subsequent renders use raw pred extrinsics — bit-identical to prior
        # behavior. Under aligned modes, GT-world cams are inverse-mapped
        # into pred-world for rendering pred outputs.
        align = self._fit_alignment(
            extrinsics_pred=extrinsics_pred,
            intrinsics_pred=intrinsics_pred,
            data=data,
            preds=preds,
        )

        # --- Render at all V poses and evaluate ---
        cond_prefix = 'raw_' if self.metrics_on_raw_cond else ''

        # Effective input-view extrinsics + intrinsics:
        #   * ``alignment_method='none'``: render at pred (E, K) (existing behavior).
        #   * aligned modes: render at the GT-cam-equivalent extrinsic in pred frame
        #     paired with **GT K** — this way the cond render matches the GT image's
        #     projection up to model error only, with no residual K mismatch from
        #     pose-solver focal/principal-point noise.
        if align is None:
            ext_input_eff = extrinsics_pred
            cond_intrinsics = raw_intrinsics_pred if self.metrics_on_raw_cond else intrinsics_pred
        else:
            ext_input_eff = self._apply_alignment_inverse(data['extrinsics'], align)
            cond_intrinsics = data[f'{cond_prefix}intrinsics']

        # Photometric — _render_reps handles [N, V, 4, 4] → [N, V, 3, H, W]
        renders = self._render_reps(preds['gaussian'], ext_input_eff, cond_intrinsics)
        gt_images = data[f'{cond_prefix}cond']
        gt_flat = gt_images.reshape(N * V, *gt_images.shape[2:])
        render_flat = renders['color'].reshape(N * V, *renders['color'].shape[2:])
        cond_renders_vis_mv = self._render_color_for_vis(preds['gaussian'], ext_input_eff, cond_intrinsics)
        cond_gt_vis_mv = self._compose_gt_for_vis(data.get(f'{cond_prefix}cond_rgba'))
        cond_render_vis_flat = (cond_renders_vis_mv.reshape(N * V, *cond_renders_vis_mv.shape[2:])
                                if cond_renders_vis_mv is not None else None)
        cond_gt_vis_flat = (cond_gt_vis_mv.reshape(N * V, *cond_gt_vis_mv.shape[2:])
                            if cond_gt_vis_mv is not None else None)
        photo_metrics, photo_samples = self.evaluate_photometric(
            'cond', gt_flat, render_flat,
            images_vis=cond_gt_vis_flat, renders_vis=cond_render_vis_flat,
        )
        # Save once with natural [N, V, ...] shape (save_images handles 5D).
        self.save_images('cond_color', indices,
            cond_gt_vis_mv if cond_gt_vis_mv is not None else gt_images,
            cond_renders_vis_mv if cond_renders_vis_mv is not None else renders['color'])

        if debug_extras is not None:
            # Visualizer ↔ raw/cropped toggle: render the predicted view with
            # *both* sets of intrinsics so the user can swap between them.
            # The metric path already rendered at `cond_intrinsics`; render the
            # other variant once more for vis. Only fires when save_debug is
            # active so eval-only runs don't pay the extra cost.
            if align is None:
                alt_intrinsics = (intrinsics_pred if self.metrics_on_raw_cond
                                  else raw_intrinsics_pred)
            else:
                alt_intrinsics = (data['intrinsics'] if self.metrics_on_raw_cond
                                  else data['raw_intrinsics'])
            alt_renders_vis_mv = self._render_color_for_vis(
                preds['gaussian'], ext_input_eff, alt_intrinsics)
            primary_render = (cond_renders_vis_mv if cond_renders_vis_mv is not None
                              else renders['color'])
            alt_render = (alt_renders_vis_mv if alt_renders_vis_mv is not None
                          else self._render_reps(
                              preds['gaussian'], ext_input_eff, alt_intrinsics)['color'])
            if self.metrics_on_raw_cond:
                cond_render_raw, cond_render_cropped = primary_render, alt_render
                cond_K_raw, cond_K_cropped = cond_intrinsics, alt_intrinsics
            else:
                cond_render_raw, cond_render_cropped = alt_render, primary_render
                cond_K_raw, cond_K_cropped = alt_intrinsics, cond_intrinsics
            debug_extras.update({
                'ssuv_step_payload': ssuv_step_payload,
                'align': align,
                'extrinsics_pred': extrinsics_pred,
                'ext_input_eff': ext_input_eff,
                'cond_intrinsics_used': cond_intrinsics,
                'cond_render_color_raw':     cond_render_raw,
                'cond_render_color_cropped': cond_render_cropped,
                'cond_intrinsics_raw':       cond_K_raw,
                'cond_intrinsics_cropped':   cond_K_cropped,
                'cond_gt_for_save': cond_gt_vis_mv if cond_gt_vis_mv is not None else gt_images,
                'cond_gt_rgba': data.get(f'{cond_prefix}cond_rgba'),
                'final_preds': preds,
            })

        # Geometric
        renders_mesh = self._render_reps(preds['mesh'], ext_input_eff, cond_intrinsics)
        renders_mesh['depth'] *= (renders_mesh['mask'] == 1.0)
        if debug_extras is not None:
            debug_extras['cond_render_depth'] = renders_mesh['depth']
        gt_depths = data[f'{cond_prefix}cond_depth']
        gt_depth_flat = gt_depths.reshape(N * V, *gt_depths.shape[2:])
        render_depth_flat = renders_mesh['depth'].reshape(N * V, *renders_mesh['depth'].shape[2:])
        gt_ext_flat = data['extrinsics'].reshape(N * V, *data['extrinsics'].shape[2:])
        gt_int_flat = data[f'{cond_prefix}intrinsics'].reshape(N * V, *data[f'{cond_prefix}intrinsics'].shape[2:])
        depth_metrics, depth_samples, cond_pointmaps = self.evaluate_geometry(
            'cond',
            gt_depth_flat, render_depth_flat, gt_ext_flat, gt_int_flat)
        # Depth: reshape normalized [N*V, 1, H, W] -> [N, V, 1, H, W] and save.
        depth_gt_5d = depth_samples['cond_depth_gt']['value'].reshape(N, V, *depth_samples['cond_depth_gt']['value'].shape[1:])
        depth_rd_5d = depth_samples['cond_depth_render']['value'].reshape(N, V, *depth_samples['cond_depth_render']['value'].shape[1:])
        self.save_images('cond_depth', indices, depth_gt_5d, depth_rd_5d)
        # Pointmaps: list of N*V variable-length tensors, save with per-view tag.
        cond_view_indices = torch.arange(V).repeat(N)
        self.save_pointmaps('cond_pointmap', indices.repeat_interleave(V),
                            cond_pointmaps['gt'], cond_pointmaps['rd'],
                            view_indices=cond_view_indices)

        # Per-view and average metrics across V views → [N]
        metrics = {}
        for k, v in {**photo_metrics, **depth_metrics}.items():
            v_per_view = v.reshape(N, V)
            metrics[k] = v_per_view.mean(dim=1)
            for vi in range(V):
                metrics[f'{k}_v{vi}'] = v_per_view[:, vi]

        # Samples: reshape [N*V, ...] -> [N, V, ...] as mv_image for all-view subgrid visualization
        samples = {}
        for k, v in {**photo_samples, **depth_samples}.items():
            if isinstance(v, dict) and 'value' in v:
                val = v['value']
                if isinstance(val, torch.Tensor) and val.shape[0] == N * V:
                    samples[k] = {'value': val.reshape(N, V, *val.shape[1:]), 'type': 'mv_image'}
                else:
                    samples[k] = v
            else:
                samples[k] = v

        if self.evaluate_novel_view:
            assert 'test_extrinsics' in data, (
                "evaluate_novel_view=True but the dataset batch contains no test_* keys; "
                "set eval_test_views=[...] on the eval_dataset (MultiViewMixin) so "
                "_gather_test_views populates test_extrinsics / test_intrinsics / test_cond."
            )
            # Test (NV) cams: by default pull GT-world test cams into pred-world
            # via the inverse alignment (see ``_apply_alignment_inverse``). When
            # ``denoise_test_view_poses=True`` is set on the evaluator, instead
            # run pose-only denoising for the test views (with x_pred pinned
            # clean) — the resulting extrinsics live in pred frame directly,
            # so no alignment fit is involved for NV rendering.
            nv_images = data[f'test_{cond_prefix}cond']                              # [N, M, 3, H, W] or [N, 3, H, W]
            nv_intrinsics_gt = data[f'test_{cond_prefix}intrinsics']
            if self.denoise_test_view_poses:
                # ``ssuv_preds`` (not ``preds``) carries the raw latents written
                # by ``_decode_ssuv``. ``preds`` here is the slat output dict.
                assert ssuv_preds is not None, (
                    "denoise_test_view_poses=True requires sample_ssuv to be "
                    "truthy (or 'gt_latent') so x_pred_latent is available; "
                    f"got sample_ssuv={self.sample_ssuv!r}.")
                test_ext_pred, test_K_pred = self._sample_test_view_poses(
                    x_pred_latent=ssuv_preds['x_pred_latent'],
                    P_input_pred_latent=(ssuv_preds['P_pred_latent']
                                         if self.num_input_views_in_joint > 0 else None),
                    input_cond=(data['cond']
                                if self.num_input_views_in_joint > 0 else None),
                    test_cond=data['test_cond'],
                    test_intrinsics=data['test_intrinsics'],
                )
                nv_extrinsics_eff = test_ext_pred
                # Reuse the just-sampled poses for nv_* fidelity metrics — the
                # base evaluator's _compute_metrics_from_preds hook isn't
                # called on this overridden path, so emit them here.
                nv_pose_metrics = self._evaluate_test_view_poses(
                    preds=None, data=data, ext_test=test_ext_pred, K_test=test_K_pred)
                metrics.update(nv_pose_metrics)
            else:
                nv_extrinsics_eff = self._apply_alignment_inverse(data['test_extrinsics'], align)
            nv_renders = self._render_reps(preds['gaussian'], nv_extrinsics_eff, nv_intrinsics_gt)
            nv_renders_vis_mv = self._render_color_for_vis(preds['gaussian'], nv_extrinsics_eff, nv_intrinsics_gt)
            nv_gt_vis_mv = self._compose_gt_for_vis(data.get(f'test_{cond_prefix}cond_rgba'))

            if debug_extras is not None:
                debug_extras['nv'] = {
                    'gt_for_save': nv_gt_vis_mv if nv_gt_vis_mv is not None else nv_images,
                    'gt_rgba': data.get(f'test_{cond_prefix}cond_rgba'),
                    'extrinsics_gt': data['test_extrinsics'],
                    'intrinsics_gt': data['test_intrinsics'],
                    'extrinsics_pred': nv_extrinsics_eff,
                    'intrinsics_pred': (test_K_pred if self.denoise_test_view_poses
                                        else nv_intrinsics_gt),
                    'render_color': (nv_renders_vis_mv if nv_renders_vis_mv is not None
                                     else nv_renders['color']),
                    'pose_source': ('denoised' if self.denoise_test_view_poses
                                    else 'aligned' if align is not None
                                    else 'gt'),
                }

            if nv_images.ndim == 5:
                M = nv_images.shape[1]
                nv_gt_flat = nv_images.reshape(N * M, *nv_images.shape[2:])
                nv_render_flat = nv_renders['color'].reshape(N * M, *nv_renders['color'].shape[2:])
                nv_render_vis_flat = (nv_renders_vis_mv.reshape(N * M, *nv_renders_vis_mv.shape[2:])
                                      if nv_renders_vis_mv is not None else None)
                nv_gt_vis_flat = (nv_gt_vis_mv.reshape(N * M, *nv_gt_vis_mv.shape[2:])
                                  if nv_gt_vis_mv is not None else None)
                nv_photo_metrics, nv_photo_samples = self.evaluate_photometric(
                    'nv', nv_gt_flat, nv_render_flat,
                    images_vis=nv_gt_vis_flat, renders_vis=nv_render_vis_flat,
                )
                # Save once with natural [N, M, ...] shape.
                self.save_images('nv_color', indices,
                    nv_gt_vis_mv if nv_gt_vis_mv is not None else nv_images,
                    nv_renders_vis_mv if nv_renders_vis_mv is not None else nv_renders['color'])
                for k, v in nv_photo_metrics.items():
                    v_per_view = v.reshape(N, M)
                    metrics[k] = v_per_view.mean(dim=1)
                    for mi in range(M):
                        metrics[f'{k}_v{mi}'] = v_per_view[:, mi]
                for k, v in nv_photo_samples.items():
                    if isinstance(v, dict) and 'value' in v:
                        val = v['value']
                        if isinstance(val, torch.Tensor) and val.shape[0] == N * M:
                            samples[k] = {'value': val.reshape(N, M, *val.shape[1:]), 'type': 'mv_image'}
                        else:
                            samples[k] = v
                    else:
                        samples[k] = v
            else:
                nv_photo_metrics, nv_photo_samples = self.evaluate_photometric(
                    'nv', nv_images, nv_renders['color'],
                    images_vis=nv_gt_vis_mv, renders_vis=nv_renders_vis_mv,
                )
                metrics.update(nv_photo_metrics)
                samples.update(nv_photo_samples)
                self.save_images('nv_color', indices,
                    nv_gt_vis_mv if nv_gt_vis_mv is not None else nv_images,
                    nv_renders_vis_mv if nv_renders_vis_mv is not None else nv_renders['color'])

            if self.evaluate_clip_metric:
                # CLIP eval expects [N, 3, H, W] cond + [N, M, 3, H, W] nv. Pick the
                # first input view as the reference cond (multi-view CLIP eval is
                # not part of this scope).
                cond_for_clip = data[f'{cond_prefix}cond']
                if cond_for_clip.ndim == 5:
                    cond_for_clip = cond_for_clip[:, 0]
                nv_clip_metrics, nv_clip_samples = self.evaluate_novel_view_clip(
                    cond_for_clip, nv_images, nv_renders['color'])
                metrics.update(nv_clip_metrics)
                samples.update(nv_clip_samples)

        if self.save_mesh:
            # Pass full [N, V, 4, 4] / [N, V, 3, 3] so per-instance pose entries
            # carry all V views (downstream JSON serializer handles tensor shape).
            self._save_mesh(indices, preds, extrinsics_pred, cond_intrinsics)

        # --- Stage 2: 3D-CD / 3D-F@τ on full point clouds (protocol §4.2). Pred
        # points live in pred frame; transform to GT frame via the forward
        # similarity (no-op when alignment_method='none').
        if self.evaluate_3d_geometry:
            pred_pts, gt_pts = self._stage2_pcd
            if align is not None:
                pred_pts_gt = transform_points_by_similarity(
                    pred_pts, align['s'], align['R_pa'], align['T_pa'])
            else:
                pred_pts_gt = pred_pts
            pred_list = [pred_pts_gt[n] for n in range(N)]
            gt_list = [gt_pts[n] for n in range(N)]
            geom3d_metrics, _ = self._evaluate_3d_geometry(
                'cond', indices, pred_list, gt_list, (0.01, 0.05, 0.10))
            metrics.update(geom3d_metrics)

        # Drop the per-batch point cache to free memory before the next batch.
        self._stage2_pcd = None

        if self.evaluate_pose and self.sample_ssuv:
            pose_metrics, _ = self._evaluate_poses(
                indices=indices,
                uv_volume=data.get('uv_volume'), ss=data.get('ss'), ssuv=data.get('ssuv'),
                uv_volume_pred=ssuv_preds['uv_volume_pred'], ssuv_recon=ssuv_preds['ssuv_recon'],
                intrinsics=data['intrinsics'], extrinsics=data['extrinsics'],
            )
            metrics.update(pose_metrics)

        if debug_extras is not None:
            # Sparse path: SparseDebugSaver decodes intermediates internally and
            # consumes debug_extras for stage-2 finals + alignment + NV block.
            self.debug_saver.write(indices, data, debug_steps=None, debug_extras=debug_extras)

        return metrics, samples


class MultiPoseSparseFlowMatchingCFGEvaluator(ClassifierFreeGuidanceMixin, MultiPoseSparseFlowMatchingEvaluator):
    """
    Multi-pose Sparse Flow Matching evaluator with classifier-free guidance.
    """
    pass


class VisualImageConditionedMultiPoseSparseFlowMatchingEvaluator(VisualImageConditionedMixin, MultiPoseSparseFlowMatchingCFGEvaluator):
    """
    Visual image-conditioned multi-pose Sparse Flow Matching evaluator.

    Args:
        ssuv_uses_visual: If False (default), strip visual_cond for the ssuv stage
            so the ssuv denoiser receives DINO-only cond (for models like
            MultiPoseSparseStructureFlowModel). If True, keep the dict cond for ssuv
            (for models like VisualMultiPoseSparseStructureFlowModel).
    """
    def __init__(self, *args, ssuv_uses_visual: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.ssuv_uses_visual = ssuv_uses_visual

    def encode_image(self, image):
        """Handle both single-view [B,3,H,W] and multi-view [B,V,3,H,W]."""
        if isinstance(image, torch.Tensor) and image.ndim == 5:
            B, V = image.shape[:2]
            features = super().encode_image(image.reshape(B * V, *image.shape[2:]))
            return features.reshape(B, V, *features.shape[1:])
        return super().encode_image(image)

    def get_inference_cond(self, cond, type: Literal['ssuv', 'slat'], **kwargs):
        args = super().get_inference_cond(cond, type=type, **kwargs)
        if type == 'ssuv':
            if self.ssuv_uses_visual:
                # Keep dict cond; ensure V dim present for both tensors.
                c, nc = args['cond'], args['neg_cond']
                if c['dino_cond'].ndim == 3:  # single-view [B, L, C] → add V=1
                    for k in list(c.keys()):
                        c[k] = c[k].unsqueeze(1)
                        nc[k] = nc[k].unsqueeze(1)
            else:
                dino = args['cond']['dino_cond']
                neg_dino = args['neg_cond']['dino_cond']
                # Add V dim if not present (single-view encode → [B, L, C])
                if dino.ndim == 2:
                    dino = dino.unsqueeze(1)
                    neg_dino = neg_dino.unsqueeze(1)
                args['cond'] = dino
                args['neg_cond'] = neg_dino
        return args
