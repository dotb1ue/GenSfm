import os
from typing import *
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode

from .base import Evaluator
from ..utils.loss_utils import psnr, ssim, lpips
from ..utils.voxel_utils import write_voxels
from ..utils.geometry_utils import *
from ..utils.render_utils import get_renderer


class BasicSingleViewEvaluator(Evaluator):
    """
    Evaluator for basic single-view flow matching.

    Args:
        evaluate_novel_view (bool): Whether to evaluate photometric metrics on the
            test-view (T) renders. Requires the eval dataset to be configured with
            ``return_test_views=True`` so each batch carries ``test_cond``,
            ``test_extrinsics``, and ``test_intrinsics`` (and the ``raw_*`` variants
            when ``metrics_on_raw_cond=True``). Default is False.
        evaluate_cross_view (bool): Whether to evaluate the 4 cross view images. Default is False.
        evaluate_clip_metric (bool): Whether to evaluate the CLIP score. Default is False.
        clip_models (List[str]): List of CLIP models to evaluate. Default is ['ViT-B/32', 'ViT-B/16', 'ViT-L/14'].
        cross_view_axis (Tuple[float, float, float]): Axis to rotate the cross view images. Default is (0.0, 0.0, 1.0).
        cross_view_pivot (Tuple[float, float, float]): Pivot to rotate the cross view images. Default is (0.0, 0.0, 0.0).
        cross_view_angles (List[float]): Angles to rotate the cross view images. Default is [torch.pi / 2, torch.pi, 3 * torch.pi / 2].
        save_results (bool): Whether to save the rendered results. Default is False.
        save_nv_as_grid (bool): Whether to save the novel view results as a grid. Default is True.
        render_resolution (int): Render resolution used by ``_render_reps`` and the
            depth/mask renderer. Default is 518 (matches the dataset image_size most
            evaluators use).
        render_bg_color (Tuple[float, float, float]): Background color for *visualization*
            (sample tensors and ``save_results`` outputs). Metrics always render on black
            to match the dataset's eval-mode GT (premultiplied on black). Default is (0, 0, 0).
    """
    def __init__(
        self,
        *args,
        evaluate_novel_view: bool = False,
        evaluate_cross_view: bool = False,
        evaluate_clip_metric: bool = False,
        clip_models: List[str] = ['ViT-B/32', 'ViT-B/16', 'ViT-L/14'],
        cross_view_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        cross_view_pivot: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        cross_view_angles: List[float] = [0, torch.pi / 2, torch.pi, 3 * torch.pi / 2],
        save_results: bool = False,
        save_nv_as_grid: bool = True,
        render_resolution: int = 518,
        render_bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.evaluate_novel_view = evaluate_novel_view
        self.evaluate_cross_view = evaluate_cross_view
        self.evaluate_clip_metric = evaluate_clip_metric
        self.clip_models = clip_models
        self.cross_view_axis = cross_view_axis
        self.cross_view_pivot = cross_view_pivot
        self.cross_view_angles = cross_view_angles
        self.save_results = save_results
        self.save_nv_as_grid = save_nv_as_grid
        self.render_resolution = int(render_resolution)
        self.render_bg_color = tuple(render_bg_color)
        if self.evaluate_clip_metric:
            import clip
            self.clip_models = {}
            for name in clip_models:
                clip_model = clip.load(name, device='cuda')[0]
                try:
                    input_resolution = clip_model.input_resolution.item()
                except:
                    input_resolution = clip_model.visual.input_resolution
                clip_transform = Compose([
                    Resize(input_resolution, InterpolationMode.BICUBIC),
                    CenterCrop(input_resolution),
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ])
                self.clip_models[name] = (clip_model, clip_transform)

    # Rep-agnostic renderer; ``get_renderer`` dispatches by ``type(reps[0])``.
    def _render_reps(self, reps, extrinsics, intrinsics,
                     bg_color=(0.0, 0.0, 0.0), return_types=None):
        # Default black bg matches the dataset's eval-mode GT (premultiplied on black);
        # vis callers override. ``return_types=None`` falls back to the renderer's default.
        render_options = {'resolution': self.render_resolution, 'bg_color': bg_color}
        renderer = get_renderer(reps[0], **render_options)

        batch_shape = extrinsics.shape[:-2]
        if extrinsics.ndim > 3:
            extrinsics = extrinsics.flatten(start_dim=1, end_dim=-3).unbind(1)
            intrinsics = intrinsics.flatten(start_dim=1, end_dim=-3).unbind(1)
        else:
            extrinsics = [extrinsics]
            intrinsics = [intrinsics]

        render_kwargs = {} if return_types is None else {'return_types': return_types}
        rets = {}
        for i, rep in enumerate(reps):
            for extrs, intrs in zip(extrinsics, intrinsics):
                res = renderer.render(rep, extrs[i], intrs[i], **render_kwargs)
                for k, v in res.items():
                    rets.setdefault(k, []).append(v)

        for k, v in rets.items():
            rets[k] = torch.stack(v, dim=0).reshape(batch_shape + v[0].shape)
        return rets

    # Per-view pred (depth, mask) shared by C1 alignment and cond-geometry metrics.
    # Routes through MeshRenderer by default; falls back to GaussianRenderer when the
    # mesh has no faces, thresholding splat alpha to
    # produce a tight mask the LSQ can trust.
    def _render_pred_depth_mask(
        self,
        preds: dict,
        extrinsics: torch.Tensor,                          # [N, V, 4, 4] or [N, 4, 4]
        intrinsics: torch.Tensor,                          # [N, V, 3, 3] or [N, 3, 3]
        *,
        alpha_threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if alpha_threshold is None:
            alpha_threshold = float(getattr(self, 'pred_depth_alpha_threshold', 0.95))
        squeezed = (extrinsics.ndim == 3)
        if squeezed:
            extrinsics = extrinsics.unsqueeze(1)
            intrinsics = intrinsics.unsqueeze(1)
        gauss_list = preds.get('gaussian')
        mesh_list = preds['mesh']
        use_gauss_depth = (
            gauss_list is not None
            and any(getattr(m, 'faces', None) is not None and m.faces.numel() == 0
                    for m in mesh_list)
        )
        if use_gauss_depth:
            renders = self._render_reps(
                gauss_list, extrinsics, intrinsics,
                return_types=("depth", "alpha"),
            )
            pred_depth = renders['depth']
            pred_alpha = renders['alpha']
            pred_mask = (pred_alpha > alpha_threshold) & (pred_depth > 0)
        else:
            renders = self._render_reps(mesh_list, extrinsics, intrinsics)
            pred_depth = renders['depth']
            pred_mask = (renders['mask'] == 1.0)
        pred_depth = pred_depth * pred_mask
        if squeezed:
            pred_depth = pred_depth.squeeze(1)
            pred_mask = pred_mask.squeeze(1)
        return pred_depth, pred_mask

    def evaluate_photometric(self, prefix, images, renders,
                             *, images_vis=None, renders_vis=None):
        """Compute photometric metrics + sample dict (no disk I/O).

        ``images`` / ``renders`` always drive the metrics. ``images_vis`` /
        ``renders_vis`` (when provided) override what is emitted as visual
        samples. Callers are responsible for calling ``save_images`` if disk
        saves are desired.
        """
        assert images.ndim == renders.ndim == 4, "Images and renders must be 4D tensors of shape [B, C, H, W]."
        assert images.shape[0] == renders.shape[0], "Images and renders must have the same batch size."
        assert images.shape[1] == renders.shape[1] == 3, "Images and renders must have 3 channels of RGB."
        metrics, samples = {}, {}

        # Rescale renders to match the size of GT images
        if images.shape[2:] != renders.shape[2:]:
            renders = F.interpolate(renders, size=images.shape[2:], mode='bilinear', align_corners=False, antialias=True)

        # Evaluate photometric metrics
        psnr_list, ssim_list, lpips_list = [], [], []
        for rgb_gt, rgb_render in zip(images, renders):
            psnr_list.append(psnr(rgb_gt, rgb_render))
            ssim_list.append(ssim(rgb_gt, rgb_render))
            lpips_list.append(lpips(rgb_gt, rgb_render))
        metrics[f'{prefix}_psnr'] = torch.stack(psnr_list, dim=0)
        metrics[f'{prefix}_ssim'] = torch.stack(ssim_list, dim=0)
        metrics[f'{prefix}_lpips'] = torch.stack(lpips_list, dim=0)

        # Pick what gets visualized; mirror the metric-side resize so the GT-
        # vs-render absdiff stays meaningful even at different render sizes.
        images_save = images_vis if images_vis is not None else images
        renders_save = renders_vis if renders_vis is not None else renders
        if renders_save.shape[2:] != images_save.shape[2:]:
            renders_save = F.interpolate(renders_save, size=images_save.shape[2:],
                                         mode='bilinear', align_corners=False, antialias=True)

        # Save visual samples
        samples[f'{prefix}_rgb_gt'] = {'value': images_save.reshape(-1, *images_save.shape[-3:]), 'type': 'image'}
        samples[f'{prefix}_rgb_render'] = {'value': renders_save.reshape(-1, *renders_save.shape[-3:]), 'type': 'image'}
        samples[f'{prefix}_rgb_absdiff'] = {'value': (images_save - renders_save).abs().reshape(-1, *renders_save.shape[-3:]), 'type': 'image'}

        return metrics, samples

    def evaluate_geometry(self, prefix, depths, depths_render, extrinsics, intrinsics, align_method='affine'):
        """Compute depth/pointmap metrics + sample dict and return pointmaps.

        Pure metric computation; emits no disk files. Returns a 3-tuple
        ``(metrics, samples, pointmaps)`` where ``pointmaps`` is a dict with
        ``'gt'`` and ``'rd'`` lists of per-instance point clouds. Callers
        are responsible for ``save_images`` (depth) and ``save_pointmaps``
        if disk saves are desired.
        """
        assert depths.ndim == depths_render.ndim == 3, "Depths must be 3D tensors of shape [B, H, W]."
        assert depths.shape[0] == depths_render.shape[0], "Depths must have the same batch size."
        metrics, samples = {}, {}

        # Set mask to True where depths are greater than 0
        # Note that the alpha transition regions should be already masked out in the depths
        mask_gt = depths > 0
        mask_render = depths_render > 0

        # Resize depthmap and mask to match the size of GT depths
        if depths.shape[1:] != depths_render.shape[1:]:
            depths_render, mask_render = masked_nearest_resize(depths_render, mask=mask_render, size=depths.shape[1:])

        # Compute the mask IOU
        mask_intersection = mask_gt & mask_render
        mask_union = mask_gt | mask_render
        metrics[f'{prefix}_mask_iou'] = (mask_intersection.sum(dim=[1, 2]) + 1) / (mask_union.sum(dim=[1, 2]) + 1)

        # Align the affine-invariant depth map
        if align_method is None:
            depths_aligned = depths_render
        else:
            depths_aligned = align_depthmap(depths_render, depths, mask_intersection, align_method=align_method)

        # Compute the Chamfer distance and F-score
        cd, pointmap_renders, pointmap_gts = pointmap_chamfer_distance(
            depths_aligned, depths, mask_render, mask_gt, extrinsics, intrinsics, extrinsics, intrinsics)
        metrics[f'{prefix}_chamfer_distance'] = cd
        metrics[f'{prefix}_fscore_0.01'], metrics[f'{prefix}_fscore_0.05'], metrics[f'{prefix}_fscore_0.10'] = \
            point_f_score(pointmap_renders, pointmap_gts, r=[0.01, 0.05, 0.10])

        # Save visual samples
        max_depth = torch.max(torch.maximum(depths_aligned, depths).flatten(start_dim=1), dim=1).values
        max_depth = max_depth[:, None, None, None].clamp_min_(1.0)
        samples[f'{prefix}_depth_gt'] = {'value': depths.unsqueeze(1) / max_depth, 'type': 'image'}
        samples[f'{prefix}_depth_render'] = {'value': depths_aligned.unsqueeze(1) / max_depth, 'type': 'image'}
        samples[f'{prefix}_depth_absdiff'] = {'value': (depths_aligned - depths).abs().unsqueeze(1) / max_depth, 'type': 'image'}

        pointmaps = {'gt': pointmap_gts, 'rd': pointmap_renders}
        return metrics, samples, pointmaps

    def _evaluate_3d_geometry(
        self,
        prefix,
        indices,
        pred_points,                                # list[N] of [P_pred, 3]
        gt_points,                                  # list[N] of [P_gt, 3]
        f_score_thresholds=[0.01, 0.05, 0.10],
    ):
        """3D geometry metrics on full point clouds (protocol §4.2).

        Both clouds must already be in the same frame; cross-frame comparison
        is the caller's responsibility (apply alignment before calling).
        Pure CD / F@τ computation lives in ``utils.geometry_utils``.
        """
        assert len(pred_points) == len(gt_points)
        metrics, samples = {}, {}
        metrics[f'{prefix}_cd_3d'] = point_chamfer_distance(pred_points, gt_points)
        fs = point_f_score(pred_points, gt_points, r=f_score_thresholds)
        for r, f in zip(f_score_thresholds, fs):
            metrics[f'{prefix}_fscore_3d_{r}'] = f
        return metrics, samples

    def evaluate_novel_view_clip(self, images, nv_images, nv_renders):
        """Evaluate the CLIP score between the novel view images and the condition images."""
        assert images.ndim == 4 and images.shape[0] == nv_images.shape[0], "Images must be 4D tensors of shape [B, C, H, W]."
        assert nv_images.ndim == nv_renders.ndim == 5, "Images and renders must be 5D tensors of shape [B, N, C, H, W]."
        assert nv_images.shape[:2] == nv_renders.shape[:2], "Images and renders must have the same batch size and number of novel views."
        assert images.shape[1] == nv_images.shape[2] == nv_renders.shape[2] == 3, "Images and renders must have 3 channels of RGB."
        metrics, samples = {}, {}

        for name, (clip_model, clip_transform) in self.clip_models.items():
            with torch.no_grad():
                preprocessed_images = clip_transform(images)
                preprocessed_nv_images = clip_transform(nv_images.flatten(start_dim=0, end_dim=1))
                preprocessed_nv_renders = clip_transform(nv_renders.flatten(start_dim=0, end_dim=1))
                image_features = clip_model.encode_image(preprocessed_images)
                nv_image_features = clip_model.encode_image(preprocessed_nv_images)
                nv_render_features = clip_model.encode_image(preprocessed_nv_renders)

            cross_clip_score = F.cosine_similarity(
                image_features.unsqueeze(1), 
                nv_render_features.reshape(nv_images.shape[:2] + nv_render_features.shape[-1:]),
                dim=-1,
            )
            metrics[f'nv_cross_clip_{name}'] = cross_clip_score.reshape(nv_images.shape[:2]).mean(dim=1)

            pairwise_clip_score = F.cosine_similarity(nv_image_features, nv_render_features, dim=-1)
            metrics[f'nv_pairwise_clip_{name}'] = pairwise_clip_score.reshape(nv_images.shape[:2]).mean(dim=1)
        
        return metrics, samples

    def evaluate_cross_view_clip(self, images, cv_renders):
        """Evaluate the CLIP score between the cross view images and the condition images."""
        assert images.ndim == 4 and images.shape[0] == cv_renders.shape[0], "Images must be 4D tensors of shape [B, C, H, W]."
        assert cv_renders.ndim == 5, "Images and renders must be 5D tensors of shape [B, N, C, H, W]."
        assert images.shape[1] == cv_renders.shape[2] == 3, "Images and renders must have 3 channels of RGB."
        metrics, samples = {}, {}

        for name, (clip_model, clip_transform) in self.clip_models.items():
            with torch.no_grad():
                preprocessed_images = clip_transform(images)
                preprocessed_cv_renders = clip_transform(cv_renders.flatten(start_dim=0, end_dim=1))
                image_features = clip_model.encode_image(preprocessed_images)
                cv_render_features = clip_model.encode_image(preprocessed_cv_renders)

            cross_clip_score = F.cosine_similarity(
                image_features.unsqueeze(1),
                cv_render_features.reshape(cv_renders.shape[:2] + cv_render_features.shape[-1:]),
                dim=-1,
            )
            metrics[f'cv_cross_clip_{name}'] = cross_clip_score.reshape(cv_renders.shape[:2]).mean(dim=1)
        
        return metrics, samples

    def save_images(self, prefix, indices, gt_images, rd_images, name_suffix: str = ""):
        """Save GT and rendered images. No-op if save_results=False.

        Inputs may be 4D ``[B, C, H, W]`` (one image per index) or 5D
        ``[B, V, C, H, W]`` (V views per index). Naming convention:

        - 4D, or 5D with ``save_nv_as_grid=True``:
          ``{idx}_gt.png``, ``{idx}_rd.png`` (a single image; in the grid
          case it is a torchvision grid composing all V views).
        - 5D with ``save_nv_as_grid=False``:
          ``{idx}_v{vi}_gt.png``, ``{idx}_v{vi}_rd.png`` (one file per view).

        The ``_v{vi}_`` tag always means "view vi"; absence means whole sample.
        Mirrors the convention used by ``save_pointmaps``.
        """
        if not self.save_results:
            return
        image_dir = os.path.join(self.results_root, prefix)
        os.makedirs(image_dir, exist_ok=True)

        def _save_one(t, idx, kind):
            """Save tensor ``t`` (3D image or 4D [V,C,H,W]) under ``{idx}_..._{kind}.png``."""
            if t.ndim > 3:
                assert t.ndim == 4
                if self.save_nv_as_grid:
                    nrow = int(np.sqrt(t.shape[0]))
                    save_image(t, os.path.join(image_dir, f'{idx}_{kind}{name_suffix}.png'),
                               normalize=True, nrow=nrow, value_range=self.dataset.value_range)
                else:
                    for vi in range(t.shape[0]):
                        save_image(t[vi], os.path.join(image_dir, f'{idx}_v{vi}_{kind}{name_suffix}.png'),
                                   normalize=True, value_range=self.dataset.value_range)
            else:
                save_image(t, os.path.join(image_dir, f'{idx}_{kind}{name_suffix}.png'),
                           normalize=True, value_range=self.dataset.value_range)

        for idx, image, render in zip(indices, gt_images, rd_images):
            _save_one(image, idx, 'gt')
            _save_one(render, idx, 'rd')

    def save_pointmaps(self, prefix, indices, gt_pointmaps, rd_pointmaps,
                       view_indices=None):
        """Save the GT pointmaps and rendered pointmaps. No-op if save_results=False.

        ``view_indices`` (optional, parallel to ``indices``) appends ``_v{vi}``
        to file names so multi-view callers avoid filename collisions.
        """
        if not self.save_results:
            return
        pointmap_dir = os.path.join(self.results_root, prefix)
        os.makedirs(pointmap_dir, exist_ok=True)
        
        for i, (idx, gt_pointmap, rd_pointmap) in enumerate(zip(indices, gt_pointmaps, rd_pointmaps)):
            tag = f'_v{int(view_indices[i])}' if view_indices is not None else ''
            write_voxels(os.path.join(pointmap_dir, f'{idx}{tag}_gt.ply'), gt_pointmap.cpu().numpy())
            write_voxels(os.path.join(pointmap_dir, f'{idx}{tag}_rd.ply'), rd_pointmap.cpu().numpy())
            write_voxels(os.path.join(pointmap_dir, f'{idx}{tag}_overlap.ply'), torch.cat([gt_pointmap, rd_pointmap], dim=0).cpu().numpy())
