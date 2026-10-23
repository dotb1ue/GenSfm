"""Output-frame → GT-frame similarity fitting helpers.

Provides alignment and 3D point sampling for GenSfM evaluation: input-view
pointmap least squares, full-pointcloud ICP, and camera-trajectory Umeyama.
Inference uses the predicted frame without fitting to reference geometry.

Subclasses must set the following attributes (typically in their own
``__init__`` — the mixin intentionally avoids one to stay composable):

  - ``self.alignment_method`` ∈ {"none", "trajectory", "input_pointmap", "full_pointcloud"}
  - ``self.num_3d_samples`` (int)
  - ``self.mesh_sample_method`` ∈ {"area", "fps", "render"}
  - ``self.n_3d_render_views`` (int)
  - ``self.n_3d_render_resolution`` (int)
  - ``self.icp_n_iters`` (int)
  - ``self.icp_n_inits`` (int)
  - ``self.gt_mesh_path_template`` (str, with ``{root}``/``{sha256}`` slots)
  - ``self.instances`` (ndarray of (root, sha256) pairs, from base evaluator)
  - ``self.get_instance_view_indices`` (callable, from base evaluator)

The C1 input-pointmap path additionally requires:

  - ``self.pointmap_keep_count`` (Optional[int]; falsy = use all views)

It also calls ``_render_reps`` / ``_render_pred_depth_mask`` and reads
``self.render_resolution``, all provided by ``BasicSingleViewEvaluator``.

Callers must populate ``self._stage2_pcd = (pred_pts, gt_pts)`` before invoking
``_fit_alignment`` with ``alignment_method='full_pointcloud'``.
"""
from typing import *

import torch
import torch.nn.functional as F

from ..utils.voxel_utils import unproject_cv
from ..utils.align_utils import (
    align_extrinsics_by_similarity,
    apply_similarity_to_extrinsics,
    fit_similarity_icp,
    fit_similarity_lsq_pointmaps,
)
from ..utils.geometry_utils import (
    sample_points_from_mesh_extract,
    sample_points_from_ply_path,
)


class AlignmentMixin:
    """Output-frame → GT-frame similarity fitting + rendering helpers.

    See module docstring for required ``self.*`` attributes and the
    ``self._stage2_pcd`` contract.
    """

    # ----------------------------------------------------------------------
    # §1 — General: dispatcher, frame-apply, and shared utilities
    # ----------------------------------------------------------------------

    def _fit_alignment(
        self,
        *,
        extrinsics_pred: torch.Tensor,                    # [N, 4, 4] or [N, V, 4, 4]
        intrinsics_pred: Optional[torch.Tensor],          # matching shape (3, 3)
        data: dict,
        preds: Optional[dict] = None,
        cond_prefix: str = '',
    ) -> Optional[dict]:
        """Dispatch to the alignment fit selected by ``self.alignment_method``.

        ``cond_prefix`` selects the GT depth/intrinsic keys the C1 LSQ reads
        from ``data`` (``''`` → ``cond_depth``/``intrinsics``, ``'raw_'`` →
        ``raw_cond_depth``/``raw_intrinsics``). It must match the prefix the
        evaluator's rendering path uses, otherwise the fitted similarity is
        valid for the LSQ-K but is then applied with a different render-K,
        which biases the projected gauss size by the fx ratio (a real bug
        that surfaces with ``metrics_on_raw_cond=true`` + ``crop_size_ratio≠1``).

        Returns ``None`` when ``alignment_method='none'``, and also from the
        ``'trajectory'`` branch on single-view inputs (``[N, 4, 4]``) where
        Umeyama is rank-deficient — caller falls through to no transform.
        Otherwise returns a dict with at least
        ``{s, R_pa, T_pa, residual, valid}`` (row-vector convention; see
        ``apply_similarity_to_extrinsics``). Branch-specific extras:

          * ``'trajectory'``: also includes ``extrinsics_pred_aligned``
            ([N, V, 4, 4]) from ``align_extrinsics_by_similarity``.
          * ``'input_pointmap'`` with ``pointmap_keep_count < V``: also
            includes ``kept_mask`` ([N, V] bool, False on dropped views).

        ``intrinsics_pred`` is consumed by the ``'input_pointmap'`` branch
        only; the other branches ignore it.
        """
        if self.alignment_method == "none":
            return None
        if self.alignment_method == "trajectory":
            return self._fit_trajectory_alignment(extrinsics_pred, data['extrinsics'])
        if self.alignment_method == "input_pointmap":
            return self._fit_pointmap_alignment(
                extrinsics_pred=extrinsics_pred,
                intrinsics_pred=intrinsics_pred,
                data=data,
                preds=preds,
                cond_prefix=cond_prefix,
            )
        if self.alignment_method == "full_pointcloud":
            return self._fit_pointcloud_alignment()
        raise NotImplementedError(f"alignment_method={self.alignment_method!r}")

    def _apply_alignment_inverse(
        self,
        extrinsics: torch.Tensor,        # [N, 4, 4] or [N, V, 4, 4]
        align: Optional[dict],
    ) -> torch.Tensor:
        """Bring GT-world extrinsics into pred-world frame using the inverse
        of the fitted similarity (output → GT). Used to render pred outputs
        at GT-camera-equivalents so the rendered image is comparable to the
        GT image taken at the GT camera. Returns ``extrinsics`` unchanged
        when ``align is None``. Math: forward similarity (row-vector) is
        ``x_gt = s * x_pred @ R_pa + T_pa``; the inverse is fed back through
        ``apply_similarity_to_extrinsics`` to obtain the equivalent pred-world
        extrinsic.
        """
        if align is None:
            return extrinsics
        s = align['s']                                          # [N]
        R_pa = align['R_pa']                                    # [N, 3, 3]
        T_pa = align['T_pa']                                    # [N, 3]
        R_pa_inv = R_pa.transpose(-1, -2)
        s_inv = 1.0 / s
        T_pa_inv = -torch.einsum("ni,nij->nj", T_pa, R_pa_inv) / s[..., None]

        assert extrinsics.shape[-2:] == (4, 4) and extrinsics.ndim in (3, 4), \
            f"expected [N, 4, 4] or [N, V, 4, 4] extrinsics, got {tuple(extrinsics.shape)}"
        squeezed = extrinsics.ndim == 3
        if squeezed:
            extrinsics = extrinsics.unsqueeze(1)
        R = extrinsics[..., :3, :3]
        t = extrinsics[..., :3, 3]
        R_new, t_new = apply_similarity_to_extrinsics(R, t, s_inv, R_pa_inv, T_pa_inv)
        out = extrinsics.clone()
        out[..., :3, :3] = R_new
        out[..., :3, 3] = t_new
        if squeezed:
            out = out.squeeze(1)
        return out

    @staticmethod
    def _pick_best_per_sample(
        stacks: Dict[str, torch.Tensor],
        scores: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Per-sample argmin gather across multiple stacked candidate fields.

        Used by ``_fit_pointmap_alignment_keep_k``: enumerate M candidate
        fits (k-view subsets), score each at [M, N], then pick the best
        per sample and gather its fields.

        Args:
            stacks: dict of [M, N, ...] tensors keyed by field name.
            scores: [M, N] (NaN/inf treated as worse-than-everything).

        Returns:
            Dict mapping each field to its [N, ...] gathered tensor;
            adds ``'best': LongTensor[N]`` with the per-sample winner.
        """
        best = scores.argmin(dim=0)                                # [N]
        idx_n = torch.arange(scores.shape[1], device=scores.device)
        out = {k: v[best, idx_n] for k, v in stacks.items()}
        out['best'] = best
        return out

    def _sample_3d_pointclouds(
        self,
        preds_mesh,
        indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample [N, num_3d_samples, 3] point clouds from pred surfaces (in
        pred frame) and GT meshes on disk (in GT frame). Strategy is selected
        by ``self.mesh_sample_method`` and applied symmetrically to both sides.
        """
        n = self.num_3d_samples
        method = self.mesh_sample_method
        render_kwargs = (
            {"n_views": self.n_3d_render_views,
             "resolution": self.n_3d_render_resolution}
            if method == "render" else {}
        )

        pred_list = [sample_points_from_mesh_extract(m, n=n, method=method, **render_kwargs) for m in preds_mesh]
        pred_pts = torch.stack(pred_list, dim=0)
        device = pred_pts.device
        dtype = pred_pts.dtype

        instance_indices, _ = self.get_instance_view_indices(indices.cpu().numpy())
        rows = self.instances[instance_indices]
        gt_list = []
        for row in rows:
            root, sha256 = str(row[0]), str(row[1])
            path = self.gt_mesh_path_template.format(root=root, sha256=sha256)
            pts = sample_points_from_ply_path(path, n=n, method=method, device=device, **render_kwargs)
            gt_list.append(pts.to(dtype=dtype))
        gt_pts = torch.stack(gt_list, dim=0)
        return pred_pts, gt_pts

    # ----------------------------------------------------------------------
    # §2 — C1 input-pointmap LSQ (§5.1)
    # ----------------------------------------------------------------------

    def _fit_pointmap_alignment(
        self,
        *,
        extrinsics_pred: torch.Tensor,                    # [N, 4, 4] or [N, V, 4, 4]
        intrinsics_pred: torch.Tensor,                    # [N, 3, 3] or [N, V, 3, 3]
        data: dict,
        preds: Optional[dict] = None,
        cond_prefix: str = '',
    ) -> dict:
        """C1 input-view pointmap LSQ (§5.1).

        Builds per-pixel pred/GT correspondences via
        ``_compute_pointmap_correspondences`` then runs Umeyama on
        ``pred_uv ↔ gt_uv``. With ``1 ≤ pointmap_keep_count < V`` set,
        runs the best-of-``C(V, k)`` keep-k variant instead. Both produce
        the row-vector similarity ``gt = s * pred @ R + T``. Samples
        without valid correspondences use the fitter's identity fallback.
        """
        assert preds is not None and 'mesh' in preds, \
            "_fit_pointmap_alignment requires preds['mesh']"
        pred_points, gt_points, masks = self._compute_pointmap_correspondences(
            extrinsics_pred=extrinsics_pred,
            intrinsics_pred=intrinsics_pred,
            data=data,
            preds=preds,
            cond_prefix=cond_prefix,
        )
        V = pred_points.shape[1]

        keep_k = self.pointmap_keep_count
        if keep_k is not None and 1 <= int(keep_k) < V:
            result = self._fit_pointmap_alignment_keep_k(
                pred_points=pred_points, gt_points=gt_points,
                masks=masks, keep_count=int(keep_k),
            )
        else:
            result = fit_similarity_lsq_pointmaps(pred_points, gt_points, masks)

        return result

    def _compute_pointmap_correspondences(
        self,
        *,
        extrinsics_pred: torch.Tensor,                    # [N, 4, 4] or [N, V, 4, 4]
        intrinsics_pred: torch.Tensor,                    # [N, 3, 3] or [N, V, 3, 3]
        data: dict,
        preds: dict,
        cond_prefix: str = '',
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render pred depth+mask, back-project pred and GT pointmaps, and
        build the joint per-pixel mask used by the C1 LSQ / ICP fits.

        Filters applied to ``masks`` in order:
          1. ``pred_render_mask & (gt_depth > 0)`` — both sides see geometry.
          2. Runtime min-coverage gate (``self.min_view_mask_frac``) — drop
             views whose joint mask covers <1% of pixels by default.

        Match the GT keys to whatever the evaluator's rendering path uses so
        the LSQ-fitted similarity is valid for that render-K. With
        ``cond_prefix='raw_'`` (set when ``metrics_on_raw_cond=true``) we
        read the pre-crop depth/intrinsic; otherwise the post-crop ones.
        ``extrinsics`` are unaffected by 2D cropping so they have no prefix.

        Single-view inputs ([N, 4, 4] / [N, H, W]) are promoted to V=1
        internally so the same code path serves single-view and multi-view evaluation.

        Returns:
            pred_points: [N, V, P, 3]; gt_points: [N, V, P, 3]
            masks:       [N, V, P] bool
        """
        assert 'mesh' in preds, \
            "_compute_pointmap_correspondences requires preds['mesh']"
        # Normalize to [N, V, ...] regardless of caller.
        if extrinsics_pred.ndim == 3:
            extrinsics_pred = extrinsics_pred.unsqueeze(1)
            intrinsics_pred = intrinsics_pred.unsqueeze(1)
        gt_depths = data[f'{cond_prefix}cond_depth']
        gt_extrinsics = data['extrinsics']
        gt_intrinsics = data[f'{cond_prefix}intrinsics']
        if gt_depths.ndim == 3:                           # [N, H, W] → [N, 1, H, W]
            gt_depths = gt_depths.unsqueeze(1)
        if gt_extrinsics.ndim == 3:                       # [N, 4, 4] → [N, 1, 4, 4]
            gt_extrinsics = gt_extrinsics.unsqueeze(1)
        if gt_intrinsics.ndim == 3:                       # [N, 3, 3] → [N, 1, 3, 3]
            gt_intrinsics = gt_intrinsics.unsqueeze(1)
        N, V, H, W = gt_depths.shape

        pred_depth, pred_mask = self._render_pred_depth_mask(
            preds, extrinsics_pred, intrinsics_pred,
        )
        if pred_depth.shape[-2:] != (H, W):
            pred_depth = F.interpolate(
                pred_depth.reshape(N * V, 1, *pred_depth.shape[-2:]),
                size=(H, W), mode='nearest').reshape(N, V, H, W)
            pred_mask = F.interpolate(
                pred_mask.float().reshape(N * V, 1, *pred_mask.shape[-2:]),
                size=(H, W), mode='nearest').reshape(N, V, H, W) > 0.5

        grid_x = (torch.arange(W, dtype=pred_depth.dtype, device=pred_depth.device) + 0.5) / W
        grid_y = (torch.arange(H, dtype=pred_depth.dtype, device=pred_depth.device) + 0.5) / H
        uv_grid = torch.stack(torch.meshgrid(grid_x, grid_y, indexing='xy'), dim=-1)  # [H, W, 2]
        uv_flat = uv_grid.reshape(H * W, 2).expand(N, V, H * W, 2).contiguous()

        pred_points = unproject_cv(
            uv_flat, pred_depth.reshape(N, V, H * W),
            intrinsics_pred, extrinsics_pred)
        gt_points = unproject_cv(
            uv_flat, gt_depths.reshape(N, V, H * W),
            gt_intrinsics, gt_extrinsics)

        # Filter 1: both sides see geometry.
        masks = pred_mask.reshape(N, V, H * W) & (gt_depths.reshape(N, V, H * W) > 0)
        # Filter 2: reject views with too few visible correspondences for
        # a stable fit, using the rendered footprint of each prediction.
        min_view_mask_frac = float(getattr(self, 'min_view_mask_frac', 0.01))
        if min_view_mask_frac > 0:
            min_px = max(1, int(min_view_mask_frac * H * W))
            view_keep = masks.sum(dim=-1) >= min_px        # [N, V]
            masks = masks & view_keep.unsqueeze(-1)

        return pred_points, gt_points, masks

    def _fit_pointmap_alignment_keep_k(
        self,
        *,
        pred_points: torch.Tensor,                         # [N, V, P, 3]
        gt_points: torch.Tensor,                           # [N, V, P, 3]
        masks: torch.Tensor,                               # [N, V, P] bool
        keep_count: int,
    ) -> dict:
        """Per-sample best-of-``C(V, keep_count)`` pointmap LSQ.

        Tries every subset of ``keep_count`` views, fits the LSQ on
        those views' pointmaps, scores each fit by **mean residual on
        the held-out (dropped) views** (transform held-out pred-pts
        via the fitted similarity, measure ℓ₂ to held-out gt-pts on
        masked pixels). Picks the subset with the lowest held-out
        residual per sample.

        Returns the same keys as ``fit_similarity_lsq_pointmaps`` plus
        ``kept_mask: BoolTensor[N, V]`` flagging the chosen subset per
        sample.
        """
        from itertools import combinations
        N, V = pred_points.shape[:2]
        device = pred_points.device
        dtype = pred_points.dtype
        assert 1 <= keep_count <= V, \
            f"keep_count must be in [1, V={V}], got {keep_count}"

        subsets = list(combinations(range(V), keep_count))
        M = len(subsets)

        # Precompute the [M, V] subset-membership mask once so the chosen-drop
        # → kept-mask gather at the end is a single index op, not a Python loop.
        subsets_mask = torch.zeros(M, V, dtype=torch.bool, device=device)
        for i, keep in enumerate(subsets):
            subsets_mask[i, torch.tensor(keep, device=device, dtype=torch.long)] = True

        s_M = torch.zeros(M, N, device=device, dtype=dtype)
        R_M = torch.zeros(M, N, 3, 3, device=device, dtype=dtype)
        T_M = torch.zeros(M, N, 3, device=device, dtype=dtype)
        valid_M = torch.zeros(M, N, dtype=torch.bool, device=device)
        score_M = torch.full((M, N), float('inf'), device=device, dtype=dtype)

        for i, keep in enumerate(subsets):
            held = tuple(v for v in range(V) if v not in keep)
            keep_t = torch.tensor(keep, device=device, dtype=torch.long)
            pp_k = pred_points.index_select(1, keep_t)
            gp_k = gt_points.index_select(1, keep_t)
            m_k = masks.index_select(1, keep_t)
            fit = fit_similarity_lsq_pointmaps(pp_k, gp_k, m_k)
            s_M[i] = fit['s']
            R_M[i] = fit['R_pa']
            T_M[i] = fit['T_pa']
            valid_M[i] = fit['valid'].to(torch.bool)

            if held:
                held_t = torch.tensor(held, device=device, dtype=torch.long)
                pp_h = pred_points.index_select(1, held_t)
                gp_h = gt_points.index_select(1, held_t)
                m_h = masks.index_select(1, held_t).to(dtype)
                # Apply forward similarity: gt' = s * pred @ R_pa + T_pa.
                tp = (fit['s'][:, None, None, None] *
                      torch.einsum('nhpi,nij->nhpj', pp_h, fit['R_pa']) +
                      fit['T_pa'][:, None, None, :])
                err = (tp - gp_h).norm(dim=-1)             # [N, H, P]
                denom = m_h.sum(dim=(1, 2)).clamp_min(1.0)
                mean_err = (err * m_h).sum(dim=(1, 2)) / denom
                # Samples with no held-out coverage fall back to the
                # fit's own in-sample residual.
                no_held = m_h.sum(dim=(1, 2)) < 1.0
                score_M[i] = torch.where(
                    no_held, fit['residual'].to(dtype), mean_err)
            else:
                score_M[i] = fit['residual'].to(dtype)
            # Invalid LSQ fits get score=+inf so they lose argmin.
            score_M[i] = torch.where(
                valid_M[i], score_M[i],
                torch.full_like(score_M[i], float('inf')))

        # Per-sample argmin over subsets, gathering all winning fields.
        # ``residual`` is the picked subset's score (held-out err or LSQ
        # residual fallback) — same semantics as the original ``res_best``.
        picked = self._pick_best_per_sample(
            {'s': s_M, 'R_pa': R_M, 'T_pa': T_M, 'valid': valid_M, 'residual': score_M},
            score_M,
        )
        best = picked.pop('best')
        return {**picked, 'kept_mask': subsets_mask[best]}                  # [N, V]

    # ----------------------------------------------------------------------
    # §3 — C2 full-pointcloud ICP (§5.2)
    # ----------------------------------------------------------------------

    def _fit_pointcloud_alignment(self) -> dict:
        """C2 full-pointcloud similarity ICP (§5.2). Reads pre-sampled
        (pred, GT) point clouds from ``self._stage2_pcd`` (populated by
        ``evaluation_metrics`` before the alignment fit) and runs
        ``fit_similarity_icp``.
        """
        assert getattr(self, '_stage2_pcd', None) is not None, \
            "_fit_pointcloud_alignment requires evaluation_metrics to pre-sample point clouds"
        pred_pts, gt_pts = self._stage2_pcd
        return fit_similarity_icp(
            pred_pts, gt_pts,
            n_iters=self.icp_n_iters,
            n_inits=self.icp_n_inits,
            init_mode=getattr(self, 'icp_init_mode', 'z_axis'),
            estimate_scale=getattr(self, 'icp_estimate_scale', False),
        )

    # ----------------------------------------------------------------------
    # §4 — C3 trajectory similarity (§5.3)
    # ----------------------------------------------------------------------

    def _fit_trajectory_alignment(
        self,
        extrinsics_pred: torch.Tensor,
        extrinsics_gt: torch.Tensor,
    ) -> Optional[dict]:
        """C3 trajectory similarity (camera centers + look tips), §5.3.

        Requires ``[N, V, 4, 4]`` with V≥3 to be well-posed; returns ``None``
        for single-view (V=1 input shape ``[N, 4, 4]``) — caller falls through
        to no transform.
        """
        if extrinsics_pred.ndim == 3:
            return None
        return align_extrinsics_by_similarity(extrinsics_pred, extrinsics_gt)
