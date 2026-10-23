import itertools
from typing import Tuple

import torch
from torch import Tensor
from . import geometry_utils, voxel_utils


def fit_similarity_transform(
    src: Tensor,
    dst: Tensor,
    weights: Tensor | None = None,
    fix_scale: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Batched similarity transform fit (row-vector / pytorch3d convention).

    Args:
        src: [N, P, 3] source point sets.
        dst: [N, P, 3] target point sets.
        weights: [N, P] optional per-point non-negative weights. If None,
            all points are weighted equally. Internally normalized so each
            instance's weights sum to 1.

    Returns:
        s:        [N]       uniform scale.
        R_pa:     [N, 3, 3] rotation in row-vector form.
        T_pa:     [N, 3]    translation.
        residual: [N]       weighted mean per-point L2 residual after the fit.
        valid:    [N] bool  False if the fit is rank-deficient or the
                            source cloud is collapsed.

    Convention: ``dst ≈ s[:, None, None] * (src @ R_pa) + T_pa[:, None, :]``.

    Requires P >= 3 AND rank(X) >= 2 (planar camera rigs are fine; only
    colinear / single-point rigs are rejected via the rank check below).
    Inputs are expected to be float32 (upstream DLT and dataset extrinsics
    are already float32; ``torch.linalg.svd`` is not on the autocast
    allowlist so float32 inputs stay float32 even under AMP).
    """
    from pytorch3d.ops import corresponding_points_alignment

    assert src.shape == dst.shape and src.ndim == 3 and src.shape[-1] == 3, \
        f"src/dst must be [N, P, 3], got {tuple(src.shape)} vs {tuple(dst.shape)}"
    assert src.shape[1] >= 3, f"need P>=3, got P={src.shape[1]}"
    assert src.device == dst.device, \
        f"src/dst must be on the same device, got {src.device} vs {dst.device}"

    N, P = src.shape[:2]

    # Normalize weights so each instance sums to 1; default = uniform.
    if weights is None:
        w = src.new_full((N, P), 1.0 / P)            # [N, P]
    else:
        assert weights.shape == (N, P), \
            f"weights must be [N, P]={(N, P)}, got {tuple(weights.shape)}"
        w = weights.clamp_min(0)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)
    w3 = w.unsqueeze(-1)                              # [N, P, 1]

    # 1. Validity check: weighted source rank. Accept planar rigs (rank>=2),
    # reject collapsed / colinear ones. Computed on weighted, centered X so
    # the singular values directly reflect the rank of the source cloud.
    mu_src = (w3 * src).sum(dim=1, keepdim=True)     # [N, 1, 3]
    X = src - mu_src                                  # [N, P, 3]
    var_src = (w * (X * X).sum(dim=-1)).sum(dim=1)   # [N]
    D_x = torch.linalg.svdvals(w3.sqrt() * X)        # [N, 3], descending
    valid = (var_src > 1e-12) & (D_x[:, 1] > D_x[:, 0] * 1e-6) & (D_x[:, 1] > 1e-9)

    # 2. Umeyama via pytorch3d (row-vector form: dst ≈ s * (src @ R) + T).
    sim = corresponding_points_alignment(
        src, dst, weights=w,
        estimate_scale=not fix_scale, allow_reflection=False,
    )
    R_pa, T_pa, s = sim.R, sim.T, sim.s              # [N,3,3], [N,3], [N]
    if fix_scale:
        # cpa already returns s=1 when estimate_scale=False, but make the
        # contract explicit and dtype-safe.
        s = torch.ones(N, device=src.device, dtype=src.dtype)

    # 3. Weighted residual.
    aligned = s[:, None, None] * (src @ R_pa) + T_pa[:, None, :]
    per_point = torch.norm(aligned - dst, dim=-1)    # [N, P]
    residual = (w * per_point).sum(dim=1)            # [N]

    # 4. Substitute identity / +inf residual for invalid instances.
    inv = ~valid
    if inv.any():
        eye = torch.eye(3, device=src.device, dtype=src.dtype)
        R_pa = torch.where(inv[:, None, None], eye.expand_as(R_pa), R_pa)
        T_pa = torch.where(inv[:, None], torch.zeros_like(T_pa), T_pa)
        s = torch.where(inv, torch.ones_like(s), s)
        residual = torch.where(
            inv, torch.full_like(residual, float("inf")), residual)

    return s, R_pa, T_pa, residual, valid


def apply_similarity_to_extrinsics(
    R_pred: Tensor,   # [N, V, 3, 3]
    t_pred: Tensor,   # [N, V, 3]
    s: Tensor,        # [N]
    R_pa: Tensor,     # [N, 3, 3]   (row-vector convention, as returned by fit)
    T_pa: Tensor,     # [N, 3]
) -> tuple[Tensor, Tensor]:
    """
    Bridge the row-vector similarity (from ``fit_similarity_transform``)
    to a column-vector extrinsic update.

    Given ``x_gt_world ≈ s * (x_pred_world @ R_pa) + T_pa`` (row-vector form),
    the equivalent column-vector form is
    ``x_gt_world = s * R_pa^T @ x_pred_world + T_pa``. Substituting into the
    pred extrinsic ``x_cam = R_pred @ x_pred_world + t_pred`` and multiplying
    through by ``s`` yields:

        R_new = R_pred @ R_pa
        t_new = s * t_pred - R_new @ T_pa

    The aligned extrinsic preserves image projection up to a uniform camera-
    coordinate scale, which cancels in perspective division.

    Returns:
        R_new: [N, V, 3, 3]
        t_new: [N, V, 3]
    """
    assert R_pred.shape[-2:] == (3, 3) and R_pred.ndim == 4
    assert t_pred.shape[-1] == 3 and t_pred.ndim == 3
    assert s.shape == (R_pred.shape[0],)
    assert R_pa.shape == (R_pred.shape[0], 3, 3)
    assert T_pa.shape == (R_pred.shape[0], 3)
    assert R_pred.device == t_pred.device == s.device == R_pa.device == T_pa.device, \
        "all inputs must be on the same device"

    # R_new[n,v] = R_pred[n,v] @ R_pa[n]
    R_new = torch.einsum("nvij,njk->nvik", R_pred, R_pa)
    # t_new[n,v] = s[n] * t_pred[n,v] - R_new[n,v] @ T_pa[n]
    t_new = s[:, None, None] * t_pred - torch.einsum("nvij,nj->nvi", R_new, T_pa)
    return R_new, t_new


def align_extrinsics_by_similarity(
    extrinsics_pred: Tensor,     # [N, V, 3, 4] or [N, V, 4, 4]
    extrinsics_gt: Tensor,       # same shape as extrinsics_pred
    look_tip_distance: float = 1.0,
    weights: Tensor | None = None,   # [N, V], per-view confidence weights
) -> dict:
    """
    High-level camera-rig alignment helper. Wraps the full pipeline of:

    1. Extract camera centers (``C = -R^T @ t``) and world-space look
       directions (third row of R, i.e. the +z_cam axis under OpenCV
       convention) from the pred and GT extrinsics.
    2. Build an augmented point set per instance: each view contributes
       its camera center AND a look-direction tip at
       ``C + look_tip_distance * R[2, :]``. This doubles the Umeyama
       correspondence count from V to 2V, which tightens the fit against
       per-view DLT noise and provides explicit rotational constraint that
       camera centers alone cannot supply.
    3. Fit a similarity transform pred-world → gt-world via
       ``fit_similarity_transform``.
    4. Apply the similarity to every pred extrinsic via
       ``apply_similarity_to_extrinsics``.

    Accepts extrinsics in either ``(N, V, 3, 4)`` or ``(N, V, 4, 4)``
    homogeneous form. Returns the aligned extrinsic in the same shape as
    the input.

    **Assumption**: the look-direction augmentation uses a look tip at a
    fixed WORLD distance ``look_tip_distance`` from the camera center. This
    is only invariant under similarity transforms with scale ``s ≈ 1``; for
    larger deviations, the tip in the gt rig would need to be at
    ``s * look_tip_distance`` to be consistent, which we don't know upfront.
    In practice the multipose evaluator sees canonical drifts with
    ``s`` close to 1 (empirically 0.85–1.05), and the resulting bias on the
    fit is negligible compared to the reduction in per-view noise variance.
    Callers using this helper for applications with strongly non-unit scale
    should expect a small systematic bias in the fitted scale.

    Args:
        extrinsics_pred: [N, V, 3, 4] or [N, V, 4, 4], column-form extrinsics
            mapping pred-world → camera.
        extrinsics_gt: same shape, mapping gt-world → camera.
        look_tip_distance: distance along each camera's forward axis to place
            the augmentation tip. Default 1.0 (canonical units). For cameras
            orbiting at radius ~2 looking at origin, this lands tips at roughly
            half the orbit radius — distinct from the camera ring and not
            collapsed to the origin.
        weights: [N, V] optional per-view confidence weights (e.g. inverse DLT
            reprojection error). Replicated across centers and look tips so
            both points per view share the same confidence.

    Returns:
        dict with:
            extrinsics_pred_aligned: same shape as ``extrinsics_pred``. Aligned
                pred extrinsics that, applied to gt-world points, reproduce the
                original pred image projections up to a uniform camera-space
                scale (which cancels in perspective division).
            s:        [N]       fitted uniform scale.
            R_pa:     [N, 3, 3] fitted rotation (row-vector convention).
            T_pa:     [N, 3]    fitted translation (pred-world origin drift).
            residual: [N]       mean per-point L2 residual after the fit.
            valid:    [N] bool  False if the fit is rank-deficient (e.g.
                collapsed rig) — in that case the aligned extrinsics are
                produced with an identity similarity (i.e. equal to pred).
    """
    assert extrinsics_pred.shape == extrinsics_gt.shape, \
        f"shape mismatch: {tuple(extrinsics_pred.shape)} vs {tuple(extrinsics_gt.shape)}"
    assert extrinsics_pred.ndim == 4, \
        f"expected 4-D [N, V, *, *], got ndim={extrinsics_pred.ndim}"
    assert extrinsics_pred.shape[-2:] in ((3, 4), (4, 4)), \
        f"expected last-2 dims (3,4) or (4,4), got {tuple(extrinsics_pred.shape[-2:])}"

    input_rows = extrinsics_pred.shape[-2]

    # Normalize to (3, 4) for internal math.
    ext_pred = extrinsics_pred[..., :3, :]
    ext_gt = extrinsics_gt[..., :3, :]
    R_pred = ext_pred[..., :3, :3]                                    # [N, V, 3, 3]
    t_pred = ext_pred[..., :3, 3]                                     # [N, V, 3]
    R_gt = ext_gt[..., :3, :3]
    t_gt = ext_gt[..., :3, 3]

    # 1. Camera centers in world coords: C = -R^T @ t.
    C_pred = -torch.einsum("nvji,nvj->nvi", R_pred, t_pred)           # [N, V, 3]
    C_gt = -torch.einsum("nvji,nvj->nvi", R_gt, t_gt)

    # 2. World-space look directions (third row of R = +z_cam in world).
    look_pred = R_pred[..., 2, :]                                      # [N, V, 3]
    look_gt = R_gt[..., 2, :]

    # 2b. Augmented 2V-point set: center + look-direction tip per view.
    L_pred = C_pred + look_tip_distance * look_pred                    # [N, V, 3]
    L_gt = C_gt + look_tip_distance * look_gt
    P_pred = torch.cat([C_pred, L_pred], dim=1)                        # [N, 2V, 3]
    P_gt = torch.cat([C_gt, L_gt], dim=1)

    # Replicate per-view weights across both points per view.
    if weights is not None:
        assert weights.shape == C_pred.shape[:2], \
            f"weights must be [N, V]={tuple(C_pred.shape[:2])}, got {tuple(weights.shape)}"
        weights = torch.cat([weights, weights], dim=1)                 # [N, 2V]

    # 3. Fit similarity.
    s, R_pa, T_pa, residual, valid = fit_similarity_transform(
        P_pred, P_gt, weights=weights)

    # 4. Apply to pred extrinsics.
    R_aln, t_aln = apply_similarity_to_extrinsics(
        R_pred=R_pred, t_pred=t_pred, s=s, R_pa=R_pa, T_pa=T_pa)
    ext_aln = torch.cat([R_aln, t_aln.unsqueeze(-1)], dim=-1)          # [N, V, 3, 4]

    # Restore input row-count if it was (4, 4).
    if input_rows == 4:
        pad_row = ext_aln.new_zeros(*ext_aln.shape[:-2], 1, 4)
        pad_row[..., 0, 3] = 1.0
        ext_aln = torch.cat([ext_aln, pad_row], dim=-2)                # [N, V, 4, 4]

    return {
        "extrinsics_pred_aligned": ext_aln,
        "s": s,
        "R_pa": R_pa,
        "T_pa": T_pa,
        "residual": residual,
        "valid": valid,
    }


def fit_similarity_lsq_pointmaps(
    pred_points: Tensor,             # [N, V, P, 3]
    gt_points: Tensor,               # [N, V, P, 3]
    masks: Tensor,                   # [N, V, P] bool — joint pred ∧ GT foreground
    max_pts_per_view: int = 50_000,
    generator: torch.Generator | None = None,
) -> dict:
    """
    Per-instance similarity fit from per-view pointmap correspondences
    (evaluation protocol §5.1, "input-view pointmap LSQ").

    For each instance, stacks foreground-masked (pred, GT) point pairs
    across V input views and runs Umeyama via ``fit_similarity_transform``.
    Each view contributes equal aggregate weight regardless of its valid
    point count (per-point weight = 1 / pts-in-view). Subsamples per-view
    to ``max_pts_per_view`` to bound memory.

    Args:
        pred_points: [N, V, P, 3] predicted per-view world-space points
            (e.g. unprojected predicted depth at predicted (E, K)).
        gt_points: [N, V, P, 3] GT per-view world-space points (unprojected
            GT depth at GT (E, K)).
        masks: [N, V, P] joint foreground mask, ``True`` where the
            correspondence should contribute to the fit.
        max_pts_per_view: random per-view subsample cap. Default 50k.
        generator: optional ``torch.Generator`` for reproducible subsampling.

    Returns:
        dict with ``s, R_pa, T_pa, residual, valid``. Same convention as
        ``align_extrinsics_by_similarity`` (row-vector form). Instances with
        no valid correspondences come back with ``valid=False`` and an
        identity similarity (handled by ``fit_similarity_transform``).
    """
    assert pred_points.shape == gt_points.shape and pred_points.ndim == 4 \
        and pred_points.shape[-1] == 3, \
        f"pred/gt must be [N, V, P, 3], got {tuple(pred_points.shape)}"
    assert masks.shape == pred_points.shape[:3], \
        f"masks must be [N, V, P]={tuple(pred_points.shape[:3])}, got {tuple(masks.shape)}"
    N, V, P = masks.shape
    device = pred_points.device
    dtype = pred_points.dtype

    src_list: list[Tensor] = []
    dst_list: list[Tensor] = []
    w_list: list[Tensor] = []
    max_active = 3                                   # fit_similarity_transform requires P>=3

    for n in range(N):
        view_src, view_dst, view_w = [], [], []
        for v in range(V):
            idx = masks[n, v].nonzero(as_tuple=False).squeeze(-1)
            num = int(idx.numel())
            if num == 0:
                continue
            if num > max_pts_per_view:
                perm = torch.randperm(num, device=device, generator=generator)[:max_pts_per_view]
                idx = idx[perm]
                num = max_pts_per_view
            view_src.append(pred_points[n, v][idx])
            view_dst.append(gt_points[n, v][idx])
            # Per-point weight = 1/num so each view contributes equal aggregate weight;
            # global per-instance normalization to sum=1 happens inside Umeyama.
            view_w.append(torch.full((num,), 1.0 / num, device=device, dtype=dtype))

        if view_w:
            src_n = torch.cat(view_src, dim=0)
            dst_n = torch.cat(view_dst, dim=0)
            w_n = torch.cat(view_w, dim=0)
        else:
            # No valid correspondences — produce a 3-point dummy so the batched
            # call has uniform shape; zero weights ensure Umeyama flags it invalid.
            src_n = torch.zeros((3, 3), device=device, dtype=dtype)
            dst_n = torch.zeros((3, 3), device=device, dtype=dtype)
            w_n = torch.zeros((3,), device=device, dtype=dtype)
        src_list.append(src_n)
        dst_list.append(dst_n)
        w_list.append(w_n)
        max_active = max(max_active, src_n.shape[0])

    # Pad to common P with zero-weight points so we can batch.
    src_pad = torch.zeros((N, max_active, 3), device=device, dtype=dtype)
    dst_pad = torch.zeros((N, max_active, 3), device=device, dtype=dtype)
    w_pad = torch.zeros((N, max_active), device=device, dtype=dtype)
    for n in range(N):
        p = src_list[n].shape[0]
        src_pad[n, :p] = src_list[n]
        dst_pad[n, :p] = dst_list[n]
        w_pad[n, :p] = w_list[n]

    s, R_pa, T_pa, residual, valid = fit_similarity_transform(src_pad, dst_pad, weights=w_pad)
    return {"s": s, "R_pa": R_pa, "T_pa": T_pa, "residual": residual, "valid": valid}


def transform_points_by_similarity(
    points: Tensor,                  # [N, P, 3]
    s: Tensor,                       # [N]
    R_pa: Tensor,                    # [N, 3, 3]
    T_pa: Tensor,                    # [N, 3]
) -> Tensor:
    """Apply the row-vector similarity ``dst = s * src @ R_pa + T_pa`` from
    ``fit_similarity_transform`` to a batched point cloud. Used to bring
    pred-frame points into GT frame for cross-frame metrics (e.g. 3D-CD)."""
    assert points.ndim == 3 and points.shape[-1] == 3, \
        f"points must be [N, P, 3], got {tuple(points.shape)}"
    return s[:, None, None] * (points @ R_pa) + T_pa[:, None, :]


def _generate_rotational_orientations() -> Tensor:
    """Enumerate the 24 det=+1 sign×permutation matrices on (x, y, z),
    i.e. the rotational symmetries of an oriented box. Sorted by
    angle-from-identity (descending trace), so identity is first and
    progressively-different rotations follow.

    Returns a tensor of shape [24, 3, 3].
    """
    Rs = []
    for perm in itertools.permutations((0, 1, 2)):
        # Row-form permutation: e_i @ P = e_{perm[i]}.
        P = torch.zeros(3, 3)
        for i in range(3):
            P[i, perm[i]] = 1.0
        for signs in itertools.product((1.0, -1.0), repeat=3):
            S = torch.diag(torch.tensor(signs))
            R = S @ P
            if torch.linalg.det(R).item() > 0:
                Rs.append(R)
    Rs.sort(key=lambda R: -float(R.trace()))    # identity (trace=3) first
    return torch.stack(Rs, dim=0)               # [24, 3, 3]


_ALL_ICP_ORIENTATIONS = _generate_rotational_orientations()
_ICP_INIT_PRESETS = {1: 1, 4: 4, 8: 8, 24: 24}     # n_inits → slice length (pca_bbox mode)
_ICP_Z_AXIS_PRESETS = {1, 2, 4}                    # n_inits options for z_axis mode


def _make_z_axis_rotations(K: int, device, dtype) -> Tensor:
    """K rotation matrices about world z, evenly spaced from 0 to 360°.

    Returns a [K, 3, 3] tensor in row-vector convention (``x_row @ R``).
    Matches the canonical-frame init pattern used in the prior eval pipeline:
    objects are stored z-up in ``[-0.5, 0.5]³``, so the only inter-frame
    ambiguity is azimuthal."""
    if K not in _ICP_Z_AXIS_PRESETS:
        raise ValueError(f"z_axis n_inits must be in {sorted(_ICP_Z_AXIS_PRESETS)}, got {K}")
    angles = torch.linspace(0.0, 2 * torch.pi, K + 1, device=device, dtype=dtype)[:-1]
    Rs = []
    for a in angles:
        c, s = torch.cos(a), torch.sin(a)
        # Row-form rotation about +z: x_row @ R where R = [[c, s, 0], [-s, c, 0], [0, 0, 1]]
        # is equivalent to standard column-form Rz(a). Sign convention picked to match
        # ``fit_similarity_transform``'s row-vector dst = src @ R + T.
        R = torch.zeros(3, 3, device=device, dtype=dtype)
        R[0, 0] = c;  R[0, 1] = s
        R[1, 0] = -s; R[1, 1] = c
        R[2, 2] = 1.0
        Rs.append(R)
    return torch.stack(Rs, dim=0)


def _pca_frame(points: Tensor) -> Tuple[Tensor, Tensor]:
    """Centroid + principal axes (columns, descending eigenvalue) of a
    point cloud. Forces det(P)=+1 so the frame is a proper rotation."""
    mu = points.mean(dim=0)                         # [3]
    centered = points - mu
    cov = (centered.transpose(-1, -2) @ centered) / max(points.shape[0] - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)       # ascending
    P = eigvecs[:, eigvals.argsort(descending=True)]
    if torch.linalg.det(P).item() < 0:
        P = P.clone()
        P[:, -1] = -P[:, -1]
    return mu, P


def fit_similarity_icp(
    src: Tensor,                     # [N, P_src, 3]
    dst: Tensor,                     # [N, P_dst, 3]
    *,
    n_iters: int = 50,
    n_inits: int = 4,
    init_mode: str = "z_axis",
    estimate_scale: bool = False,
    convergence_eps: float = 1e-6,
) -> dict:
    """Per-instance similarity ICP with multi-init (protocol §5.2).

    Two init strategies:

    - ``init_mode="z_axis"`` (default): enumerate ``n_inits ∈ {1, 2, 4}``
      rotations evenly spaced about world +z, no PCA. Appropriate for
      canonical-frame data (objects stored z-up in ``[-0.5, 0.5]³``) where
      the only inter-frame ambiguity is azimuthal. Mirrors the working
      eval pipeline pattern.
    - ``init_mode="pca_bbox"``: enumerate ``n_inits ∈ {1, 4, 8, 24}``
      rotations from the PCA-frame box-orbit. Appropriate for cross-frame
      data where the relative orientation is unknown; can fail on
      near-isotropic clouds because PCA noise misaligns the orbit.

    ``estimate_scale=False`` (default) pins ``s = 1`` throughout, which is
    correct for both clouds living in the same canonical metric scale.
    Setting it True restores Umeyama's scale estimation (needed for
    cross-frame baselines whose 3D output is in an unknown metric).

    Args:
        src: [N, P_src, 3] source point cloud.
        dst: [N, P_dst, 3] target point cloud.
        n_iters: max ICP iterations per init.
        n_inits: number of inits per instance; valid set depends on init_mode.
        init_mode: ``"z_axis"`` or ``"pca_bbox"``.
        estimate_scale: if False (default), pin scale to 1; if True, estimate.
        convergence_eps: residual-decrease threshold for early termination.

    Returns:
        Dict ``{s, R_pa, T_pa, residual, valid}`` (row-vector convention).
    """
    from pytorch3d.ops import iterative_closest_point, knn_points
    from pytorch3d.ops.points_alignment import SimilarityTransform

    assert src.ndim == 3 and dst.ndim == 3 and src.shape[-1] == 3 and dst.shape[-1] == 3
    assert src.shape[0] == dst.shape[0]
    N, P_src = src.shape[:2]
    P_dst = dst.shape[1]
    device, dtype = src.device, src.dtype

    # Resolve K and per-(n, k) init rotations / scales.
    if init_mode == "z_axis":
        if n_inits not in _ICP_Z_AXIS_PRESETS:
            raise ValueError(
                f"z_axis n_inits must be in {sorted(_ICP_Z_AXIS_PRESETS)}, got {n_inits}")
        K = n_inits
        Rs = _make_z_axis_rotations(K, device, dtype)                           # [K, 3, 3]
        R_init = Rs.unsqueeze(0).expand(N, K, 3, 3).contiguous()                # [N, K, 3, 3]
        s_init = torch.ones(N, K, device=device, dtype=dtype)
    elif init_mode == "pca_bbox":
        if n_inits not in _ICP_INIT_PRESETS:
            raise ValueError(
                f"pca_bbox n_inits must be in {sorted(_ICP_INIT_PRESETS)}, got {n_inits}")
        K = _ICP_INIT_PRESETS[n_inits]
        init_Ms = _ALL_ICP_ORIENTATIONS[:K].to(device=device, dtype=dtype)      # [K, 3, 3]
        R_init = torch.empty(N, K, 3, 3, device=device, dtype=dtype)
        s_init = torch.ones(N, K, device=device, dtype=dtype)
        for n in range(N):
            _, P_src_n = _pca_frame(src[n])
            _, P_dst_n = _pca_frame(dst[n])
            for k in range(K):
                R_init[n, k] = P_src_n @ init_Ms[k] @ P_dst_n.transpose(-1, -2)
            if estimate_scale:
                spread_src = (src[n] - src[n].mean(0)).norm(dim=-1).mean().clamp_min(1e-6)
                spread_dst = (dst[n] - dst[n].mean(0)).norm(dim=-1).mean().clamp_min(1e-6)
                s_init[n, :] = (spread_dst / spread_src).to(dtype)
    else:
        raise ValueError(f"unknown init_mode={init_mode!r}; use 'z_axis' or 'pca_bbox'")

    # Per-(n, k) initial T = mu_dst - s * (mu_src @ R_init).
    mu_src = src.mean(dim=1)                                                    # [N, 3]
    mu_dst = dst.mean(dim=1)                                                    # [N, 3]
    mu_src_rot = torch.einsum("ni,nkij->nkj", mu_src, R_init)                   # [N, K, 3]
    T_init = mu_dst.unsqueeze(1) - s_init.unsqueeze(-1) * mu_src_rot            # [N, K, 3]

    # Flatten N*K and run a single batched ICP across all inits.
    NK = N * K
    src_b = src.unsqueeze(1).expand(N, K, P_src, 3).reshape(NK, P_src, 3)
    dst_b = dst.unsqueeze(1).expand(N, K, P_dst, 3).reshape(NK, P_dst, 3)
    init = SimilarityTransform(
        R=R_init.reshape(NK, 3, 3),
        T=T_init.reshape(NK, 3),
        s=s_init.reshape(NK),
    )
    sol = iterative_closest_point(
        src_b, dst_b,
        init_transform=init,
        max_iterations=n_iters,
        relative_rmse_thr=convergence_eps,
        estimate_scale=estimate_scale,
        allow_reflection=False,
    )

    # Pick best init per N by rmse (treat NaN/inf as worst).
    R_NK = sol.RTs.R.reshape(N, K, 3, 3)
    T_NK = sol.RTs.T.reshape(N, K, 3)
    s_NK = sol.RTs.s.reshape(N, K)
    rmse_NK = sol.rmse.reshape(N, K)
    rmse_NK = torch.where(torch.isfinite(rmse_NK), rmse_NK,
                          torch.full_like(rmse_NK, float("inf")))
    best_k = rmse_NK.argmin(dim=1)                                              # [N]
    n_idx = torch.arange(N, device=device)
    R_pa = R_NK[n_idx, best_k]
    T_pa = T_NK[n_idx, best_k]
    s = s_NK[n_idx, best_k]
    best_rmse = rmse_NK[n_idx, best_k]

    # Recompute residual as mean L2 NN-distance (matches fit_similarity_transform
    # semantic, not pytorch3d's rmse).
    src_aln = s[:, None, None] * (src @ R_pa) + T_pa[:, None, :]                # [N, P_src, 3]
    nn = knn_points(src_aln, dst, K=1, return_sorted=False)
    residual = nn.dists.squeeze(-1).clamp_min(0).sqrt().mean(dim=1)             # [N]

    valid = torch.isfinite(best_rmse)
    inv = ~valid
    if inv.any():
        eye = torch.eye(3, device=device, dtype=dtype)
        R_pa = torch.where(inv[:, None, None], eye.expand_as(R_pa), R_pa)
        T_pa = torch.where(inv[:, None], torch.zeros_like(T_pa), T_pa)
        s = torch.where(inv, torch.ones_like(s), s)
        residual = torch.where(inv, torch.full_like(residual, float("inf")), residual)

    return {"s": s, "R_pa": R_pa, "T_pa": T_pa, "residual": residual, "valid": valid}


def align_depthmap(intrinsic_sources, depth_sources, intrinsic_targets, depth_targets, source_masks, target_masks, recompute_intrinsic=False, use_moge_depth=False):
    if recompute_intrinsic:
        raise NotImplementedError('Intrinsic recomputation during depth alignment is not supported.')
    from pytorch3d.ops import corresponding_points_alignment

    RTs, scales = [], []
    all_points = []
    for intrinsic_source, depth_source, intrinsic_target, depth_target, source_mask, target_mask in zip(intrinsic_sources, depth_sources, intrinsic_targets, depth_targets, source_masks, target_masks):
        assert intrinsic_source.shape == (3,3) and intrinsic_target.shape == (3,3)
        assert depth_source.shape == depth_target.shape

        assert depth_source.shape[0] == source_mask.shape[0] and depth_source.shape[1] == source_mask.shape[1]
        assert depth_target.shape[0] == target_mask.shape[0] and depth_target.shape[1] == target_mask.shape[1]
        assert depth_source.shape == depth_target.shape 
        assert depth_source.shape[0] == depth_source.shape[1] # square depthmaps

        assert source_mask.sum() > 0, "Mask is empty!"
        assert target_mask.sum() > 0, "Mask is empty!"

        # Erode the mask to avoid edge artifacts
        import torch.nn.functional as F
        mask = source_mask & target_mask
        if mask.sum() == 0:
            raise ValueError("No valid correspondences found!")
        mask_float = mask.float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        kernel = torch.ones((1, 1, 5, 5), device=mask.device)
        eroded = F.conv2d(mask_float, kernel, padding=2) == 25  # 5x5 kernel, keep only full neighborhoods
        mask = eroded.squeeze(0).squeeze(0)


        max_edge = max(depth_source.shape)
        device = depth_source.device

        grid_x = (torch.arange(0, max_edge, dtype=torch.float32, device=device) + 0.5) / max_edge
        grid_y = (torch.arange(0, max_edge, dtype=torch.float32, device=device) + 0.5) / max_edge
        uvs = torch.stack(torch.meshgrid(grid_x, grid_y, indexing='xy'), dim=-1)

        pointmap_source = voxel_utils.unproject_cv(uvs, depth_source, intrinsic_source)[mask]
        if use_moge_depth:
            pointmap_target = voxel_utils.unproject_cv(uvs, depth_target, intrinsic_target)[mask]
        else:
            depth_src_aligned = geometry_utils.align_depthmap(depth_source.unsqueeze(0), depth_target.unsqueeze(0), mask=mask.unsqueeze(0))[0]
            pointmap_target = voxel_utils.unproject_cv(uvs, depth_src_aligned, intrinsic_target)[mask]
        
        for i in range(1):
            similarity_transform = corresponding_points_alignment(
                pointmap_source.unsqueeze(0), 
                pointmap_target.unsqueeze(0), 
                estimate_scale=True)
            
            R, T, scale = similarity_transform.R[0], similarity_transform.T[0], similarity_transform.s[0]
            pointmap_source_aligned = scale * pointmap_source @ R + T.unsqueeze(0)

        RT_new = torch.eye(4, device='cuda')
        RT_new[:3,:3] = R.t()
        RT_new[:3,3] = T
        RTs.append(RT_new)
        scales.append(scale)
        all_points.append(pointmap_target)
        all_points.append(pointmap_source_aligned)
    
    RTs = torch.stack(RTs, dim=0)
    scales = torch.stack(scales, dim=0)
    intrinsic_compute = intrinsic_targets[0]

    return RTs, scales, intrinsic_compute, all_points


def align_depthmap_batch(
    intrinsic_sources, 
    depth_sources, 
    intrinsic_targets, 
    depth_targets, 
    source_masks, 
    target_masks, 
    align_solver='umeyama',
    ):
    assert len(intrinsic_sources) == len(depth_sources) == len(intrinsic_targets) == len(depth_targets) == len(source_masks) == len(target_masks)
    if align_solver == 'umeyama':
        RTs, scales, intrinsic, points_target = align_depthmap(intrinsic_sources, depth_sources, intrinsic_targets, depth_targets, source_masks, target_masks)
    elif align_solver == 'moge':
        RTs, scales, intrinsic, points_target = align_depthmap(intrinsic_sources, depth_sources, intrinsic_targets, depth_targets, source_masks, target_masks, use_moge_depth=True)
    else:
        raise ValueError(f"Unknown align_solver: {align_solver}")
    return RTs, scales, intrinsic, points_target
