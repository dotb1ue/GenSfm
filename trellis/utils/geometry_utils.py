import math
import torch
import torch.nn.functional as F
from torch import Tensor, LongTensor
from typing import *
from itertools import chain
from numbers import Integral, Number
from .voxel_utils import unproject_cv

# Code borrowed from https://github.com/microsoft/MoGe and https://github.com/EasternJournalist/utils3d

def scatter_min(size: int, dim: int, index: LongTensor, src: Tensor) -> torch.return_types.min:
    "Scatter the minimum value along the given dimension of `input` into `src` at the indices specified in `index`."
    shape = src.shape[:dim] + (size,) + src.shape[dim + 1:]
    minimum = torch.full(shape, float('inf'), dtype=src.dtype, device=src.device).scatter_reduce(dim=dim, index=index, src=src, reduce='amin', include_self=False)
    minimum_where = torch.where(src == torch.gather(minimum, dim=dim, index=index))
    indices = torch.full(shape, -1, dtype=torch.long, device=src.device)
    indices[(*minimum_where[:dim], index[minimum_where], *minimum_where[dim + 1:])] = minimum_where[dim]
    return torch.return_types.min((minimum, indices))


def split_batch_fwd(fn: Callable, chunk_size: int, *args, **kwargs):
    batch_size = next(x for x in (*args, *kwargs.values()) if isinstance(x, Tensor)).shape[0]
    n_chunks = batch_size // chunk_size + (batch_size % chunk_size > 0)
    splited_args = tuple(arg.split(chunk_size, dim=0) if isinstance(arg, Tensor) else [arg] * n_chunks for arg in args)
    splited_kwargs = {k: [v.split(chunk_size, dim=0) if isinstance(v, Tensor) else [v] * n_chunks] for k, v in kwargs.items()}
    results = []
    for i in range(n_chunks):
        chunk_args = tuple(arg[i] for arg in splited_args)
        chunk_kwargs = {k: v[i] for k, v in splited_kwargs.items()}
        results.append(fn(*chunk_args, **chunk_kwargs))

    if isinstance(results[0], tuple):
        return tuple(torch.cat(r, dim=0) for r in zip(*results))
    else:
        return torch.cat(results, dim=0)


def _pad_inf(x_: Tensor):
    return torch.cat([torch.full_like(x_[..., :1], -torch.inf), x_, torch.full_like(x_[..., :1], torch.inf)], dim=-1)


def _pad_cumsum(cumsum: Tensor):
    return torch.cat([torch.zeros_like(cumsum[..., :1]), cumsum, cumsum[..., -1:]], dim=-1)


def _compute_residual(a: Tensor, xyw: Tensor, trunc: float):
    return a.mul(xyw[..., 0]).sub_(xyw[..., 1]).abs_().mul_(xyw[..., 2]).clamp_max_(trunc).sum(dim=-1)


def align(x: Tensor, y: Tensor, w: Tensor, trunc: Optional[Union[float, Tensor]] = None, eps: float = 1e-7) -> Tuple[Tensor, Tensor, torch.LongTensor]:
    """
    If trunc is None, solve `min sum_i w_i * |a * x_i - y_i|`, otherwise solve `min sum_i min(trunc, w_i * |a * x_i - y_i|)`.
    
    w_i must be >= 0.

    ### Parameters:
    - `x`: tensor of shape (..., n)
    - `y`: tensor of shape (..., n)
    - `w`: tensor of shape (..., n)
    - `trunc`: optional, float or tensor of shape (..., n) or None

    ### Returns:
    - `a`: tensor of shape (...), differentiable
    - `loss`: tensor of shape (...), value of loss function at `a`, detached
    - `index`: tensor of shape (...), where a = y[idx] / x[idx]
    """
    if trunc is None:
        x, y, w = torch.broadcast_tensors(x, y, w)
        sign = torch.sign(x)
        x, y = x * sign, y * sign
        y_div_x = y / x.clamp_min(eps)
        y_div_x, argsort = y_div_x.sort(dim=-1)

        wx = torch.gather(x * w, dim=-1, index=argsort)
        derivatives = 2 * wx.cumsum(dim=-1) - wx.sum(dim=-1, keepdim=True)
        search = torch.searchsorted(derivatives, torch.zeros_like(derivatives[..., :1]), side='left').clamp_max(derivatives.shape[-1] - 1)

        a = y_div_x.gather(dim=-1, index=search).squeeze(-1)
        index = argsort.gather(dim=-1, index=search).squeeze(-1)
        loss = (w * (a[..., None] * x - y).abs()).sum(dim=-1)
        
    else:
        # Reshape to (batch_size, n) for simplicity
        x, y, w = torch.broadcast_tensors(x, y, w)
        batch_shape = x.shape[:-1]
        batch_size = math.prod(batch_shape)
        x, y, w = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1]), w.reshape(-1, w.shape[-1])

        sign = torch.sign(x)
        x, y = x * sign, y * sign
        wx, wy = w * x, w * y
        xyw = torch.stack([x, y, w], dim=-1)    # Stacked for convenient gathering

        y_div_x = A = y / x.clamp_min(eps)
        B = (wy - trunc) / wx.clamp_min(eps)
        C = (wy + trunc) / wx.clamp_min(eps)
        with torch.no_grad():
            # Caculate prefix sum by orders of A, B, C    
            A, A_argsort = A.sort(dim=-1)
            Q_A = torch.cumsum(torch.gather(wx, dim=-1, index=A_argsort), dim=-1)
            A, Q_A = _pad_inf(A), _pad_cumsum(Q_A)    # Pad [-inf, A1, ..., An, inf] and [0, Q1, ..., Qn, Qn] to handle edge cases.

            B, B_argsort = B.sort(dim=-1)
            Q_B = torch.cumsum(torch.gather(wx, dim=-1, index=B_argsort), dim=-1)
            B, Q_B = _pad_inf(B), _pad_cumsum(Q_B)

            C, C_argsort = C.sort(dim=-1)
            Q_C = torch.cumsum(torch.gather(wx, dim=-1, index=C_argsort), dim=-1)
            C, Q_C = _pad_inf(C), _pad_cumsum(Q_C)
            
            # Caculate left and right derivative of A
            j_A = torch.searchsorted(A, y_div_x, side='left').sub_(1)
            j_B = torch.searchsorted(B, y_div_x, side='left').sub_(1)
            j_C = torch.searchsorted(C, y_div_x, side='left').sub_(1)
            left_derivative = 2 * torch.gather(Q_A, dim=-1, index=j_A) - torch.gather(Q_B, dim=-1, index=j_B) - torch.gather(Q_C, dim=-1, index=j_C)
            j_A = torch.searchsorted(A, y_div_x, side='right').sub_(1)
            j_B = torch.searchsorted(B, y_div_x, side='right').sub_(1)
            j_C = torch.searchsorted(C, y_div_x, side='right').sub_(1)
            right_derivative = 2 * torch.gather(Q_A, dim=-1, index=j_A) - torch.gather(Q_B, dim=-1, index=j_B) - torch.gather(Q_C, dim=-1, index=j_C)

            # Find extrema
            is_extrema = (left_derivative < 0) & (right_derivative >= 0)
            is_extrema[..., 0] |= ~is_extrema.any(dim=-1)                       # In case all derivatives are zero, take the first one as extrema.
            where_extrema_batch, where_extrema_index = torch.where(is_extrema)          

            # Calculate objective value at extrema
            extrema_a = y_div_x[where_extrema_batch, where_extrema_index]               # (num_extrema,)
            MAX_ELEMENTS = 4096 ** 2      # Split into small batches to avoid OOM in case there are too many extrema.(~1G)
            SPLIT_SIZE = MAX_ELEMENTS // x.shape[-1]
            extrema_value = torch.cat([
                _compute_residual(extrema_a_split[:, None], xyw[extrema_i_split, :, :], trunc)
                for extrema_a_split, extrema_i_split in zip(extrema_a.split(SPLIT_SIZE), where_extrema_batch.split(SPLIT_SIZE))
            ])          # (num_extrema,)
            
            # Find minima among corresponding extrema
            minima, indices = scatter_min(size=batch_size, dim=0, index=where_extrema_batch, src=extrema_value)        # (batch_size,)
            index = where_extrema_index[indices]

        a = torch.gather(y, dim=-1, index=index[..., None]) / torch.gather(x, dim=-1, index=index[..., None]).clamp_min(eps)
        a = a.reshape(batch_shape)
        loss = minima.reshape(batch_shape)
        index = index.reshape(batch_shape)

    return a, loss, index


def align_depth_scale(depth_src: Tensor, depth_tgt: Tensor, weight: Optional[Tensor], trunc: Optional[Union[float, Tensor]] = None):
    """
    Align `depth_src` to `depth_tgt` with given constant weights. 

    ### Parameters:
    - `depth_src: Tensor` of shape (..., N)
    - `depth_tgt: Tensor` of shape (..., N)

    """
    scale, _, _ = align(depth_src, depth_tgt, weight, trunc)

    return scale


def align_depth_affine(depth_src: Tensor, depth_tgt: Tensor, weight: Optional[Tensor], trunc: Optional[Union[float, Tensor]] = None):
    """
    Align `depth_src` to `depth_tgt` with given constant weights.

    ### Parameters:
    - `depth_src: Tensor` of shape (..., N)
    - `depth_tgt: Tensor` of shape (..., N)
    - `weight: Tensor` of shape (..., N)
    - `trunc: float` or tensor of shape (..., N) or None

    ### Returns:
    - `scale: Tensor` of shape (...).
    - `shift: Tensor` of shape (...).
    """
    dtype, device = depth_src.dtype, depth_src.device
 
    # Flatten batch dimensions for simplicity
    batch_shape, n = depth_src.shape[:-1], depth_src.shape[-1]
    batch_size = math.prod(batch_shape)
    depth_src, depth_tgt, weight = depth_src.reshape(batch_size, n), depth_tgt.reshape(batch_size, n), weight.reshape(batch_size, n)

    # Here, we take anchors only for non-zero weights.
    # Although the results will be still correct even anchor points have zero weight,
    # it is wasting computation and may cause instability in some cases, e.g. too many extrema.
    anchors_where_batch, anchors_where_n = torch.where(weight > 0)

    # Stop gradient when solving optimal anchors
    with torch.no_grad():
        depth_src_anchor = depth_src[anchors_where_batch, anchors_where_n]                              # (anchors)
        depth_tgt_anchor = depth_tgt[anchors_where_batch, anchors_where_n]                              # (anchors)

        depth_src_anchored = depth_src[anchors_where_batch, :] - depth_src_anchor[..., None]            # (anchors, n)
        depth_tgt_anchored = depth_tgt[anchors_where_batch, :] - depth_tgt_anchor[..., None]            # (anchors, n)
        weight_anchored = weight[anchors_where_batch, :]                                                # (anchors, n)

        scale, loss, index = align(depth_src_anchored, depth_tgt_anchored, weight_anchored, trunc)      # (anchors)

        loss, index_anchor = scatter_min(size=batch_size, dim=0, index=anchors_where_batch, src=loss)   # (batch_size,)

    # Reproduce by indexing for shorter compute graph
    index_1 = anchors_where_n[index_anchor]      # (batch_size,)
    index_2 = index[index_anchor]                # (batch_size,)

    tgt_1, src_1 = torch.gather(depth_tgt, dim=1, index=index_1[..., None]).squeeze(-1), torch.gather(depth_src, dim=1, index=index_1[..., None]).squeeze(-1)
    tgt_2, src_2 = torch.gather(depth_tgt, dim=1, index=index_2[..., None]).squeeze(-1), torch.gather(depth_src, dim=1, index=index_2[..., None]).squeeze(-1)

    scale = (tgt_2 - tgt_1) / torch.where(src_2 != src_1, src_2 - src_1, 1e-7)
    shift = tgt_1 - scale * src_1

    scale, shift = scale.reshape(batch_shape), shift.reshape(batch_shape)

    return scale, shift


def align_depth_affine_irls(depth_src: Tensor, depth_tgt: Tensor, weight: Optional[Tensor], max_iter: int = 100, eps: float = 1e-12):
    """
    Align `depth_src` to `depth_tgt` with given constant weights using IRLS.
    """
    dtype, device = depth_src.dtype, depth_src.device
    
    w = weight
    x = torch.stack([depth_src, torch.ones_like(depth_src)], dim=-1)
    y = depth_tgt

    for i in range(max_iter):
        beta = (x.transpose(-1, -2) @ (w * y)) @ (x.transpose(-1, -2) @ (w[..., None] * x)).inverse().transpose(-2, -1)
        w = 1 / (y - (x @ beta[..., None])[..., 0]).abs().clamp_min(eps)

    return beta[..., 0], beta[..., 1]


def align_points_scale(points_src: Tensor, points_tgt: Tensor, weight: Optional[Tensor], trunc: Optional[Union[float, Tensor]] = None):
    """
    ### Parameters:
    - `points_src: Tensor` of shape (..., N, 3)
    - `points_tgt: Tensor` of shape (..., N, 3)
    - `weight: Tensor` of shape (..., N)

    ### Returns:
    - `a: Tensor` of shape (...). Only positive solutions are garunteed. You should filter out negative scales before using it.
    - `b: Tensor` of shape (...)
    """
    dtype, device = points_src.dtype, points_src.device
    
    scale, _, _ = align(points_src.flatten(-2), points_tgt.flatten(-2), weight[..., None].expand_as(points_src).flatten(-2), trunc)

    return scale


def align_points_scale_z_shift(points_src: Tensor, points_tgt: Tensor, weight: Optional[Tensor], trunc: Optional[Union[float, Tensor]] = None):
    """
    Align `points_src` to `points_tgt` with respect to a shared xyz scale and z shift. 
    It is similar to `align_affine` but scale and shift are applied to different dimensions.

    ### Parameters:
    - `points_src: Tensor` of shape (..., N, 3)
    - `points_tgt: Tensor` of shape (..., N, 3)
    - `weights: Tensor` of shape (..., N)

    ### Returns:
    - `scale: Tensor` of shape (...).
    - `shift: Tensor` of shape (..., 3). x and y shifts are zeros.
    """
    dtype, device = points_src.dtype, points_src.device

    # Flatten batch dimensions for simplicity
    batch_shape, n = points_src.shape[:-2], points_src.shape[-2]
    batch_size = math.prod(batch_shape)
    points_src, points_tgt, weight = points_src.reshape(batch_size, n, 3), points_tgt.reshape(batch_size, n, 3), weight.reshape(batch_size, n)

    # Take anchors
    anchor_where_batch, anchor_where_n = torch.where(weight > 0)
    with torch.no_grad():
        zeros = torch.zeros(anchor_where_batch.shape[0], device=device, dtype=dtype)
        points_src_anchor = torch.stack([zeros, zeros, points_src[anchor_where_batch, anchor_where_n, 2]], dim=-1)      # (anchors, 3)
        points_tgt_anchor = torch.stack([zeros, zeros, points_tgt[anchor_where_batch, anchor_where_n, 2]], dim=-1)      # (anchors, 3)

        points_src_anchored = points_src[anchor_where_batch, :, :] - points_src_anchor[..., None, :]    # (anchors, n, 3)
        points_tgt_anchored = points_tgt[anchor_where_batch, :, :] - points_tgt_anchor[..., None, :]    # (anchors, n, 3)
        weight_anchored = weight[anchor_where_batch, :, None].expand(-1, -1, 3)                         # (anchors, n, 3)

        # Solve optimal scale and shift for each anchor
        MAX_ELEMENTS = 2 ** 20
        scale, loss, index = split_batch_fwd(align, MAX_ELEMENTS // n, points_src_anchored.flatten(-2), points_tgt_anchored.flatten(-2), weight_anchored.flatten(-2), trunc)   # (anchors,)

        loss, index_anchor = scatter_min(size=batch_size, dim=0, index=anchor_where_batch, src=loss)    # (batch_size,)

    # Reproduce by indexing for shorter compute graph
    index_2 = index[index_anchor]                               # (batch_size,) [0, 3n)
    index_1 = anchor_where_n[index_anchor] * 3 + index_2 % 3    # (batch_size,) [0, 3n)

    zeros = torch.zeros((batch_size, n), device=device, dtype=dtype)
    points_tgt_00z, points_src_00z = torch.stack([zeros, zeros, points_tgt[..., 2]], dim=-1), torch.stack([zeros, zeros, points_src[..., 2]], dim=-1)
    tgt_1, src_1 = torch.gather(points_tgt_00z.flatten(-2), dim=1, index=index_1[..., None]).squeeze(-1), torch.gather(points_src_00z.flatten(-2), dim=1, index=index_1[..., None]).squeeze(-1)
    tgt_2, src_2 = torch.gather(points_tgt.flatten(-2), dim=1, index=index_2[..., None]).squeeze(-1), torch.gather(points_src.flatten(-2), dim=1, index=index_2[..., None]).squeeze(-1)

    scale = (tgt_2 - tgt_1) / torch.where(src_2 != src_1, src_2 - src_1, 1.0)
    shift = torch.gather(points_tgt_00z, dim=1, index=(index_1 // 3)[..., None, None].expand(-1, -1, 3)).squeeze(-2) - scale[..., None] * torch.gather(points_src_00z, dim=1, index=(index_1 // 3)[..., None, None].expand(-1, -1, 3)).squeeze(-2)
    scale, shift = scale.reshape(batch_shape), shift.reshape(*batch_shape, 3)

    return scale, shift


def align_points_scale_xyz_shift(points_src: Tensor, points_tgt: Tensor, weight: Optional[Tensor], trunc: Optional[Union[float, Tensor]] = None, max_iters: int = 30, eps: float = 1e-6):
    """
    Align `points_src` to `points_tgt` with respect to a shared xyz scale and z shift. 
    It is similar to `align_affine` but scale and shift are applied to different dimensions.

    ### Parameters:
    - `points_src: Tensor` of shape (..., N, 3)
    - `points_tgt: Tensor` of shape (..., N, 3)
    - `weights: Tensor` of shape (..., N)

    ### Returns:
    - `scale: Tensor` of shape (...).
    - `shift: Tensor` of shape (..., 3)
    """
    dtype, device = points_src.dtype, points_src.device

    # Flatten batch dimensions for simplicity
    batch_shape, n = points_src.shape[:-2], points_src.shape[-2]
    batch_size = math.prod(batch_shape)
    points_src, points_tgt, weight = points_src.reshape(batch_size, n, 3), points_tgt.reshape(batch_size, n, 3), weight.reshape(batch_size, n)

    # Take anchors
    anchor_where_batch, anchor_where_n = torch.where(weight > 0)

    with torch.no_grad():
        points_src_anchor = points_src[anchor_where_batch, anchor_where_n]          # (anchors, 3)
        points_tgt_anchor = points_tgt[anchor_where_batch, anchor_where_n]          # (anchors, 3)

        points_src_anchored = points_src[anchor_where_batch, :, :] - points_src_anchor[..., None, :]    # (anchors, n, 3)
        points_tgt_anchored = points_tgt[anchor_where_batch, :, :] - points_tgt_anchor[..., None, :]    # (anchors, n, 3)
        weight_anchored = weight[anchor_where_batch, :, None].expand(-1, -1, 3)                         # (anchors, n, 3)

        # Solve optimal scale and shift for each anchor
        MAX_ELEMENTS = 2 ** 20
        scale, loss, index = split_batch_fwd(align, MAX_ELEMENTS // 2, points_src_anchored.flatten(-2), points_tgt_anchored.flatten(-2), weight_anchored.flatten(-2), trunc)   # (anchors,)

        # Get optimal scale and shift for each batch element
        loss, index_anchor = scatter_min(size=batch_size, dim=0, index=anchor_where_batch, src=loss)    # (batch_size,)

    index_2 = index[index_anchor]                               # (batch_size,) [0, 3n)
    index_1 = anchor_where_n[index_anchor] * 3 + index_2 % 3    # (batch_size,) [0, 3n)

    src_1, tgt_1 = torch.gather(points_src.flatten(-2), dim=1, index=index_1[..., None]).squeeze(-1), torch.gather(points_tgt.flatten(-2), dim=1, index=index_1[..., None]).squeeze(-1)
    src_2, tgt_2 = torch.gather(points_src.flatten(-2), dim=1, index=index_2[..., None]).squeeze(-1), torch.gather(points_tgt.flatten(-2), dim=1, index=index_2[..., None]).squeeze(-1)

    scale = (tgt_2 - tgt_1) / torch.where(src_2 != src_1, src_2 - src_1, 1.0)
    shift = torch.gather(points_tgt, dim=1, index=(index_1 // 3)[..., None, None].expand(-1, -1, 3)).squeeze(-2) - scale[..., None] * torch.gather(points_src, dim=1, index=(index_1 // 3)[..., None, None].expand(-1, -1, 3)).squeeze(-2)

    scale, shift = scale.reshape(batch_shape), shift.reshape(*batch_shape, 3)

    return scale, shift


def align_depthmap(depth_src: Tensor, depth_dst: Tensor, mask: Tensor, align_method: str = 'affine') -> Tensor:
    """
    Align the scale of the predicted depths and target depths.
    Pixels with depths less or equal to 0 are masked during alignment.

    Args:
        depth_src: (N, H, W) tensor of source depth maps
        depth_dst: (N, H, W) tensor of target depth maps
        mask: (N, H, W) bool tensor of valid map
        align_method: Method to align the scale. Options: 'scale', 'affine', 'irls'.

    Returns:
        depth_src_aligned: (N, H, W) tensor of aligned predicted depths
    """
    assert depth_src.shape == depth_dst.shape == mask.shape and depth_src.ndim == 3
    N = depth_src.shape[0]

    depth_src_resized, depth_dst_resized, mask_resized = masked_nearest_resize(depth_src, depth_dst, mask=mask, size=(64, 64))
    depth_src_aligned = torch.zeros_like(depth_src)

    for i in range(N):
        mask_i = mask_resized[i]
        # If the intersection of mask is empty, skip
        if mask_i.sum() == 0:
            # try to align the depth map by their median
            depth_src_i = depth_src[i][depth_src[i] > 0]
            depth_dst_i = depth_dst[i][depth_dst[i] > 0]
            scale = depth_src_i.median() / depth_dst_i.median()
            depth_src_aligned[i] = depth_src[i] * scale
            continue

        depth_src_i = depth_src_resized[i][mask_i]
        depth_dst_i = depth_dst_resized[i][mask_i]
        weight_i = 1.0 / depth_dst_i
        if align_method == 'scale':
            scale = align_depth_scale(depth_src_i, depth_dst_i, weight_i)
            shift = None
        elif align_method == 'affine':
            scale, shift = align_depth_affine(depth_src_i, depth_dst_i, weight_i)
        elif align_method == 'irls':
            scale, shift = align_depth_affine_irls(depth_src_i, depth_dst_i, weight_i)
        else:
            raise ValueError(f"Invalid align_method: {align_method}")

        depth_src_aligned_i = depth_src[i] * scale
        if shift is not None:
            depth_src_aligned_i += shift
        depth_src_aligned[i] = depth_src_aligned_i
    
    return depth_src_aligned


def align_pointmap(pointmap_src: Tensor, pointmap_dst: Tensor, mask: Tensor, align_method: str = 'affine') -> Tensor:
    """
    Align the scale of the predicted point clouds and target point clouds.

    Args:
        pointmap_src: (N, H, W, 3) tensor of source point clouds
        pointmap_dst: (N, H, W, 3) tensor of target point clouds
        mask: (N, H, W) bool tensor of valid map
        align_method: Method to align the scale.

    Returns:
        pointmap_src_aligned: (N, H, W, 3) tensor of aligned predicted point clouds
    """
    assert pointmap_src.shape == pointmap_dst.shape and pointmap_src.ndim == 4
    assert pointmap_src.shape[:-1] == mask.shape
    N = pointmap_src.shape[0]

    pointmap_src_resized, pointmap_dst_resized, mask_resized = masked_nearest_resize(pointmap_src, pointmap_dst, mask=mask, size=(64, 64))
    pointmap_src_aligned = torch.zeros_like(pointmap_src)

    for i in range(N):
        mask_i = mask_resized[i]
        # If the intersection of mask is empty, skip
        if mask_i.sum() == 0:
            pointmap_src_aligned[i] = pointmap_src[i]
            continue

        pointmap_src_i = pointmap_src_resized[i][mask_i]
        pointmap_dst_i = pointmap_dst_resized[i][mask_i]
        weight_i = 1.0 / pointmap_dst_i[:, 2]
        if align_method == 'scale':
            scale = align_points_scale(pointmap_src_i, pointmap_dst_i, weight_i)
            shift = None
        elif align_method == 'scale_z_shift':
            scale, shift = align_points_scale_z_shift(pointmap_src_i, pointmap_dst_i, weight_i)
        elif align_method == 'scale_xyz_shift':
            scale, shift = align_points_scale_xyz_shift(pointmap_src_i, pointmap_dst_i, weight_i)
        else:
            raise ValueError(f"Invalid align_method: {align_method}")

        pointmap_src_aligned_i = pointmap_src[i] * scale
        if shift is not None:
            pointmap_src_aligned_i += shift
        pointmap_src_aligned[i] = pointmap_src_aligned_i

    return pointmap_src_aligned


def pointmap_chamfer_distance(
    depth_pred: Tensor, 
    depth_gt: Tensor, 
    mask_pred: Optional[Tensor] = None, 
    mask_gt: Optional[Tensor] = None,
    extrinsics_pred: Optional[Tensor] = None,
    intrinsics_pred: Optional[Tensor] = None,
    extrinsics_gt: Optional[Tensor] = None,
    intrinsics_gt: Optional[Tensor] = None,
    pointmap_align_method: Optional[str] = None,
    max_distance: float = 1.0,
) -> tuple[Tensor, list[Tensor], list[Tensor]]:
    """
    Compute the Chamfer distance between the predicted and target point clouds.

    Args:
        pred: (N, H, W) tensor of predicted depths
        target: (N, H, W) tensor of target point clouds
        mask_pred: (N, H, W) tensor of mask
        mask_gt: (N, H, W) tensor of mask
        extrinsics_pred: (N, 4, 4) tensor of predicted extrinsics
        intrinsics_pred: (N, 3, 3) tensor of predicted intrinsics
        extrinsics_gt: (N, 4, 4) tensor of target extrinsics
        intrinsics_gt: (N, 3, 3) tensor of target intrinsics
        pointmap_align_method: Method to align the point clouds.
        max_distance: Maximum distance when points are empty or alignment completely fails. Default is 1.0.

    Returns:
        chamfer_distance: (N,) tensor of Chamfer distance
        pointmap_preds: N x (n, 3) tensor of predicted point cloud
        pointmap_gts: N x (n, 3) tensor of target point cloud
    """
    assert depth_pred.ndim == 3 and depth_gt.ndim == 3 and depth_pred.shape == depth_gt.shape
    N, H, W = depth_pred.shape
    if extrinsics_gt is None:
        extrinsics_gt = extrinsics_pred
    if intrinsics_gt is None:
        intrinsics_gt = intrinsics_pred

    grid_x = (torch.arange(0, W, dtype=torch.float32, device=depth_pred.device) + 0.5) / W
    grid_y = (torch.arange(0, H, dtype=torch.float32, device=depth_pred.device) + 0.5) / H
    uvs = torch.stack(torch.meshgrid(grid_x, grid_y, indexing='ij'), dim=-1)

    empty = torch.zeros((0, 3), device=depth_pred.device)
    pointmap_preds = []
    pointmap_gts = []
    for i in range(N):
        depth_pred_i = depth_pred[i]
        depth_gt_i = depth_gt[i]
        mask_pred_i = mask_pred[i] if mask_pred is not None else torch.ones_like(depth_pred_i, dtype=torch.bool)
        mask_gt_i = mask_gt[i] if mask_gt is not None else torch.ones_like(depth_gt_i, dtype=torch.bool)
        # GT empty: chamfer=0 via point_chamfer_distance. Pred empty (gt non-empty)
        # falls through to the masked-unproject branch, producing an empty pred so
        # point_chamfer_distance returns max_distance.
        if mask_gt_i.sum() == 0:
            pointmap_preds.append(empty)
            pointmap_gts.append(empty)
            continue

        extrinsics_pred_i = None if extrinsics_pred is None else extrinsics_pred[i]
        extrinsics_gt_i = None if extrinsics_gt is None else extrinsics_gt[i]
        intrinsics_pred_i = intrinsics_pred[i]
        intrinsics_gt_i = intrinsics_gt[i]

        if pointmap_align_method is not None and mask_pred_i.sum() > 0:
            pointmap_pred_i = unproject_cv(uvs, depth_pred_i, intrinsics_pred_i, extrinsics_pred_i)
            pointmap_gt_i = unproject_cv(uvs, depth_gt_i, intrinsics_gt_i, extrinsics_gt_i)
            mask_intersection_i = mask_pred_i & mask_gt_i
            pointmap_pred_i = align_pointmap(pointmap_pred_i[None], pointmap_gt_i[None], mask_intersection_i[None], pointmap_align_method)[0]
            pointmap_pred_i = pointmap_pred_i[mask_pred_i]
            pointmap_gt_i = pointmap_gt_i[mask_gt_i]
        else:
            depth_pred_i_masked = depth_pred_i[mask_pred_i]
            depth_gt_i_masked = depth_gt_i[mask_gt_i]
            uvs_pred_i_masked = uvs[mask_pred_i]
            uvs_gt_i_masked = uvs[mask_gt_i]
            pointmap_pred_i = unproject_cv(uvs_pred_i_masked, depth_pred_i_masked, intrinsics_pred_i, extrinsics_pred_i)
            pointmap_gt_i = unproject_cv(uvs_gt_i_masked, depth_gt_i_masked, intrinsics_gt_i, extrinsics_gt_i)
        pointmap_preds.append(pointmap_pred_i)
        pointmap_gts.append(pointmap_gt_i)

    chamfer_distances = point_chamfer_distance(pointmap_preds, pointmap_gts, max_distance).clamp_max_(max_distance)
    return chamfer_distances, pointmap_preds, pointmap_gts


def point_chamfer_distance(point_preds: list[Tensor], point_gts: list[Tensor], max_distance: float = 1.0) -> Tensor:
    """
    Bidirectional symmetric Chamfer distance between predicted and GT
    point clouds (protocol §4.2 / §8.1 #3).

    For each (pred, gt) pair: forward = mean_pred ||pred -> nearest gt||²,
    backward = mean_gt ||gt -> nearest pred||², CD = (forward + backward) * scale.
    Mean is over the squared L2 of nearest-neighbor pairs (chamferdist's default).
    Both clouds must be in the same frame.

    Args:
        point_preds: List[N] of [P_pred, 3] predicted point clouds.
        point_gts: List[N] of [P_gt, 3] GT point clouds.
        max_distance: Per-instance fallback when one side is empty. Default 1.0.

    Returns:
        Tensor[N] of per-instance CD.
    """
    from pytorch3d.loss import chamfer_distance

    assert len(point_preds) == len(point_gts)
    out = []
    for pred, gt in zip(point_preds, point_gts):
        if gt.shape[0] == 0:
            out.append(torch.tensor(0.0, device=pred.device))
        elif pred.shape[0] == 0:
            out.append(torch.tensor(max_distance, device=pred.device))
        else:
            d, _ = chamfer_distance(pred[None], gt[None])
            out.append(d)
    return torch.stack(out)


def point_f_score(point_preds: list[Tensor], point_gts: list[Tensor], r: List[float]) -> List[Tensor]:
    """
    Compute the F-score between the predicted and target point clouds.

    Supports mismatched per-pair point counts (P_pred ≠ P_gt). Returns 0 when
    either side is empty.

    Args:
        point_preds: List[N] of [P_pred, 3] predicted point clouds.
        point_gts: List[N] of [P_gt, 3] ground truth point clouds.
        r: A list of distance thresholds for matching points.

    Returns:
        A list of F-score tensors (one per threshold), each shape [N].
    """
    from pytorch3d.ops import knn_points

    assert len(point_preds) == len(point_gts)
    NR = len(r)
    N = len(point_preds)
    device = point_preds[0].device

    r_t = torch.tensor(r, dtype=torch.float32, device=device).view(NR, 1)
    precision = torch.zeros(NR, N, device=device)
    recall = torch.zeros(NR, N, device=device)

    for i, (point_pred, point_gt) in enumerate(zip(point_preds, point_gts)):
        n_pred, n_gt = point_pred.shape[0], point_gt.shape[0]
        if n_pred == 0 or n_gt == 0:
            continue

        # sqrt because knn_points returns squared dists
        sq_p2g, _, _ = knn_points(point_pred[None], point_gt[None], K=1)
        min_p2g = sq_p2g.squeeze(0).squeeze(-1).clamp_min_(0).sqrt_()
        sq_g2p, _, _ = knn_points(point_gt[None], point_pred[None], K=1)
        min_g2p = sq_g2p.squeeze(0).squeeze(-1).clamp_min_(0).sqrt_()

        # precision = (# pred within r of some GT) / |pred|
        # recall    = (# GT within r of some pred) / |gt|
        precision[:, i] = (min_p2g[None] <= r_t).sum(dim=1).float() / n_pred
        recall[:, i] = (min_g2p[None] <= r_t).sum(dim=1).float() / n_gt

    f_score = 2 * precision * recall / (precision + recall)
    f_score = torch.nan_to_num(f_score, nan=0.0)
    return f_score.unbind(dim=0)


def uv_map(
    *size: Union[int, Tuple[int, int]],
    top: float = 0.,
    left: float = 0.,
    bottom: float = 1.,
    right: float = 1.,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None
) -> Tensor:
    """
    Get image UV coordinate map. By default, (0., 0.) is the top-left corner of the image, and (1., 1.) is the bottom-right corner of the image.

    ## Parameters
    - `*size`: `Tuple[int, int]` or two integers of map size `(height, width)`
    - `top`: `float` defaults to 0.
    - `left`: `float` defaults to 0.
    - `bottom`: `float` defaults to 1.
    - `right`: `float` defaults to 1.
    - `dtype`: `torch.dtype` data type of the output uv map. Defaults to torch.float32.
    - `device`: `torch.device`, device of the output uv map. Defaults to None.

    ## Returns
    - `uv (Tensor)`: shape `(height, width, 2)`

    ## Example Usage

    >>> uv_map(10, 10):
    [[[0.05, 0.05], [0.15, 0.05], ..., [0.95, 0.05]],
     [[0.05, 0.15], [0.15, 0.15], ..., [0.95, 0.15]],
      ...             ...                  ...
     [[0.05, 0.95], [0.15, 0.95], ..., [0.95, 0.95]]]
    """
    if len(size) == 1 and isinstance(size[0], tuple):
        height, width = size[0]
    else:
        height, width = size
    u = torch.linspace(left + 0.5 / width * (right - left), right - 0.5 / width * (right - left), width, dtype=dtype, device=device)
    v = torch.linspace(top + 0.5 / height * (bottom - top), bottom - 0.5 / height * (bottom - top), height, dtype=dtype, device=device)
    uv = torch.stack(torch.meshgrid(u, v, indexing='xy'), dim=-1)
    return uv


def pixel_coord_map(
    *size: Union[int, Tuple[int, int]],
    top: int = 0,
    left: int = 0,
    convention: Literal['integer-center', 'integer-corner'] = 'integer-center',
    dtype: torch.dtype = torch.float32,
    device: torch.device = None
) -> Tensor:
    """
    Get image pixel coordinates map. Support two conventions: `'integer-center'` and `'integer-corner'`.

    ## Parameters
    - `*size`: `Tuple[int, int]` or two integers of map size `(height, width)`
    - `top`: `int`, optional top boundary of the pixel coord map. Defaults to 0.
    - `left`: `int`, optional left boundary of the pixel coord map. Defaults to 0.
    - `convention`: `str`, optional `'integer-center'` or `'integer-corner'`, whether integer coordinates correspond to pixel centers or corners. Defaults to 'integer-center'.
        - `'integer-center'`: `pixel[i][j]` has integer coordinates `(j, i)` as its center, and occupies square area `[j - 0.5, j + 0.5) × [i - 0.5, i + 0.5)`. 
            The top-left corner of the top-left pixel is `(-0.5, -0.5)`, and the bottom-right corner of the bottom-right pixel is `(width - 0.5, height - 0.5)`.
        - `'integer-corner'`: `pixel[i][j]` has coordinates `(j + 0.5, i + 0.5)` as its center, and occupies square area `[j, j + 1) × [i, i + 1)`.
            The top-left corner of the top-left pixel is `(0, 0)`, and the bottom-right corner of the bottom-right pixel is `(width, height)`.
    - `dtype`: `torch.dtype`, optional data type of the output pixel coord map. Defaults to torch.float32.

    ## Returns
        Tensor: shape (height, width, 2)
    
    >>> pixel_coord_map(10, 10, convention='integer-center', dtype=torch.long):
    [[[0, 0], [1, 0], ..., [9, 0]],
     [[0, 1], [1, 1], ..., [9, 1]],
        ...      ...         ...
     [[0, 9], [1, 9], ..., [9, 9]]]

    >>> pixel_coord_map(10, 10, convention='integer-corner', dtype=torch.float32):
    [[[0.5, 0.5], [1.5, 0.5], ..., [9.5, 0.5]],
     [[0.5, 1.5], [1.5, 1.5], ..., [9.5, 1.5]],
      ...             ...                  ...
    [[0.5, 9.5], [1.5, 9.5], ..., [9.5, 9.5]]]
    """
    if len(size) == 1 and isinstance(size[0], tuple):
        height, width = size[0]
    else:
        height, width = size
    u = torch.arange(left, left + width, dtype=dtype, device=device)
    v = torch.arange(top, top + height, dtype=dtype, device=device)
    if convention == 'integer-corner':
        assert torch.is_floating_point(u), "dtype should be a floating point type when convention is 'integer-corner'"
        u = u + 0.5
        v = v + 0.5
    u, v = torch.meshgrid(u, v, indexing='xy')
    return torch.stack([u, v], dim=2)


def sliding_window(
    x: Tensor, 
    window_size: Union[int, Tuple[int, ...]], 
    stride: Optional[Union[int, Tuple[int, ...]]] = None, 
    dilation: Optional[Union[int, Tuple[int, ...]]] = None,
    pad_size: Optional[Union[int, Tuple[int, int], Tuple[Tuple[int, int]]]] = None, 
    pad_mode: str = 'constant',
    pad_value: Number = 0,
    dim: Tuple[int, ...] = None
) -> Tensor:
    """
    Get a sliding window of the input array.
    This function is a wrapper of `torch.nn.functional.unfold` with additional support for padding and stride.

    ## Parameters
    - `x` (Tensor): Input tensor.
    - `window_size` (int or Tuple[int,...]): Size of the sliding window. If int
        is provided, the same size is used for all specified axes.
    - `stride` (Optional[Tuple[int,...]]): Stride between the sliding windows. If None,
        no stride is applied. If int is provided, the same stride is used for all specified axes.
    - `dilation` (Optional[Tuple[int,...]]): Dilation in each sliding window. If None,
        no dilation is applied. If int is provided, the same dilation is used for all specified axes.
    - `pad_size` (Optional[Union[int, Tuple[int, int], Tuple[Tuple[int, int]]]]): Size of padding to apply before sliding window.
        Corresponding to `axis`.
        - General format is `((before_1, after_1), (before_2, after_2), ...)`.
        - Shortcut formats: 
            - `int` -> same padding before and after for all axes;
            - `(int, int)` -> same padding before and after for each axis;
            - `((int,), (int,) ...)` -> specify padding for each axis, same before and after.
    - `pad_mode` (str): Padding mode to use. Refer to `numpy.pad` for more details.
    - `pad_value` (Union[int, float]): Value to use for constant padding. Only used
        when `pad_mode` is 'constant'.
    - `axis` (Optional[Tuple[int,...]]): Axes to apply the sliding window. If None, all axes are used.

    ## Returns
    - (Tensor): Sliding window of the input array. 
        - If no padding, the output is a view of the input array with zero copy.
        - Otherwise, the output is no longer a view but a copy of the padded array.
    """
    if dim is None:
        dim = tuple(range(x.ndim))
    if isinstance(dim, Integral):
        dim = (dim,)
    dim = [dim[i] % x.ndim for i in range(len(dim))]
    if isinstance(window_size, Integral):
        window_size = (window_size,) * len(dim)
    if stride is None:
        stride = (1,) * len(dim)
    elif isinstance(stride, Integral):
        stride = (stride,) * len(dim)
    if dilation is None:
        dilation = (1,) * len(dim)
    elif isinstance(dilation, Integral):
        dilation = (dilation,) * len(dim)
    assert len(window_size) == len(stride) == len(dim)

    # Pad the input array if needed
    if pad_size is not None:
        if isinstance(pad_size, Integral):
            pad_size = ((pad_size, pad_size),) * len(dim)
        elif isinstance(pad_size, tuple) and len(pad_size) == 2 and all(isinstance(p, Integral) for p in pad_size):
            pad_size = (pad_size,) * len(dim)
        elif isinstance(pad_size, tuple) and all(isinstance(p, tuple) and 1 <= len(p) <= 2 for p in pad_size):
            if len(pad_size) == 1:
                pad_size = pad_size * len(dim)
            else:
                assert len(pad_size) == len(dim), f"pad_size {pad_size} must match the number of axes {len(dim)}"
            pad_size = tuple(p * 2 if len(p) == 1 else p for p in pad_size)
        else:
            raise ValueError(f"Invalid pad_size {pad_size}")
        full_pad = [(0, 0) if i not in dim else pad_size[dim.index(i)] for i in range(x.ndim)]
        full_pad = tuple(chain(*reversed(full_pad)))
        x = F.pad(x, full_pad, mode=pad_mode, value=pad_value)
    
    for i in range(len(window_size)):
        x = x.unfold(dim[i], (window_size[i] - 1) * dilation[i] + 1, stride[i])[..., ::dilation[i]]
    return x


def masked_nearest_resize(
    *image: Tensor,
    mask: Tensor, 
    size: Tuple[int, int], 
    return_index: bool = False
) -> Tuple[Unpack[Tuple[Tensor, ...]], Tensor, Tuple[Tensor, ...]]:
    """
    Resize image(s) by nearest sampling with mask awareness. Suitable for sparse maps. ![masked_nearest_resize.png](doc/masked_nearest_resize.png)
    - Downsampling: Assign the nearest valid pixel within the target pixel's receptive field.
    - Upsampling: Assign the valid pixel to only the nearest pixel in the resized map.


    ### Parameters
    - `*image`: Itorchut image(s) of shape `(..., H, W, C)` or `(... , H, W)` 
        - You can pass multiple images to be resized at the same time for efficiency.
    - `mask`: itorchut mask of shape `(..., H, W)`, dtype=bool
    - `size`: target size `(H', W')`
    - `return_index`: whether to return the nearest neighbor indices in the original map for each pixel in the resized map.
        Defaults to False.

    ### Returns
    - `*resized_image`: resized image(s) of shape `(..., H', W', C)`. or `(..., H', W')`
    - `resized_mask`: mask of the resized map of shape `(..., H', W')`
    - `nearest_indices`: tuple of shape `(..., H', W')`. The nearest neighbor indices of the resized map of each dimension.
    """
    device = mask.device
    height, width = mask.shape[-2:]
    target_height, target_width = size
    filter_h_f, filter_w_f = height / target_height, width / target_width
    filter_h_i, filter_w_i = math.ceil(filter_h_f), math.ceil(filter_w_f)
    filter_size = filter_h_i * filter_w_i
    filter_shape = (filter_h_i, filter_w_i)
    padding_h, padding_w = filter_h_i // 2 + 1, filter_w_i // 2 + 1
    padding_shape = ((padding_h, padding_h), (padding_w, padding_w))
    
    # Window the original mask and uv
    pixels = pixel_coord_map(height, width, convention='integer-corner', dtype=torch.float32, device=device)
    indices = torch.arange(height * width, dtype=torch.long, device=device).reshape(height, width)
    window_pixels = sliding_window(pixels, window_size=filter_shape, pad_size=padding_shape, dim=(0, 1))
    window_indices = sliding_window(indices, window_size=filter_shape, pad_size=padding_shape, dim=(0, 1))
    window_mask = sliding_window(mask, window_size=filter_shape, pad_size=padding_shape, dim=(-2, -1))

    # Gather the target pixels's local window
    target_centers = uv_map(target_height, target_width, dtype=torch.float32, device=device) * torch.tensor([width, height], dtype=torch.float32, device=device)
    target_lefttop = target_centers - torch.tensor((filter_w_f / 2, filter_h_f / 2), dtype=torch.float32, device=device)
    target_window = torch.round(target_lefttop).to(torch.long) + torch.tensor((padding_w, padding_h), dtype=torch.long, device=device)

    target_window_pixels = window_pixels[target_window[..., 1], target_window[..., 0], :, :, :].reshape(target_height, target_width, 2, filter_size)                  # (target_height, tgt_width, 2, filter_size)
    target_window_mask = window_mask[..., target_window[..., 1], target_window[..., 0], :, :].reshape(*mask.shape[:-2], target_height, target_width, filter_size)     # (..., target_height, tgt_width, filter_size)
    target_window_indices = window_indices[target_window[..., 1], target_window[..., 0], :, :].reshape(target_height, target_width, filter_size)                      # (target_height, tgt_width, filter_size)

    # Compute nearest neighbor in the local window for each pixel 
    delta = target_window_pixels - target_centers[..., None]
    eps = torch.finfo(torch.float32).eps * max(height, width) # Shift a small epsilon to avoid numerical issues when the pixel is exactly on the border
    target_window_mask &= (-filter_w_f / 2 + eps < delta[..., 0, :]) & (delta[..., 0, :] <= filter_w_f / 2 + eps) & (-filter_h_f / 2 + eps < delta[..., 1, :]) & (delta[..., 1, :] <= filter_h_f / 2 + eps)
    dist = torch.where(target_window_mask, torch.square(delta[..., 0, :]) + torch.square(delta[..., 1, :]), torch.inf)      # (..., target_height, tgt_width, filter_size)
    nearest_in_window = torch.argmin(dist, dim=-1, keepdim=True)                                                 # (..., target_height, tgt_width, 1)
    nearest_idx = torch.gather(
        target_window_indices.expand(dist.shape),
        dim=-1,
        index=nearest_in_window,
    ).squeeze(-1)     # (..., target_height, tgt_width)
    nearest_i, nearest_j = nearest_idx // width, nearest_idx % width
    target_mask = torch.any(target_window_mask, dim=-1)
    batch_indices = [torch.arange(n, device=device).reshape([1] * i + [n] + [1] * (mask.ndim - i - 1)) for i, n in enumerate(mask.shape[:-2])]

    nearest_indices = (*batch_indices, nearest_i, nearest_j)
    outputs = tuple(x[nearest_indices] for x in image)

    if return_index:
        ret = (*outputs, target_mask, nearest_indices)
    else:
        ret = (*outputs, target_mask)

    if len(ret) == 1:
        return ret[0]
    else:
        return ret


# Cache MeshRenderer instances (avoid per-call RasterizeCudaContext init).
# Keyed on (resolution, ssaa, device-str); single-process sequential use only.
_MESH_RENDER_CACHE: dict = {}


def farthest_point_sampling(points: Tensor, K: int) -> Tensor:
    """Greedy farthest-point sampling via pytorch3d CUDA op.

    Returns ``[K, 3]`` points spatially well-spread across the input cloud.
    If ``points.shape[0] <= K``, returns the input unchanged. Uses a fixed
    starting point (seed=0) for determinism.
    """
    N = points.shape[0]
    if N <= K:
        return points
    from pytorch3d.ops import sample_farthest_points
    out, _ = sample_farthest_points(points.unsqueeze(0), K=K, random_start_point=False)
    return out.squeeze(0)


def sample_points_from_mesh_views(
    vertices: Tensor,
    faces: Tensor,
    n_samples: int = 4_000,
    n_views: int = 64,
    resolution: int = 512,
    fov: float = 40.0,
    radius: float = 2.0,
    use_fps: bool = True,
) -> Tensor:
    """Render the mesh from ``n_views`` sphere-Hammersley cameras, unproject
    visible-surface depth, return ``n_samples`` 3D points.

    Mirrors the eval1 pipeline: produces only externally-visible surface
    samples (interior cavities excluded), spatially uniform after FPS.

    Args:
        vertices: [V, 3] mesh vertices on GPU.
        faces: [F, 3] mesh faces (long).
        n_samples: target point count after FPS / random subsample.
        n_views: number of sphere-Hammersley camera placements.
        resolution: per-view render resolution (square).
        fov: vertical/horizontal FOV degrees (camera intrinsics use FOVx=FOVy).
        radius: camera radius (cams placed at this distance from origin).
        use_fps: if True, FPS-subsample to ``n_samples``; else uniform random.

    Returns ``[n_samples, 3]`` GPU points. Empty meshes yield zeros.
    """
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        return vertices.new_zeros((n_samples, 3))

    import nvdiffrast.torch as dr
    from ..representations import MeshExtractResult
    from ..renderers.mesh_renderer import intrinsics_to_projection
    from .render_utils import (
        get_renderer,
        yaw_pitch_r_fov_to_extrinsics_intrinsics,
    )
    from .random_utils import sphere_hammersley_sequence
    from .voxel_utils import unproject_cv

    device = vertices.device
    dtype = vertices.dtype

    # Wrap in a MeshExtractResult so get_renderer dispatches to the mesh renderer.
    mesh = MeshExtractResult(vertices=vertices, faces=faces)

    cams = [sphere_hammersley_sequence(i, n_views) for i in range(n_views)]
    yaws = [c[0] for c in cams]
    pitches = [c[1] for c in cams]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws, pitches, radius, fov)

    # Reuse renderer across calls — RasterizeCudaContext init is non-trivial.
    # ssaa=1 (vs MeshRenderer's default 4) skips a 16× internal supersample.
    cache_key = (int(resolution), 1, str(device))
    renderer = _MESH_RENDER_CACHE.get(cache_key)
    if renderer is None:
        renderer = get_renderer(mesh, resolution=resolution, ssaa=1, bg_color=(0, 0, 0))
        _MESH_RENDER_CACHE[cache_key] = renderer

    grid_x = (torch.arange(resolution, device=device, dtype=dtype) + 0.5) / resolution
    grid_y = (torch.arange(resolution, device=device, dtype=dtype) + 0.5) / resolution
    uv_grid = torch.stack(torch.meshgrid(grid_x, grid_y, indexing='xy'), dim=-1)   # [H, W, 2]

    # Batched rasterize: stack all N views into one nvdiffrast call.
    # Mirrors MeshRenderer.render's mask/depth path but vectorized over views.
    # Mask uses dr.antialias + strict ==1.0 to drop silhouette-edge pixels
    # (matches single-view semantics); depth skips AA since edge pixels are
    # filtered out by the strict mask anyway.
    ext_b = torch.stack([e.to(device=device, dtype=dtype) for e in extrinsics], dim=0)    # [N, 4, 4]
    intr_b = torch.stack([i.to(device=device, dtype=dtype) for i in intrinsics], dim=0)   # [N, 3, 3]
    near, far = renderer.rendering_options.near, renderer.rendering_options.far
    persp = torch.stack([intrinsics_to_projection(intr_b[i], near, far) for i in range(ext_b.shape[0])], dim=0)  # [N, 4, 4]
    full_proj = torch.bmm(persp, ext_b)                                                   # [N, 4, 4]

    verts_h = torch.cat([mesh.vertices, torch.ones_like(mesh.vertices[..., :1])], dim=-1) # [V, 4]
    verts_h_b = verts_h.unsqueeze(0).expand(ext_b.shape[0], -1, -1)                       # [N, V, 4]
    verts_cam = torch.bmm(verts_h_b, ext_b.transpose(-1, -2))                             # [N, V, 4]
    verts_clip = torch.bmm(verts_h_b, full_proj.transpose(-1, -2))                        # [N, V, 4]

    faces_int = mesh.faces.int()
    rast, _ = dr.rasterize(renderer.glctx, verts_clip, faces_int, (resolution, resolution))                # [N, H, W, 4]
    mask_aa = dr.antialias((rast[..., -1:] > 0).float(), rast, verts_clip, faces_int)                      # [N, H, W, 1]
    mask_b = mask_aa[..., 0] == 1.0                                                                        # [N, H, W]
    depth_b, _ = dr.interpolate(verts_cam[..., 2:3].contiguous(), rast, faces_int)                         # [N, H, W, 1]
    depth_b = depth_b[..., 0]                                                                              # [N, H, W]

    pts_all = []
    for i in range(ext_b.shape[0]):
        m = mask_b[i]
        uv_vis = uv_grid[m]
        depth_vis = depth_b[i][m]
        pts = unproject_cv(uv_vis, depth_vis, intr_b[i], ext_b[i])
        pts_all.append(pts)

    if not pts_all:
        return vertices.new_zeros((n_samples, 3))
    points = torch.cat(pts_all, dim=0)

    if use_fps and points.shape[0] > n_samples:
        return farthest_point_sampling(points, n_samples)
    if points.shape[0] >= n_samples:
        idx = torch.randperm(points.shape[0], device=device)[:n_samples]
        return points[idx]
    # Fewer than n_samples visible — pad with last point so caller's batching works.
    pad = points[-1:].expand(n_samples - points.shape[0], 3)
    return torch.cat([points, pad], dim=0)


_VALID_MESH_SAMPLE_METHODS = ("area", "fps", "render")


def _sample_points_from_verts_faces(
    verts: Tensor,                   # [V, 3]
    faces: Tensor,                   # [F, 3]
    n: int,
    method: str = "area",
    fps_oversample: int = 4,
) -> Tensor:
    """Triangle-mesh point sampling, GPU-resident.

    ``method ∈ {"area", "fps"}``:
      - ``"area"``: face picked with prob ∝ triangle area; uniform over
        physical surface area.
      - ``"fps"``: oversample area-weighted (``fps_oversample × n``), then
        FPS subsample to ``n``; spatially uniform.
    """
    if verts.shape[0] == 0:
        return verts.new_zeros((n, 3))

    if method not in ("area", "fps"):
        raise ValueError(f"unknown method={method!r}; use one of {_VALID_MESH_SAMPLE_METHODS}")

    # No-face mesh = vertex-only proxy (gauss-as-mesh stand-in for splat-only
    # baselines). Sample directly from vertices with replacement so every
    # batch element returns exactly n points regardless of V.
    if faces.shape[0] == 0:
        V = verts.shape[0]
        if method == "area":
            idx = torch.randint(0, V, (n,), device=verts.device)
            return verts[idx]
        # method == "fps": oversample to n*fps_oversample, then FPS to n.
        idx = torch.randint(0, V, (n * fps_oversample,), device=verts.device)
        return farthest_point_sampling(verts[idx], n)

    from pytorch3d.ops import sample_points_from_meshes
    from pytorch3d.structures import Meshes
    m = n if method == "area" else n * fps_oversample
    mesh = Meshes(verts=[verts], faces=[faces])
    pts = sample_points_from_meshes(mesh, num_samples=m).squeeze(0)
    if method == "fps":
        return farthest_point_sampling(pts, n)
    return pts


def sample_points_from_mesh_extract(
    mesh,
    n: int = 20_000,
    method: str = "area",
    **render_kwargs,
) -> Tensor:
    """Sample ``n`` points from a ``MeshExtractResult``.

    ``method`` selects the sampling strategy:

    - ``"area"`` (default): triangle-area-weighted, uniform over physical
      surface (current default).
    - ``"fps"``: area-weighted oversample, then farthest-point subsample;
      spatially uniform.
    - ``"render"``: render-unproject + FPS via
      ``sample_points_from_mesh_views``; visible-surface only. ``render_kwargs``
      forwarded (``n_views``, ``resolution``, ``fov``, ``radius``, ``use_fps``).

    Empty meshes return ``[n, 3]`` zeros so downstream batching is safe.
    """
    verts = mesh.vertices                                    # [V, 3]
    faces = mesh.faces.long()                                # [F, 3]
    if method == "render":
        return sample_points_from_mesh_views(verts, faces, n_samples=n, **render_kwargs)
    return _sample_points_from_verts_faces(verts, faces, n, method=method)


def sample_points_from_ply_path(
    path: str,
    n: int = 20_000,
    method: str = "area",
    device=None,
    **render_kwargs,
) -> Tensor:
    """Load a mesh from disk and sample ``n`` points; returns a torch.Tensor.

    For all GPU paths (``method ∈ {"fps", "render"}``) ``device`` is required.
    For ``method="area"`` (default) ``device`` is optional — when omitted the
    CPU trimesh fallback is used (returns CPU tensor).
    """
    import trimesh
    mesh = trimesh.load(path, force='mesh')

    if method == "area" and device is None:
        import numpy as np
        pts, _ = trimesh.sample.sample_surface(mesh, n)
        return torch.from_numpy(np.asarray(pts, dtype=np.float32))

    if device is None:
        raise ValueError(
            f"sample_points_from_ply_path: method={method!r} requires device=")
    verts = torch.as_tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.as_tensor(mesh.faces, dtype=torch.long, device=device)

    if method == "render":
        return sample_points_from_mesh_views(verts, faces, n_samples=n, **render_kwargs)
    return _sample_points_from_verts_faces(verts, faces, n, method=method)