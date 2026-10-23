import torch
import torch.nn as nn
from .. import SparseTensor
from .. import DEBUG
from . import SPCONV_ALGO

_INT32_MAX = 2**31 - 1


def _subm_byte_span(conv, data):
    channels = max(conv.in_channels, conv.out_channels)
    itemsize = max(data.features.element_size(), conv.weight.element_size())
    return data.features.shape[0] * channels * itemsize


def _can_tile_subm(conv, data):
    return (
        not torch.is_grad_enabled() and conv.subm and data.batch_size == 1
        and not conv.inverse and not conv.transposed and conv.groups == 1
        and len(conv.kernel_size) == 3 and all(k > 0 and k % 2 for k in conv.kernel_size)
        and any(k > 1 for k in conv.kernel_size) and all(s == 1 for s in conv.stride)
        and all(d > 0 for d in conv.dilation) and data.features.ndim == 2
        and data.indices.shape[1] == 4 and data.features.is_floating_point()
        and conv.weight.is_floating_point()
    )


def _subm_conv_core_halo(conv, data, max_tile_points=None):
    """Run one no-grad SubM layer in spatial cores with complete input halos."""
    if not _can_tile_subm(conv, data) or data.features.shape[0] == 0:
        return conv(data)
    import spconv.pytorch as spconv

    bytes_per_row = max(conv.in_channels, conv.out_channels) * max(
        data.features.element_size(), conv.weight.element_size())
    limit = (_INT32_MAX - 1) // bytes_per_row
    if max_tile_points is not None:
        if not isinstance(max_tile_points, int) or max_tile_points <= 0:
            raise ValueError('max_tile_points must be a positive integer')
        limit = min(limit, max_tile_points)
    if limit < 1:
        raise RuntimeError('A single SubM feature row exceeds the int32 byte span')

    coords = data.indices[:, 1:]
    radius = coords.new_tensor([d * (k // 2) for d, k in zip(conv.dilation, conv.kernel_size)])
    pending = [torch.arange(coords.shape[0], device=coords.device)]
    stitched = None
    tile_count, largest_halo = 0, 0
    while pending:
        core = pending.pop()
        core_coords = coords[core]
        lower, upper = core_coords.amin(0), core_coords.amax(0)
        # Always search the full input, including neighbors outside parent tiles.
        mask = torch.ones(coords.shape[0], dtype=torch.bool, device=coords.device)
        for axis in range(3):
            mask &= (coords[:, axis] >= lower[axis] - radius[axis]) & (coords[:, axis] <= upper[axis] + radius[axis])
        halo = torch.nonzero(mask, as_tuple=False).flatten()  # Sorted original row IDs.
        if halo.numel() > limit:
            axis = int((upper - lower).argmax().item())
            if int((upper[axis] - lower[axis]).item()) == 0:
                raise RuntimeError(f'Cannot split SubM core: singleton-coordinate halo has {halo.numel()} rows, limit {limit}')
            midpoint = lower[axis] + (upper[axis] - lower[axis]) // 2
            left = core_coords[:, axis] <= midpoint
            # Actual bbox extrema straddle the midpoint, so both cores shrink.
            pending.extend([core[~left], core[left]])
            continue

        tile = spconv.SparseConvTensor(data.features[halo], data.indices[halo],
                                       data.spatial_shape, data.batch_size, indice_dict={})
        tile_out = conv(tile)  # Keep the original weights, bias, training flag, and algo.
        if not torch.equal(tile_out.indices, tile.indices):
            raise RuntimeError('SubM tile output indices do not preserve input row order')
        positions = torch.searchsorted(halo, core)
        if not torch.equal(halo[positions], core):
            raise RuntimeError('SubM core row is missing from its full-input halo')
        if stitched is None:
            stitched = tile_out.features.new_empty((coords.shape[0], tile_out.features.shape[1]))
        stitched[core] = tile_out.features[positions]
        del tile, tile_out  # Release this tile's features and indice cache before the next convolution.
        tile_count += 1
        largest_halo = max(largest_halo, halo.numel())
    if not getattr(conv, '_trellis_subm_tiling_logged', False):
        print(f'[SPCONV] SubM spatial tiling layer={id(conv):x} rows={coords.shape[0]} '
              f'Cin={conv.in_channels} Cout={conv.out_channels} tiles={tile_count} '
              f'max_halo_rows={largest_halo} limit={limit}', flush=True)
        conv._trellis_subm_tiling_logged = True
    return data.replace_feature(stitched)


class SparseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
        super(SparseConv3d, self).__init__()
        if 'spconv' not in globals():
            import spconv.pytorch as spconv
        algo = None
        if SPCONV_ALGO == 'native':
            algo = spconv.ConvAlgo.Native
        elif SPCONV_ALGO == 'implicit_gemm':
            algo = spconv.ConvAlgo.MaskImplicitGemm
        if stride == 1 and (padding is None):
            self.conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, dilation=dilation, bias=bias, indice_key=indice_key, algo=algo)
        else:
            self.conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias, indice_key=indice_key, algo=algo)
        self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, stride, stride)
        self.padding = padding

    def forward(self, x: SparseTensor) -> SparseTensor:
        spatial_changed = any(s != 1 for s in self.stride) or (self.padding is not None)
        if (not torch.is_grad_enabled() and _subm_byte_span(self.conv, x.data) >= _INT32_MAX
                and _can_tile_subm(self.conv, x.data)):
            new_data = _subm_conv_core_halo(self.conv, x.data)
        else:
            new_data = self.conv(x.data)
        new_shape = [x.shape[0], self.conv.out_channels]
        new_layout = None if spatial_changed else x.layout

        if spatial_changed and (x.shape[0] != 1):
            import spconv.pytorch as spconv

            # spconv was non-1 stride will break the contiguous of the output tensor, sort by the coords
            fwd = new_data.indices[:, 0].argsort()
            bwd = torch.zeros_like(fwd).scatter_(0, fwd, torch.arange(fwd.shape[0], device=fwd.device))
            sorted_feats = new_data.features[fwd]
            sorted_coords = new_data.indices[fwd]
            unsorted_data = new_data
            new_data = spconv.SparseConvTensor(sorted_feats, sorted_coords, unsorted_data.spatial_shape, unsorted_data.batch_size)  # type: ignore

        out = SparseTensor(
            new_data, shape=torch.Size(new_shape), layout=new_layout,
            scale=tuple([s * stride for s, stride in zip(x._scale, self.stride)]),
            spatial_cache=x._spatial_cache,
        )

        if spatial_changed and (x.shape[0] != 1):
            out.register_spatial_cache(f'conv_{self.stride}_unsorted_data', unsorted_data)
            out.register_spatial_cache(f'conv_{self.stride}_sort_bwd', bwd)
 
        return out


class SparseInverseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None):
        super(SparseInverseConv3d, self).__init__()
        if 'spconv' not in globals():
            import spconv.pytorch as spconv
        self.conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, bias=bias, indice_key=indice_key)
        self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, stride, stride)

    def forward(self, x: SparseTensor) -> SparseTensor:
        spatial_changed = any(s != 1 for s in self.stride)
        if spatial_changed:
            # recover the original spconv order
            data = x.get_spatial_cache(f'conv_{self.stride}_unsorted_data')
            bwd = x.get_spatial_cache(f'conv_{self.stride}_sort_bwd')
            data = data.replace_feature(x.feats[bwd])
            if DEBUG:
                assert torch.equal(data.indices, x.coords[bwd]), 'Recover the original order failed'
        else:
            data = x.data

        new_data = self.conv(data)
        new_shape = [x.shape[0], self.conv.out_channels]
        new_layout = None if spatial_changed else x.layout
        out = SparseTensor(
            new_data, shape=torch.Size(new_shape), layout=new_layout,
            scale=tuple([s // stride for s, stride in zip(x._scale, self.stride)]),
            spatial_cache=x._spatial_cache,
        )
        return out
