from typing import *
from abc import abstractmethod
import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from ..utils.pose_utils import camera_parameters_from_frame_data
from ..utils.voxel_utils import decode_depth_map


class StandardDatasetBase(Dataset):
    """
    Base class for standard datasets.

    Args:
        roots (str): paths to the dataset
        min_aesthetic_score (float): minimum aesthetic score for the dataset
        is_eval (bool): whether to evaluate all the views of the dataset
        max_num_retry (int): maximum number of retries for each instance when
            exception occurs, for example, no valid coords for instance and view
        instances_sha256 (List[str]): list of sha256 of the instances to be included in the dataset
    """

    def __init__(self,
        roots: str,
        min_aesthetic_score: float = 5.0,
        is_eval: bool = False,
        max_num_retry: int = 10,
        instances_sha256: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__()
        self.roots = roots.split(',')
        self.instances = []
        self.metadata = pd.DataFrame()
        self.min_aesthetic_score = min_aesthetic_score
        self.is_eval = is_eval
        self.max_num_retry = max_num_retry
        self.instances_sha256 = instances_sha256
        
        self._stats = {}
        for root in self.roots:
            key = os.path.basename(root)
            self._stats[key] = {}
            metadata = pd.read_csv(os.path.join(root, 'metadata.csv'))
            self._stats[key]['Total'] = len(metadata)
            metadata, stats = self.filter_metadata(metadata)
            self._stats[key].update(stats)
            self.instances.extend([(root, sha256) for sha256 in metadata['sha256'].values])
            metadata.set_index('sha256', inplace=True)
            self.metadata = pd.concat([self.metadata, metadata])

        # filter instances by sha256
        if self.instances_sha256 is not None:
            self.instances = [instance for instance in self.instances if instance[1] in self.instances_sha256]
            
    def filter_metadata(self, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        stats = {}
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root: str, instance: str, **kwargs) -> Dict[str, Any]:
        return {}
        
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index) -> Dict[str, Any]:
        num_retry = 0
        while True:
            try:
                root, instance = self.instances[index]
                return self.get_instance(root, instance)
            except Exception as e:
                print(f"Error getting data instance {index}: {e}")
                num_retry += 1
                if num_retry >= self.max_num_retry:
                    raise e
                index = np.random.randint(0, len(self))
        
    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Total instances: {len(self)}')
        lines.append(f'  - Sources:')
        for key, stats in self._stats.items():
            lines.append(f'    - {key}:')
            for k, v in stats.items():
                lines.append(f'      - {k}: {v}')
        return '\n'.join(lines)


class ImageConditionedMixin:
    def __init__(
        self, 
        roots, 
        *, 
        image_size : int = 518, 
        crop_size_ratio : Optional[Union[float, Tuple[float, float]]] = None,
        crop_prob : float = 1.0,
        crop_in_eval : bool = True,
        crop_num_discretization_buckets: Optional[int] = None,
        crop_allow_padding: bool = False,
        return_depth_map: bool = False,
        return_raw_cond: bool = False,
        return_rgba_cond: bool = False,
        random_bg_color: bool = False,
        **kwargs
    ):
        self.image_size = image_size
        self.crop_size_ratio = crop_size_ratio
        self.crop_prob = crop_prob
        self.crop_in_eval = crop_in_eval
        self.crop_num_discretization_buckets = crop_num_discretization_buckets
        self.crop_allow_padding = crop_allow_padding
        self.override_buekct_idx = None
        self.last_crop_ratio = None
        self.return_depth_map = return_depth_map
        self.return_raw_cond = return_raw_cond
        self.return_rgba_cond = return_rgba_cond
        self.random_bg_color = random_bg_color
        super().__init__(roots, **kwargs)
    
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata[f'cond_rendered']]
        stats['Cond rendered'] = len(metadata)
        if 'has_missing_resources' in metadata.columns:
            metadata = metadata[~metadata['has_missing_resources'].fillna(False).astype(bool)]
            stats['Resources present'] = len(metadata)
        return metadata, stats

    def get_crop_bbox(self, image: Image.Image) -> Tuple[int, int, int, int]:
        if isinstance(self.crop_size_ratio, float):
            ratio = self.crop_size_ratio
        elif isinstance(self.crop_size_ratio, (tuple, list)):
            min_ratio, max_ratio = self.crop_size_ratio[:2]
            if self.is_eval:  # Decay to midpoint for eval mode
                ratio = (min_ratio + max_ratio) / 2
            elif len(self.crop_size_ratio) == 2:  # Uniform distribution
                ratio = np.random.uniform(min_ratio, max_ratio)
            elif len(self.crop_size_ratio) == 3:  # Beta distribution
                k = self.crop_size_ratio[2]
                ratio = min_ratio + (max_ratio - min_ratio) * np.random.beta(k, k)
            else:
                raise ValueError(f'Invalid crop_size_ratio tuple len: {self.crop_size_ratio}')

            if self.crop_num_discretization_buckets is not None:
                if self.override_buekct_idx is not None:
                    bucket_idx = self.override_buekct_idx
                else:
                    bucket_t = (ratio - min_ratio) / (max_ratio - min_ratio)
                    bucket_idx = int(bucket_t * self.crop_num_discretization_buckets)
                    bucket_idx = min(max(bucket_idx, 0), self.crop_num_discretization_buckets - 1)
                bucket_t = (bucket_idx + 0.5) / self.crop_num_discretization_buckets
                ratio = min_ratio + (max_ratio - min_ratio) * bucket_t
        else:
            raise ValueError(f'Invalid crop_size_ratio: {self.crop_size_ratio}')
        self.last_crop_ratio = ratio

        assert image.mode == 'RGBA', 'Image must be in RGBA mode for cropping to work'
        width, height = image.size
        ys, xs = np.array(image.getchannel(3)).nonzero()
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()), int(ys.max())

        cx = int(round((x0 + (x1 + 1)) / 2))
        cy = int(round((y0 + (y1 + 1)) / 2))
        obj_hsize = 0.5 * max(x1 - x0 + 1, y1 - y0 + 1)
        aug_hsize = int(round(obj_hsize * ratio))
        if not self.crop_allow_padding:
            max_hsize = min(cx, cy, width - cx, height - cy)
            aug_hsize = min(aug_hsize, max_hsize)
        aug_size = aug_hsize * 2

        if not self.crop_allow_padding:
            left = max(0, min(cx - aug_hsize, width - aug_size))
            top  = max(0, min(cy - aug_hsize, height - aug_size))
        else:
            left = cx - aug_hsize
            top  = cy - aug_hsize
        right  = left + aug_size
        bottom = top + aug_size
        
        return (left, top, right, bottom)
    
    def crop_image(self, image: Image.Image, depth_image: Optional[Image.Image], pack: Dict[str, torch.Tensor]) -> Tuple[Image.Image, Image.Image]:
        image_size = torch.tensor(image.size, dtype=torch.float32)
        crop_bbox = self.get_crop_bbox(image)
        image = image.crop(crop_bbox)
        if self.return_depth_map:
           depth_image = depth_image.crop(crop_bbox)
        crop_bbox = torch.tensor(crop_bbox, dtype=torch.float32).reshape(2, 2)
        crop_bbox = crop_bbox / image_size

        if 'uv_volume' in pack:
            bbox_min = crop_bbox[0][:, None, None, None]
            bbox_max = crop_bbox[1][:, None, None, None]
            crop_uvs = (pack['uv_volume'] - bbox_min) / (bbox_max - bbox_min)
            pack['uv_volume'] = crop_uvs.clamp_(0.0, 1.0)

        if 'uvs' in pack:
            bbox_min = crop_bbox[0][None, :]
            bbox_max = crop_bbox[1][None, :]
            crop_uvs = (pack['uvs'] - bbox_min) / (bbox_max - bbox_min)
            pack['uvs'] = crop_uvs.clamp_(0.0, 1.0)

        if 'intrinsics' in pack:
            intrinsics = pack.pop('intrinsics')
            if self.return_raw_cond:
                pack['raw_intrinsics'] = intrinsics.clone()
            scale = crop_bbox[1] - crop_bbox[0]
            intrinsics[0,0] = intrinsics[0,0] / scale[0]
            intrinsics[0,2] = (intrinsics[0,2] - crop_bbox[0, 0]) / scale[0]
            intrinsics[1,1] = intrinsics[1,1] / scale[1]
            intrinsics[1,2] = (intrinsics[1,2] - crop_bbox[0, 1]) / scale[1]
            pack['intrinsics'] = intrinsics

        pack['crop_bbox'] = crop_bbox

        return image, depth_image

    def get_image_cond(self, image: Image.Image, bg: Optional[torch.Tensor] = None) -> torch.Tensor:
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        alpha = alpha.unsqueeze(0)
        if bg is None:
            image = image * alpha
        else:
            image = image * alpha + bg * (1 - alpha)
        return image

    def get_rgba_cond(self, image: Image.Image) -> torch.Tensor:
        """Resize to image_size, return un-premultiplied [4, H, W] RGBA in [0, 1]."""
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        return torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0

    def get_depth_cond(self, depth_image: Image.Image, frame_data: Dict[str, Any]) -> torch.Tensor:
        depth_map = torch.from_numpy(decode_depth_map(depth_image, frame_data))
        depth_map = torch.nn.functional.interpolate(
            depth_map[None, None, :, :], 
            size=(self.image_size, self.image_size), 
            mode='bilinear', 
            align_corners=False, 
            antialias=True,
        )[0, 0]
        depth_mask = depth_map > 0
        depth_map[~depth_mask] = 0
        return depth_map

    def get_instance(self, root, instance, frame_data=None, **kwargs):
        image_root = os.path.join(root, 'renders_cond', instance)
        if frame_data is None:
            with open(os.path.join(image_root, 'transforms.json')) as f:
                metadata = json.load(f)
                view_idx = np.random.randint(len(metadata['frames']))
                frame_data = metadata['frames'][view_idx]
        if 'render_dir' in frame_data:
            image_root = frame_data['render_dir']

        pack = super().get_instance(root, instance, frame_data=frame_data, **kwargs)
        
        image_path = os.path.join(image_root, frame_data['file_path'])
        image = Image.open(image_path)
        depth_image = None
        if self.return_depth_map:
            image_path_base = os.path.splitext(frame_data["file_path"])[0]
            depth_image_path = os.path.join(image_root, f'{image_path_base}_depth.png')
            depth_image = Image.open(depth_image_path)

        bg = torch.rand(3, 1, 1) if (self.random_bg_color and not self.is_eval) else None

        if self.return_raw_cond:
            pack['raw_cond'] = self.get_image_cond(image, bg=bg)
            if self.return_rgba_cond:
                pack['raw_cond_rgba'] = self.get_rgba_cond(image)
            if depth_image is not None:
                pack['raw_cond_depth'] = self.get_depth_cond(depth_image, frame_data)
            if 'intrinsics' in pack:
                pack['raw_intrinsics'] = pack['intrinsics'].clone()

        if self.crop_size_ratio is not None and \
            (not self.is_eval or self.crop_in_eval) and \
            (self.is_eval or np.random.rand() < self.crop_prob):
            image, depth_image = self.crop_image(image, depth_image, pack)
        else:
            self.last_crop_ratio = None

        pack['cond'] = self.get_image_cond(image, bg=bg)
        if self.return_rgba_cond:
            pack['cond_rgba'] = self.get_rgba_cond(image)
        if depth_image is not None:
            pack['cond_depth'] = self.get_depth_cond(depth_image, frame_data)
       
        return pack
    

class TransformConditionedMixin:
    """
    Mixin for datasets that are conditioned on camera transforms.
    
    Args:
        num_ref_views (int): number of reference views to use per instance
        num_cond_views (int): number of conditional views to use per instance
        eval_view_sub_sample (int): split number of views to subsample for evaluation
        eval_view_sub_sample_offset (int): offset for evaluation view subsampling
    """
    def __init__(self, 
        roots, 
        *, 
        num_ref_views: int = 0, 
        num_cond_views: int = 24, 
        eval_view_sub_sample: int = 1,
        eval_view_sub_sample_offset: int = 0,
        **kwargs
    ):
        self.num_ref_views = num_ref_views
        self.num_cond_views = num_cond_views
        self.eval_view_sub_sample = eval_view_sub_sample
        self.eval_view_sub_sample_offset = eval_view_sub_sample_offset
        self.eval_total_views = (self.num_ref_views + self.num_cond_views) // self.eval_view_sub_sample
        super().__init__(roots, **kwargs)

    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        if self.num_ref_views > 0:
            metadata = metadata[metadata[f'rendered']]
            stats['Rendered'] = len(metadata)
        if self.num_cond_views > 0:
            metadata = metadata[metadata[f'cond_rendered']]
            stats['Cond rendered'] = len(metadata)
        return metadata, stats

    def _get_frame_data(self, root, instance, view_idx):
        if view_idx < self.num_cond_views:
            with open(os.path.join(root, 'renders_cond', instance, 'transforms.json')) as f:
                metadata = json.load(f)
            frame_data = metadata['frames'][view_idx]
            frame_data['render_dir'] = os.path.join(root, 'renders_cond', instance)
        else:
            with open(os.path.join(root, 'renders', instance, 'transforms.json')) as f:
                metadata = json.load(f)
            frame_data = metadata['frames'][view_idx - self.num_cond_views]
            frame_data['render_dir'] = os.path.join(root, 'renders', instance)
        return frame_data

    def get_instance(self, root, instance, frame_data, **kwargs):
        """Get base data, and camera transforms for an instance."""
        pack = super().get_instance(root, instance, frame_data=frame_data, **kwargs)
        extrinsic, intrinsic = camera_parameters_from_frame_data(frame_data)
        pack['intrinsics'] = intrinsic
        pack['extrinsics'] = extrinsic
        return pack

    def get_instance_view_index(self, index):
        assert self.is_eval, "get_instance_view_index should be only used in eval mode"
        instance_idx, view_idx = divmod(index, self.eval_total_views)
        view_idx = view_idx * self.eval_view_sub_sample + self.eval_view_sub_sample_offset
        return instance_idx, view_idx

    def __getitem__(self, index):
        if self.is_eval:
            instance_idx, view_idx = self.get_instance_view_index(index)
            root, instance = self.instances[instance_idx]
            frame_data = self._get_frame_data(root, instance, view_idx)
            return self.get_instance(root, instance, frame_data=frame_data)
        else:
            num_retry = 0
            while True:
                try:
                    root, instance = self.instances[index]
                    view_idx = np.random.randint(self.num_ref_views + self.num_cond_views)
                    frame_data = self._get_frame_data(root, instance, view_idx=view_idx)
                    return self.get_instance(root, instance, frame_data=frame_data)
                except Exception as e:
                    print(f"Error getting data instance {index}: {e}")
                    num_retry += 1
                    if num_retry >= self.max_num_retry:
                        raise e
                    index = np.random.randint(0, len(self))

    def __len__(self):
        if self.is_eval:
            return len(self.instances) * self.eval_total_views
        else:
            return len(self.instances)


_DEFAULT_EVAL_VIEWSETS_K4: Tuple[Tuple[int, ...], ...] = (
    (0, 6, 12, 18),
    (3, 9, 15, 21),
)

# Protocol default test views (docs/evaluation_protocol.md §3, T set).
# Disjoint from _DEFAULT_EVAL_VIEWSETS_K4 (V0 ∪ V1 = {0,3,6,9,12,15,18,21}).
_DEFAULT_EVAL_TEST_VIEWS_K4: Tuple[int, ...] = (2, 4, 8, 10, 14, 16, 20, 22)


class MultiViewMixin:
    """
    Mixin that samples V views per instance for multi-view training/eval.

    Insert before TransformConditionedMixin-based classes in MRO.
    At num_pose_views=1, falls through to single-view behavior.

    Args:
        num_pose_views: number of views to sample per instance (default 1)
        view_selection: "random" or "stratified" (training only; eval is fixed)
        eval_viewsets: explicit per-sample view-index tuples for eval. Each
            entry must have length ``num_pose_views`` and indices in
            ``[0, num_ref_views + num_cond_views)``. If ``None`` and
            ``num_pose_views == 4``, defaults to the protocol's
            ``V0={0,6,12,18}`` and ``V1={3,9,15,21}``
            (see ``docs/evaluation_protocol.md`` §3). For other
            ``num_pose_views``, must be supplied explicitly when used in eval.
        eval_test_views: explicit test-view indices to gather as ground truth
            in eval mode. When non-empty (and ``is_eval=True``), each per-view
            tensor from those views is attached to the pack under a ``test_``
            prefix. Indices must be unique, in range, and disjoint from every
            entry of ``eval_viewsets``. If ``None`` and ``num_pose_views == 4``,
            defaults to the protocol's ``T={2,4,8,10,14,16,20,22}``
            (see ``docs/evaluation_protocol.md`` §3) — disjoint from the
            default ``eval_viewsets``. Pass an empty list to disable explicitly.

    Subclass hook:
        ``_SHARED_KEYS``: class-level tuple of pack keys that are *shared* across
        views (taken from ``pack[0]``) rather than stacked. Extend this tuple in
        subclasses whose packs carry additional view-invariant fields
        (e.g. SLat's ``coords``/``feats``).

        ``_SKIP_TEST_KEYS``: class-level tuple of pack keys to *omit* from the
        ``test_*`` gather. Use for per-voxel quantities whose voxel-axis size
        varies per sample (e.g. ``uvs``), which would otherwise break the batch
        collate's naive ``torch.stack``. Declare on the data-side base (e.g.
        ``SlatWithUVTransforms``); intentionally not defined here so MRO
        attribute lookup can fall through to that base in subclasses like
        ``class X(MultiViewMixin, SlatWithUVTransforms)``.
    """

    _SHARED_KEYS: Tuple[str, ...] = ('x_0', 'ss')

    def __init__(self,
        *args,
        num_pose_views: int = 1,
        view_selection: Literal["random", "stratified"] = "random",
        eval_viewsets: Optional[Sequence[Sequence[int]]] = None,
        eval_test_views: Optional[Sequence[int]] = None,
        **kwargs
    ):
        kwargs.setdefault('seperate_pose_latent', True)
        super().__init__(*args, **kwargs)
        self.num_pose_views = num_pose_views
        self.view_selection = view_selection
        if num_pose_views > 1 and not getattr(self, 'seperate_pose_latent', True):
            raise ValueError("num_pose_views > 1 requires seperate_pose_latent=True")

        if eval_viewsets is None and num_pose_views == 4:
            eval_viewsets = _DEFAULT_EVAL_VIEWSETS_K4
        if eval_viewsets is not None:
            total_views = self.num_ref_views + self.num_cond_views
            normalized: List[Tuple[int, ...]] = []
            for vs in eval_viewsets:
                vs_t = tuple(int(v) for v in vs)
                if len(vs_t) != num_pose_views:
                    raise ValueError(f"eval_viewsets entry {vs_t} has length {len(vs_t)}, expected num_pose_views={num_pose_views}")
                if len(set(vs_t)) != len(vs_t):
                    raise ValueError(f"eval_viewsets entry {vs_t} has duplicate indices")
                for v in vs_t:
                    if not 0 <= v < total_views:
                        raise ValueError(f"eval_viewsets index {v} out of range [0, {total_views})")
                normalized.append(vs_t)
            eval_viewsets = tuple(normalized)
        elif num_pose_views > 1 and getattr(self, 'is_eval', False):
            raise ValueError(
                f"eval_viewsets must be provided when is_eval=True and num_pose_views={num_pose_views}"
                f" (built-in default is only defined for num_pose_views=4)"
            )
        self.eval_viewsets: Optional[Tuple[Tuple[int, ...], ...]] = eval_viewsets
        self.eval_num_viewsets = len(eval_viewsets) if eval_viewsets is not None else 1

        if eval_test_views is None and num_pose_views == 4:
            eval_test_views = _DEFAULT_EVAL_TEST_VIEWS_K4
        if eval_test_views:
            total_views = self.num_ref_views + self.num_cond_views
            tv_t = tuple(int(v) for v in eval_test_views)
            if len(set(tv_t)) != len(tv_t):
                raise ValueError(f"eval_test_views {tv_t} has duplicate indices")
            for v in tv_t:
                if not 0 <= v < total_views:
                    raise ValueError(f"eval_test_views index {v} out of range [0, {total_views})")
            if eval_viewsets is not None:
                tv_set = set(tv_t)
                for vs in eval_viewsets:
                    overlap = tv_set & set(vs)
                    if overlap:
                        raise ValueError(
                            f"eval_test_views {tv_t} overlaps eval_viewsets entry {vs} at indices {sorted(overlap)};"
                            f" test partition T must be disjoint from input viewset I"
                        )
            self.eval_test_views: Tuple[int, ...] = tv_t
        else:
            self.eval_test_views = ()

    def _select_train_views(self, total_views, num_views: Optional[int] = None) -> List[int]:
        n = self.num_pose_views if num_views is None else num_views
        if self.view_selection == "stratified":
            bucket_size = total_views // n
            return [np.random.randint(i * bucket_size, (i + 1) * bucket_size if i < n - 1 else total_views) for i in range(n)]
        else:
            return np.random.choice(total_views, n, replace=False).tolist()

    def _select_eval_views(self, viewset_idx: int = 0) -> List[int]:
        if self.eval_viewsets is None:
            raise RuntimeError(
                "eval_viewsets is not configured; pass it to MultiViewMixin or"
                " use num_pose_views=4 (which defaults to the protocol viewsets)."
            )
        return list(self.eval_viewsets[viewset_idx])

    def _gather_test_views(self, root, instance) -> Dict[str, Any]:
        """Load GT data at ``self.eval_test_views`` and return a dict with
        ``test_`` prefix on every per-view key. Shared (latent) keys are
        dropped — only per-view tensors/lists survive."""
        assert self.eval_test_views, "_gather_test_views requires non-empty eval_test_views"
        test_packs = []
        for vi in self.eval_test_views:
            frame_data = self._get_frame_data(root, instance, vi)
            test_packs.append(self.get_instance(root, instance, frame_data=frame_data))
        merged = self._merge_packs(test_packs)
        skip_test_keys = getattr(self, '_SKIP_TEST_KEYS', ())
        out = {f'test_{k}': v for k, v in merged.items()
               if k not in self._SHARED_KEYS and k not in skip_test_keys}
        out['test_view_indices'] = torch.tensor(self.eval_test_views, dtype=torch.long)
        return out

    def _merge_packs(self, packs):
        merged = {}
        for k, v in packs[0].items():
            if k in self._SHARED_KEYS:
                merged[k] = v
            elif isinstance(v, torch.Tensor):
                merged[k] = torch.stack([p[k] for p in packs])
            else:
                merged[k] = [p[k] for p in packs]
        return merged

    def __getitem__(self, index):
        total_views = self.num_ref_views + self.num_cond_views

        # Train path supports (idx, V) tuples from VarViewBatchSampler to override
        # num_pose_views per call; eval path always uses self.num_pose_views.
        train_view_override: Optional[int] = None
        if isinstance(index, tuple):
            index, train_view_override = index
            assert train_view_override <= total_views, (
                f"V={train_view_override} exceeds total_views={total_views}"
            )

        if self.is_eval:
            if self.num_pose_views == 1:
                instance_idx, view_idx = super().get_instance_view_index(index)
                view_indices = [view_idx]
            else:
                instance_idx, viewset_idx = divmod(index, self.eval_num_viewsets)
                view_indices = self._select_eval_views(viewset_idx)
        else:
            instance_idx = index
            view_indices = self._select_train_views(total_views, train_view_override)

        num_retry = 0
        while True:
            try:
                root, instance = self.instances[instance_idx]
                packs = []
                for vi in view_indices:
                    frame_data = self._get_frame_data(root, instance, vi)
                    pack = self.get_instance(root, instance, frame_data=frame_data)
                    packs.append(pack)
                merged = self._merge_packs(packs)
                if self.is_eval:
                    merged['view_indices'] = torch.tensor(view_indices, dtype=torch.long)
                    if self.eval_test_views:
                        merged.update(self._gather_test_views(root, instance))
                return merged
            except Exception as e:
                print(f"Error getting multi-view data instance {instance_idx}: {e}")
                num_retry += 1
                if num_retry >= self.max_num_retry:
                    raise e
                instance_idx = np.random.randint(0, len(self.instances))

    def get_instance_view_index(self, index):
        if self.num_pose_views == 1:
            return super().get_instance_view_index(index)
        assert self.is_eval, "get_instance_view_index should only be used in eval mode"
        instance_idx, viewset_idx = divmod(index, self.eval_num_viewsets)
        return instance_idx, viewset_idx

    def __len__(self):
        if self.num_pose_views == 1:
            return super().__len__()
        if self.is_eval:
            return len(self.instances) * self.eval_num_viewsets
        return super().__len__()
