"""Native joint shape/camera and appearance inference for GenSfM bundles."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import random
from tempfile import TemporaryDirectory


IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.tif', '.tiff'}
MODEL_ROLES = ('denoiser', 'slat_denoiser', 'ss_decoder', 'suv_decoder',
               'slat_decoder_gs', 'slat_decoder_mesh')
NATIVE_INFERENCE_OPTIONS = dict(
    batch_size_per_gpu=1, num_workers=0, max_num_samples=1,
    sample_ssuv=True, use_gt_intrinsics=False, alignment_method='none',
    evaluate_pose=False, evaluate_3d_geometry=False, evaluate_novel_view=False,
    evaluate_cross_view=False, evaluate_clip_metric=False, save_debug=False,
    eval_intermediate_steps=False, save_results=False, save_mesh=False,
    save_pose=False, image_cond_model_compile=False,
)


def collect_images(images=None, input_dir=None):
    if input_dir is not None:
        folder = Path(input_dir).expanduser().resolve()
        if not folder.is_dir():
            raise ValueError(f'Input directory does not exist: {folder}')
        images = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    paths = [Path(p).expanduser().resolve() for p in (images or [])]
    if len(paths) < 2:
        raise ValueError('Provide at least two views of the same object; four are recommended.')
    if len(set(paths)) != len(paths):
        raise ValueError('Input views must be distinct files.')
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f'Not a supported image file: {path}')
    return paths


class InputViews:
    """One image-only input; no benchmark dataset, GT poses or GT geometry."""
    is_eval = True
    image_size = 518
    crop_allow_padding = False
    crop_num_discretization_buckets = None

    def __init__(self, paths, crop_ratio=1.2, remove_background=False):
        import numpy as np
        import torch
        from PIL import Image, ImageOps
        from torch.utils.data import default_collate
        from trellis.datasets.components import ImageConditionedMixin

        self.crop_size_ratio = crop_ratio
        self.collate_fn = default_collate
        self.instances = [('', 'input')]
        self.records, self.rgba, conditions = [], [], []
        session = None
        for path in paths:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert('RGBA')
            alpha = np.asarray(image.getchannel('A'))
            if np.all(alpha == 255):
                if not remove_background:
                    raise ValueError(f'Foreground alpha is required: {path}. Supply RGBA masks or use --remove-background.')
                import rembg
                if session is None:
                    session = rembg.new_session('u2net')
                image = rembg.remove(image, session=session).convert('RGBA')
                alpha = np.asarray(image.getchannel('A'))
            if not np.any(alpha):
                raise ValueError(f'Empty foreground mask: {path}')
            bbox = ImageConditionedMixin.get_crop_bbox(self, image)
            cropped = image.crop(bbox)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1] or np.count_nonzero(np.asarray(cropped.getchannel('A'))) != np.count_nonzero(alpha):
                raise ValueError(f'Training crop would truncate the foreground: {path}. Add transparent padding around the object.')
            self.rgba.append(ImageConditionedMixin.get_rgba_cond(self, cropped))
            conditions.append(ImageConditionedMixin.get_image_cond(self, cropped))
            self.records.append({'path': str(path), 'source_size': list(image.size), 'crop_pixels': list(bbox),
                                 'crop_normalized': (np.array(bbox).reshape(2, 2) / np.array(image.size)).tolist()})
        self.condition = torch.stack(conditions)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {'cond': self.condition}


def source_intrinsics(intrinsics, records):
    """Map complete normalized crop intrinsics back to EXIF-oriented source images."""
    import numpy as np
    result = []
    for matrix, record in zip(intrinsics, records):
        (left, top), (right, bottom) = record['crop_normalized']
        mapping = np.array([[right-left, 0, left], [0, bottom-top, top], [0, 0, 1]])
        result.append((mapping @ matrix).tolist())
    return result


def reconstruct(evaluator, inputs, seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    condition = inputs.condition[None].to(evaluator.device)
    with torch.no_grad():
        print('Stage 1: joint shape and camera generation', flush=True)
        ssuv, _ = evaluator._sample_ssuv(cond=condition)
        extrinsics, intrinsics, _ = evaluator._solve_pose_from_ssuv(
            ssuv['ssuv_recon'].flatten(0, 1), ssuv['uv_volume_pred'].flatten(0, 1))
        if not torch.isfinite(extrinsics).all() or not torch.isfinite(intrinsics).all():
            raise RuntimeError('Non-finite predicted camera parameters')
        if not (intrinsics[:, 0, 0] > 0).all() or not (intrinsics[:, 1, 1] > 0).all():
            raise RuntimeError('Predicted focal lengths must be positive')
        noise = evaluator._make_randn_sparse_latent_from_ss(ssuv['ss_recon'])
        if len(noise.coords) == 0:
            raise RuntimeError('The model predicted no occupied voxels')
        uvs = evaluator._make_sparse_uvs_from_pose(noise, extrinsics[None], intrinsics[None])
        print('Stage 2: appearance and detailed geometry generation', flush=True)
        outputs = evaluator._sample_slat(cond=condition, noise=noise, uvs=uvs)
    return outputs, extrinsics, intrinsics


class GenSfMPipeline:
    """A lazy native adapter with six evaluator roles and a bundled SLat encoder."""

    def __init__(self, model_cfgs, inference, preprocessing):
        self.model_cfgs = copy.deepcopy(model_cfgs)
        self.inference = copy.deepcopy(inference)
        self.preprocessing = copy.deepcopy(preprocessing)
        self.device = 'cuda'

    @classmethod
    def from_pretrained(cls, path, revision=None):
        """Read a local bundle or download one HF snapshot; defer model loading."""
        directory = Path(path).expanduser()
        if directory.is_dir():
            directory = directory.resolve()
        elif isinstance(path, Path) or directory.is_absolute() or str(path).startswith(('./', '../', '~')):
            raise ValueError(f'Local bundle directory does not exist: {directory}')
        else:
            from huggingface_hub import snapshot_download
            directory = Path(snapshot_download(
                str(path), revision=revision,
                allow_patterns=['pipeline.json', 'ckpts/*.json', 'ckpts/*.safetensors'],
            )).resolve()
        manifest = directory / 'pipeline.json'
        if not manifest.is_file():
            raise ValueError(f'Missing bundle manifest: {manifest}')
        config = json.loads(manifest.read_text())
        if not isinstance(config, dict) or config.get('name') != 'GenSfMPipeline':
            raise ValueError('The bundle must declare name="GenSfMPipeline" in pipeline.json.')
        args = config.get('args', {})
        if not isinstance(args, dict):
            raise ValueError('The bundle must provide an args object.')
        models = args.get('models', {})
        if not isinstance(models, dict) or set(models) != set(MODEL_ROLES):
            raise ValueError(f'The bundle must contain exactly these model roles: {", ".join(MODEL_ROLES)}')
        inference = args.get('inference')
        preprocessing = args.get('preprocessing')
        if not isinstance(inference, dict) or not isinstance(preprocessing, dict):
            raise ValueError('The bundle must provide args.inference and args.preprocessing objects.')
        if not isinstance(inference.get('normalization'), dict):
            raise ValueError('args.inference.normalization must contain the stage-2 latent statistics.')
        for key in ('ssuv_sampler_args', 'slat_sampler_args'):
            sampler = inference.get(key)
            if not isinstance(sampler, dict) or not isinstance(sampler.get('params'), dict):
                raise ValueError(f'args.inference.{key} must provide sampler parameters.')
            steps = sampler['params'].get('steps')
            if not isinstance(steps, int) or steps < 1:
                raise ValueError(f'args.inference.{key}.params.steps must be a positive integer.')
        if preprocessing.get('image_size') != 518:
            raise ValueError('Native GenSfM inputs require preprocessing.image_size=518.')
        ratio = preprocessing.get('crop_size_ratio')
        limit = preprocessing.get('max_views')
        if not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio <= 0:
            raise ValueError('preprocessing.crop_size_ratio must be positive and finite.')
        if not isinstance(limit, int) or limit < 2:
            raise ValueError('preprocessing.max_views must be an integer of at least two.')

        model_cfgs = {}
        for role in MODEL_ROLES:
            value = models[role]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'Model prefix for {role} must be a nonempty relative path.')
            prefix = Path(value)
            if prefix.is_absolute() or '..' in prefix.parts or not prefix.parts or prefix.parts[0] != 'ckpts':
                raise ValueError(f'Model prefix for {role} must be relative to the bundle under ckpts/: {value}')
            prefix = directory / prefix
            for suffix in ('.json', '.safetensors'):
                if not Path(f'{prefix}{suffix}').is_file():
                    raise ValueError(f'Missing bundle component: {prefix}{suffix}')
            model_cfgs[role] = str(prefix)
        stage2 = json.loads(Path(f'{model_cfgs["slat_denoiser"]}.json').read_text())
        stage2_args = stage2.get('args', {})
        dependency = stage2_args.get('pretrained_slat_enc')
        if not isinstance(dependency, str) or not dependency.startswith('./'):
            raise ValueError('The stage-2 pretrained_slat_enc must be an explicit ./ bundle-relative prefix.')
        if stage2_args.get('slat_enc_path') is not None or stage2_args.get('slat_enc_ckpt') is not None:
            raise ValueError('Bundled stage-2 models must not override the encoder with slat_enc_path/ckpt.')
        encoder_prefix = (Path(model_cfgs['slat_denoiser']).parent / dependency).resolve()
        if not encoder_prefix.is_relative_to(directory):
            raise ValueError('The stage-2 SLat encoder must stay inside the bundle.')
        for suffix in ('.json', '.safetensors'):
            if not Path(f'{encoder_prefix}{suffix}').is_file():
                raise ValueError(f'Missing bundled SLat encoder: {encoder_prefix}{suffix}')
        architecture = json.loads(Path(f'{model_cfgs["denoiser"]}.json').read_text())
        slots = architecture['args'].get('num_view_embeds', 0)
        if slots > 0:
            preprocessing = dict(preprocessing, max_views=min(limit, slots))
        return cls(model_cfgs, inference, preprocessing)

    def to(self, device):
        """Select a CUDA device without initializing the models."""
        value = str(device)
        if value != 'cuda' and not (value.startswith('cuda:') and value[5:].isdigit()):
            raise ValueError('GenSfM inference requires a CUDA device.')
        self.device = value
        return self

    def cuda(self):
        return self.to('cuda')

    def run(self, images, seed=1, steps=None, remove_background=False):
        """Reconstruct ordered Path/string inputs on CUDA.

        Models and DINOv2 initialize during each run, rather than during bundle
        loading. Returns decoded representations, cameras and the prepared inputs.
        """
        if not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError('seed must be between 0 and 2**32-1')
        if steps is not None and (not isinstance(steps, int) or steps < 1):
            raise ValueError('steps must be a positive integer')
        if isinstance(images, (str, Path)):
            raise ValueError('Provide an ordered sequence of image paths.')
        paths = collect_images(images)
        limit = self.preprocessing['max_views']
        if len(paths) > limit:
            raise ValueError(f'This stage-1 checkpoint supports at most {limit} views, received {len(paths)}. No images were discarded.')
        options = copy.deepcopy(self.inference)
        options.update(NATIVE_INFERENCE_OPTIONS)
        for key in ('ssuv_sampler_args', 'slat_sampler_args'):
            if steps is not None:
                options[key]['params']['steps'] = steps
        os.environ.setdefault('SPCONV_ALGO', 'native')
        import torch
        from trellis import evaluators

        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required; run GenSfM in a GPU allocation.')
        with torch.cuda.device(self.device):
            inputs = InputViews(paths, self.preprocessing['crop_size_ratio'], remove_background)
            with TemporaryDirectory(prefix='gensfm-') as runtime:
                evaluator = evaluators.VisualImageConditionedMultiPoseSparseFlowMatchingEvaluator(
                    self.model_cfgs, inputs, output_dir=runtime, **options)
                for model in evaluator.models.values():
                    model.eval()
                evaluator._init_image_cond_model()
                outputs, extrinsics, intrinsics = reconstruct(evaluator, inputs, seed)
                return dict(outputs, extrinsics=extrinsics, intrinsics=intrinsics, inputs=inputs,
                            loaded_models=dict(evaluator.loaded_paths),
                            stage1_sampler=options['ssuv_sampler_args'],
                            stage2_sampler=options['slat_sampler_args'])
