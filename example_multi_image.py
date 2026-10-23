"""Reconstruct one object and its cameras from masked multi-view images."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
DATA_TO_MESH = [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]


def __getattr__(name):
    if name in {'InputViews', 'collect_images', 'reconstruct', 'source_intrinsics'}:
        from trellis.pipelines import gensfm
        return getattr(gensfm, name)
    raise AttributeError(f'module {__name__} has no attribute {name}')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--images', nargs='+', type=Path, help='Ordered views of the same object.')
    inputs.add_argument('--input-dir', type=Path, help='One object; use every image in filename order.')
    parser.add_argument('--output-dir', type=Path, required=True, help='A new directory for the result.')
    parser.add_argument('--pretrained', help='Local pretrained bundle directory or Hugging Face repository ID.')
    parser.add_argument('--revision', help='Hugging Face revision for --pretrained.')
    parser.add_argument('--stage1', type=Path, help='Local joint shape/camera checkpoint file.')
    parser.add_argument('--stage2', type=Path, help='Local trained appearance checkpoint file.')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--steps', type=int, help='Override denoising steps for both stages (local checkpoints: 25).')
    parser.add_argument('--remove-background', action='store_true', help='Use rembg for images without foreground alpha.')
    args = parser.parse_args(argv)
    if args.pretrained:
        if args.stage1 is not None or args.stage2 is not None:
            parser.error('--pretrained cannot be combined with --stage1 or --stage2')
    elif args.stage1 is None or args.stage2 is None:
        parser.error('provide --pretrained, or both --stage1 and --stage2')
    if args.revision is not None and not args.pretrained:
        parser.error('--revision requires --pretrained')
    if args.steps is not None and args.steps < 1:
        parser.error('--steps must be positive')
    if not 0 <= args.seed < 2**32:
        parser.error('--seed must be between 0 and 2**32-1')
    return args


def resolve_model_path(value, *, checkpoint=False):
    """Resolve filesystem references while preserving pretrained Hub identifiers."""
    if not str(value).strip():
        raise ValueError('Model paths must not be empty.')
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    local = (checkpoint or str(value).startswith(('./', '../', '~'))
             or (path.parts and path.parts[0] in ('checkpoint', 'outputs'))
             or (PROJECT / path).exists()
             or (PROJECT / f'{value}.json').is_file())
    return str((PROJECT / path).resolve()) if local else str(value)


def inference_config(stage1, stage2, num_views, steps):
    """Use each checkpoint's own architecture and the trained stage-2 sampler."""
    checkpoints = [Path(p).expanduser().resolve() for p in (stage1, stage2)]
    configs = []
    for path in checkpoints:
        if not path.is_file():
            raise ValueError(f'Checkpoint does not exist: {path}')
        config_path = path.parent.parent / 'config.json'
        if not config_path.is_file():
            raise ValueError(f'Missing training configuration: {config_path}. Keep config.json beside the ckpts directory.')
        configs.append(json.loads(config_path.read_text()))
    coarse, fine = configs
    limit = coarse['models']['denoiser']['args'].get('num_view_embeds', 0)
    if limit > 0 and num_views > limit:
        raise ValueError(f'This stage-1 checkpoint supports at most {limit} views, received {num_views}. No images were discarded.')
    model_cfgs = {'denoiser': copy.deepcopy(coarse['models']['denoiser']),
                  'slat_denoiser': copy.deepcopy(fine['models']['denoiser'])}
    model_cfgs['denoiser']['path'] = str(checkpoints[0])
    model_cfgs['slat_denoiser']['path'] = str(checkpoints[1])
    for name in ('ss_decoder', 'suv_decoder'):
        model_cfgs[name] = copy.deepcopy(coarse['eval_models'][name])
    for name in ('slat_decoder_gs', 'slat_decoder_mesh'):
        model_cfgs[name] = copy.deepcopy(fine['eval_models'][name])
    for name, cfg in list(model_cfgs.items()):
        if isinstance(cfg, str):
            model_cfgs[name] = resolve_model_path(cfg)
        else:
            cfg['compile'] = False
            if 'path' in cfg:
                cfg['path'] = resolve_model_path(cfg['path'], checkpoint='name' in cfg)
    options = copy.deepcopy(coarse['evaluator']['args'])
    options.update(batch_size_per_gpu=1, num_workers=0, max_num_samples=1,
                   sample_ssuv=True, use_gt_intrinsics=False, alignment_method='none',
                   evaluate_pose=False, evaluate_3d_geometry=False, evaluate_novel_view=False,
                   evaluate_cross_view=False, evaluate_clip_metric=False, save_debug=False,
                   eval_intermediate_steps=False, save_results=False, save_mesh=False,
                   save_pose=False, image_cond_model_compile=False)
    options['normalization'] = copy.deepcopy(fine['dataset']['args']['normalization'])
    options['slat_sampler_args'] = copy.deepcopy(fine['evaluator']['args']['slat_sampler_args'])
    for key in ('ssuv_sampler_args', 'slat_sampler_args'):
        options[key]['params']['steps'] = steps
    options['image_cond_model'] = coarse['trainer']['args'].get('image_cond_model', 'dinov2_vitl14_reg')
    return model_cfgs, options, coarse['dataset']['args']['crop_size_ratio']


def mesh_reprojections(mesh, extrinsics, intrinsics, resolution=518):
    """Render the exported textured mesh with the complete predicted K, including skew."""
    import numpy as np
    import torch
    import nvdiffrast.torch as dr

    device = extrinsics.device
    vertices = torch.as_tensor(np.asarray(mesh.vertices).copy(), dtype=torch.float64, device=device)
    vertices = torch.cat((vertices, torch.ones_like(vertices[:, :1])), dim=1)
    basis = torch.tensor(DATA_TO_MESH, dtype=torch.float64, device=device)
    vertices = vertices @ torch.linalg.inv(basis).T
    faces = torch.as_tensor(np.asarray(mesh.faces).copy(), dtype=torch.int32, device=device)
    uv = torch.as_tensor(np.asarray(mesh.visual.uv).copy(), dtype=torch.float32, device=device)[None]
    material = mesh.visual.material
    texture = torch.as_tensor(np.array(material.baseColorTexture.convert('RGBA'))[::-1].copy(),
                              dtype=torch.float32, device=device)[None] / 255
    factor = torch.as_tensor(np.asarray(material.baseColorFactor).copy(), dtype=torch.float32, device=device) / 255
    context = dr.RasterizeCudaContext(device=device)
    images = []
    for extrinsic, intrinsic in zip(extrinsics.double(), intrinsics.double()):
        if not torch.allclose(intrinsic[2], intrinsic.new_tensor([0, 0, 1]), atol=1e-6, rtol=0):
            raise ValueError('Expected a normalized pinhole intrinsic matrix')
        camera = vertices @ extrinsic.T
        q = camera[:, :3] @ intrinsic.T
        depth = camera[:, 2]
        near, far = 0.001, max(100.0, float(depth.max()) + 1)
        # nvdiffrast's first array row is saved as the first image row (top).
        clip = torch.stack((2*q[:, 0]-q[:, 2], 2*q[:, 1]-q[:, 2],
                            depth*(far+near)/(far-near)-2*near*far/(far-near), q[:, 2]), dim=1)
        clip = clip.float()[None].contiguous()
        raster, _ = dr.rasterize(context, clip, faces, (resolution, resolution))
        texcoords, _ = dr.interpolate(uv, raster, faces)
        rgba = dr.texture(texture, texcoords, filter_mode='linear') * factor
        alpha = rgba[..., 3:] * (raster[..., 3:] > 0)
        color = dr.antialias((rgba[..., :3]*alpha).contiguous(), raster, clip, faces)[0]
        images.append((color.clamp(0, 1).cpu().numpy()*255).round().astype(np.uint8))
    return images


def export_result(outputs, extrinsics, intrinsics, inputs, output_dir):
    import numpy as np
    import torch
    from PIL import Image
    from trellis.utils import postprocessing_utils
    from trellis.pipelines.gensfm import source_intrinsics

    print('Exporting textured GLB, Gaussians and camera reprojections', flush=True)
    with torch.enable_grad():
        mesh = postprocessing_utils.to_glb(outputs['gaussian'][0], outputs['mesh'][0],
                                          simplify=0.95, texture_size=1024, verbose=False)
    mesh.export(output_dir / 'mesh.glb')
    outputs['gaussian'][0].save_ply(str(output_dir / 'gaussian.ply'))
    with torch.no_grad():
        renders = mesh_reprojections(mesh, extrinsics, intrinsics)
    for i, (condition, rgba, render) in enumerate(zip(inputs.condition, inputs.rgba, renders)):
        Image.fromarray((condition.permute(1, 2, 0).numpy()*255).round().clip(0, 255).astype(np.uint8)).save(output_dir / f'input-{i:02d}.png')
        Image.fromarray((rgba.permute(1, 2, 0).numpy()*255).round().clip(0, 255).astype(np.uint8)).save(output_dir / f'input-rgba-{i:02d}.png')
        Image.fromarray(render.astype(np.uint8)).save(output_dir / f'reprojection-{i:02d}.png')
    k = intrinsics.detach().cpu().numpy()
    cameras = {'convention': 'OpenCV world-to-camera; normalized complete K; data coordinates are Z-up',
               'intrinsics': k.tolist(), 'extrinsics': extrinsics.detach().cpu().tolist(),
               'intrinsics_source': source_intrinsics(k, inputs.records),
               'source_frame': 'EXIF-oriented source image before foreground crop',
               'data_to_mesh': DATA_TO_MESH, 'gt_assisted_alignment': False}
    (output_dir / 'cameras.json').write_text(json.dumps(cameras, indent=2, allow_nan=False)+'\n')


def main(argv=None):
    args = parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise ValueError(f'Output directory already exists: {output}. Choose a new directory.')
    os.environ.setdefault('SPCONV_ALGO', 'native')
    import torch
    from trellis.pipelines.gensfm import GenSfMPipeline, InputViews, collect_images, reconstruct

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; run this example in a GPU allocation.')
    paths = collect_images(args.images, args.input_dir)
    if args.pretrained:
        pipeline = GenSfMPipeline.from_pretrained(args.pretrained, revision=args.revision).cuda()
        outputs = pipeline.run(paths, seed=args.seed, steps=args.steps, remove_background=args.remove_background)
        extrinsics, intrinsics, inputs = outputs['extrinsics'], outputs['intrinsics'], outputs['inputs']
        loaded_models = outputs['loaded_models']
        stage1_sampler, stage2_sampler = outputs['stage1_sampler'], outputs['stage2_sampler']
    else:
        from trellis import evaluators
        models, options, crop_ratio = inference_config(
            args.stage1, args.stage2, len(paths), args.steps if args.steps is not None else 25)
        inputs = InputViews(paths, crop_ratio, args.remove_background)
        output.mkdir(parents=True)
        evaluator = evaluators.VisualImageConditionedMultiPoseSparseFlowMatchingEvaluator(
            models, inputs, output_dir=str(output / 'runtime'), **options)
        for model in evaluator.models.values():
            model.eval()
        evaluator._init_image_cond_model()
        outputs, extrinsics, intrinsics = reconstruct(evaluator, inputs, args.seed)
        loaded_models = evaluator.loaded_paths
        stage1_sampler, stage2_sampler = options['ssuv_sampler_args'], options['slat_sampler_args']
    output.mkdir(parents=True, exist_ok=True)
    export_result(outputs, extrinsics, intrinsics, inputs, output)
    metadata = {'seed': args.seed, 'num_views': len(paths), 'inputs': inputs.records,
                'loaded_models': loaded_models,
                'stage1_sampler': stage1_sampler, 'stage2_sampler': stage2_sampler}
    if args.pretrained:
        metadata.update(pretrained=args.pretrained, revision=args.revision)
    (output / 'run.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    print(f'Result saved to {output}', flush=True)


if __name__ == '__main__':
    main()
