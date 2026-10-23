import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from easydict import EasyDict as edict
from torch.utils.data import DataLoader, Dataset
from concurrent.futures import ThreadPoolExecutor

import trellis.models as models
from trellis.datasets import ImageConditionedSparseUVStructure


torch.set_grad_enabled(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=4.5,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--enc_pretrained', type=str, default=None,
                        help='Pretrained encoder model')
    parser.add_argument('--model_root', type=str, default='results',
                        help='Root directory of models')
    parser.add_argument('--enc_model', type=str, default=None,
                        help='Encoder model. if specified, use this model instead of pretrained model')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Checkpoint to load')
    parser.add_argument('--resolution', type=int, default=64,
                        help='Resolution of the voxel grid')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for processing samples (adjust based on GPU memory)')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of worker processes for DataLoader')
    parser.add_argument('--discard_occluded_voxels', type=lambda x: str(x).lower() in ['true','1','yes'], default=False,
                        help='Whether to discard occluded voxels')
    parser.add_argument('--return_relative_depth', action='store_true',
                        help='Whether to return relative depth')
    parser.add_argument('--dilation_sigma', type=float, default=0.0,
                        help='Standard deviation of Gaussian kernel for dilation')
    parser.add_argument('--dilation_min_weight', type=float, default=0.05,
                        help='Minimum weight to include in the dilated grid')
    parser.add_argument('--crop_size_ratio', type=str, default="",
                        help='Crop size ratio for image conditioning')
    parser.add_argument('--crop_num_discretization_buckets', type=int, default=None,
                        help='Number of discretization buckets for crop size ratio')
    parser.add_argument('--const_ssuv', type=lambda x: str(x).lower() in ['true','1','yes'], default=False,
                        help='Whether to use constant ssuv')
    # Input configuration (match evaluator defaults)
    parser.add_argument('--input_ssuv', type=lambda x: str(x).lower() in ['true','1','yes'], default=True,
                        help='Include sparse UV structure (ssuv) as input')
    parser.add_argument('--input_uv_volume', type=lambda x: str(x).lower() in ['true','1','yes'], default=True,
                        help='Include uv_volume as input')
    parser.add_argument('--input_depth', type=lambda x: str(x).lower() in ['true','1','yes'], default=False,
                        help='Include depth_volume as input (requires --return_relative_depth)')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = parser.parse_args()
    opt = edict(vars(opt))

    # Load encoder
    if opt.enc_model is None:
        assert opt.enc_pretrained is not None, 'Either --enc_pretrained or --enc_model/--ckpt must be specified'
        latent_name = f'{opt.enc_pretrained.split("/")[-1]}'
        encoder = models.from_pretrained(opt.enc_pretrained).eval().cuda()
    else:
        latent_name = f'{opt.enc_model}_{opt.ckpt}'
        cfg = edict(json.load(open(os.path.join(opt.model_root, opt.enc_model, 'config.json'), 'r')))
        encoder = getattr(models, cfg.models.encoder.name)(**cfg.models.encoder.args).cuda()
        ckpt_path = os.path.join(opt.model_root, opt.enc_model, 'ckpts', f'encoder_{opt.ckpt}.pt')
        encoder.load_state_dict(torch.load(ckpt_path, weights_only=True), strict=True)
        encoder.eval()
        print(f'Loaded model from {ckpt_path}')

    save_root = os.path.join(opt.output_dir, 'suv_latents', latent_name)
    os.makedirs(save_root, exist_ok=True)

    # Build dataset (iterate all conditional views)
    dataset_args = {
        'roots': opt.output_dir,
        'resolution': opt.resolution,
        'min_aesthetic_score': opt.filter_low_aesthetic_score,
        'discard_occluded_voxels': opt.discard_occluded_voxels,
        'return_relative_depth': opt.return_relative_depth,
        'dilation_sigma': opt.dilation_sigma,
        'dilation_min_weight': opt.dilation_min_weight,
        'crop_size_ratio': eval(opt.crop_size_ratio) if opt.crop_size_ratio else None,
        'crop_in_eval': bool(opt.crop_size_ratio),
        'crop_num_discretization_buckets': opt.crop_num_discretization_buckets,
        'const_ssuv': opt.const_ssuv,
        'is_eval': True,
    }
    if opt.rank == 0:
        print("Dataset arguments:", dataset_args)
    dataset = ImageConditionedSparseUVStructure(**dataset_args)

    # Use DataLoader to iterate tasks (instance, view-window) in parallel
    # len(dataset) == num_instances * num_cond_views when is_eval=True
    num_instances = len(dataset) // dataset.num_cond_views
    inst_start = num_instances * opt.rank // opt.world_size
    inst_end = num_instances * (opt.rank + 1) // opt.world_size

    class EncodeTasksDataset(Dataset):
        def __init__(self, base_dataset, instance_range, batch_size, opt):
            self.base_dataset = base_dataset
            self.instance_range = instance_range
            self.batch_size = batch_size
            self.opt = opt
            self.tasks = []  # (inst_idx, start_view)

            column_name = f'suv_latent_{latent_name}'
            for inst_idx in range(instance_range[0], instance_range[1]):
                # filter out objects that are already processed
                if column_name in self.base_dataset.metadata.columns:
                    sha256 = self.base_dataset.instances[inst_idx][1]
                    if self.base_dataset.metadata.loc[sha256, column_name] == True:
                        continue
                for start_view in range(0, self.base_dataset.num_cond_views, self.batch_size):
                    self.tasks.append((inst_idx, start_view))

        def __len__(self):
            if self.opt.crop_num_discretization_buckets is not None:
                return len(self.tasks) * self.opt.crop_num_discretization_buckets
            else:
                return len(self.tasks)

        def __getitem__(self, idx):
            try:
                if self.opt.crop_num_discretization_buckets is not None:
                    idx, bucket_idx = divmod(idx, self.opt.crop_num_discretization_buckets)
                    self.base_dataset.override_buekct_idx = bucket_idx
                inst_idx, start_view = self.tasks[idx]
                root, sha256 = self.base_dataset.instances[inst_idx]

                end_view = min(self.base_dataset.num_cond_views, start_view + self.batch_size)
                batch_inputs = []
                batch_view_names = []

                for view_idx in range(start_view, end_view):
                    sample = self.base_dataset[inst_idx * self.base_dataset.num_cond_views + view_idx]
                    inputs = []
                    if self.opt.input_ssuv:
                        inputs.append(sample['ssuv'].float())
                    if self.opt.input_uv_volume:
                        inputs.append(sample['uv_volume'].float())
                    if self.opt.input_depth and 'depth_volume' in sample:
                        inputs.append(sample['depth_volume'].float())
                    if len(inputs) == 0:
                        raise ValueError('No inputs selected. Enable at least one of --input_ssuv/--input_uv_volume/--input_depth')
                    input_volume = torch.cat(inputs, dim=0)
                    batch_inputs.append(input_volume)
                    # assume the image name is like "000.png", "001.png", ... in frame order
                    batch_view_names.append(f"{view_idx:03d}")

                batch_inputs = torch.stack(batch_inputs).float()
                ratio = self.base_dataset.last_crop_ratio
                if ratio is None:
                    batch_latent_names = [f'{vn}_mean' for vn in batch_view_names]
                else:
                    batch_latent_names = [f'{vn}_mean_crop{int(ratio * 1e4)}' for vn in batch_view_names]

                return {
                    'sha256': sha256,
                    'inputs': batch_inputs,
                    'latent_names': batch_latent_names,
                }
            except Exception as e:
                print(f"Error processing idx={idx}: {e}")
                return {'sha256': None}

    task_dataset = EncodeTasksDataset(dataset, (inst_start, inst_end), opt.batch_size, opt)
    loader = DataLoader(
        task_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        collate_fn=lambda items: items[0],  # unwrap singleton
    )

    records = []
    current_sha = None
    current_skip = False
    pack = {}  # latent_name -> (cuda_event, cpu_tensor_view)
    column_name = f'suv_latent_{latent_name}'

    # Background saver to overlap GPU compute with CPU compression+IO
    max_pending_saves = 24  # backpressure to avoid unbounded memory growth
    d2h_stream = torch.cuda.Stream()
    pending = []  # list of (sha256, future)

    def _async_d2h(latents_gpu: torch.Tensor, stream: torch.cuda.Stream):
        """
        Asynchronously copy a CUDA tensor to pinned CPU memory.

        Returns:
            (event, cpu_tensor) where event signals copy completion.
        """
        latents_gpu = latents_gpu.detach()
        latents_cpu = torch.empty_like(latents_gpu, device='cpu', pin_memory=True)
        compute_stream = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            stream.wait_stream(compute_stream)
            latents_cpu.copy_(latents_gpu, non_blocking=True)
            # Ensure source storage isn't reused before copy completes
            latents_gpu.record_stream(stream)
            ev = torch.cuda.Event()
            ev.record(stream)
        return ev, latents_cpu

    def _submit_save(executor: ThreadPoolExecutor, sha256: str, pack_dict: dict):
        save_path = os.path.join(save_root, f'{sha256}.npz')

        def _save_job(local_pack: dict, path: str):
            # Wait for all async D2H copies to finish
            events = {}
            for ev, _t in local_pack.values():
                events[id(ev)] = ev
            for ev in events.values():
                ev.synchronize()

            np_pack = {k: t.contiguous().numpy() for k, (_ev, t) in local_pack.items()}
            np.savez_compressed(path, **np_pack)

        return executor.submit(_save_job, pack_dict, save_path)

    def _is_valid_npz(path: str) -> bool:
        if not os.path.isfile(path):
            return False
        try:
            with np.load(path) as data:
                _ = data.files
            return True
        except Exception:
            return False

    def _drain_one():
        done_sha, done_fut = pending.pop(0)
        done_fut.result()
        records.append({'sha256': done_sha, column_name: True})

    def _flush_current(executor: ThreadPoolExecutor, sha: str, pack_dict: dict):
        if sha is None or len(pack_dict) == 0:
            return None, {}

        fut = _submit_save(executor, sha, pack_dict)
        pending.append((sha, fut))

        while len(pending) >= max_pending_saves:
            _drain_one()

        return None, {}

    with ThreadPoolExecutor(max_workers=32) as saver_executor:
        for item in tqdm(loader, desc='Extracting latents (views)', mininterval=10, maxinterval=60):
            sha256 = item.get('sha256', None)
            if sha256 is None:
                continue

            if current_sha is None:
                current_sha = sha256
                pack = {}
                save_path = os.path.join(save_root, f'{sha256}.npz')
                current_skip = _is_valid_npz(save_path)
                if current_skip:
                    continue
            elif sha256 != current_sha:
                if not current_skip:
                    current_sha, pack = _flush_current(saver_executor, current_sha, pack)
                current_sha = sha256
                pack = {}
                save_path = os.path.join(save_root, f'{sha256}.npz')
                current_skip = _is_valid_npz(save_path)
                if current_skip:
                    continue
            elif current_skip:
                continue

            with torch.no_grad():
                batch = item['inputs'].cuda(non_blocking=True)
                latents_gpu = encoder(batch, sample_posterior=False, return_raw=False)
                assert torch.isfinite(latents_gpu).all(), 'Non-finite latent'
                ev, latents_cpu = _async_d2h(latents_gpu, d2h_stream)

            for i, vn in enumerate(item['latent_names']):
                pack[vn] = (ev, latents_cpu[i])

        # Save last instance pack if exists
        current_sha, pack = _flush_current(saver_executor, current_sha, pack)

        # Drain all pending saves
        while len(pending) > 0:
            _drain_one()

    # Save a small CSV log per-rank
    df = pd.DataFrame.from_records(records)
    df.to_csv(os.path.join(opt.output_dir, f'{column_name}_{opt.rank}.csv'), index=False)