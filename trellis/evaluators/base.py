import os
import copy
import time
import json
from typing import List, Optional
from tqdm import tqdm
from functools import partial
from contextlib import nullcontext
from abc import abstractmethod

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import utils
import numpy as np
import pandas as pd

from ..datasets.components import StandardDatasetBase
from ..utils.general_utils import *
from ..utils.data_utils import recursive_to_device
from ..utils.dist_utils import *


class IndexedDatasetWrapper(StandardDatasetBase):
    """A wrapper dataset that returns both the original data and indices.

    ``keep_indices`` (optional): subset of *original* dataset indices to
    iterate over. ``__len__`` shrinks to ``len(keep_indices)`` while
    ``__getitem__`` still emits the original index in the second tuple
    element — so save_images filenames stay aligned with the un-subsetted
    dataset (i.e. ``{orig_idx}_v{vi}_*.png`` even when only a few samples
    were evaluated).
    """
    def __init__(self, dataset: StandardDatasetBase, keep_indices: Optional[List[int]] = None):
        self.dataset = dataset
        self.keep_indices = list(keep_indices) if keep_indices is not None else None

    def __len__(self):
        return len(self.keep_indices) if self.keep_indices is not None else len(self.dataset)

    def __getitem__(self, idx: int):
        orig_idx = self.keep_indices[idx] if self.keep_indices is not None else idx
        data = self.dataset[orig_idx]
        index = torch.tensor(orig_idx, dtype=torch.long)
        return data, index

    def _collate(self, batch):
        data_list = [item[0] for item in batch]
        indices_list = [item[1] for item in batch]
        collated_data = self.dataset.collate_fn(data_list)
        collated_indices = torch.stack(indices_list)
        return collated_data, collated_indices

    @property
    def collate_fn(self):
        return self._collate if hasattr(self.dataset, 'collate_fn') else None


class Evaluator:
    """
    Base Evaluator for evaluation loop.
    
    Args:
        model_cfgs (dict): Configs for the evaluation models.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        load_step (int): Step to load.
        pretrained_ckpt (dict): Pretrained checkpoint.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        i_sample (int): Sample interval.
        max_num_samples (int): Maximum number of samples to evaluate.
    """
    def __init__(
        self,
        model_cfgs,
        dataset,
        *,
        output_dir,
        load_dir=None,
        load_step=None,
        load_ema=None,
        pretrained_ckpt=None,
        batch_size=None,
        batch_size_per_gpu=None,
        fp16_mode='inflat_all',
        i_sample=1000,
        max_num_samples=None,
        instance_indices: Optional[List[int]] = None,
        save_postfix=None,
        **kwargs
    ):
        assert batch_size is not None or batch_size_per_gpu is not None, 'Either batch_size or batch_size_per_gpu must be specified.'

        self.device = torch.device('cuda')
        self.dataset = dataset
        self.fp16_mode = fp16_mode
        self.output_dir = output_dir
        self.i_sample = i_sample
        self.max_num_samples = max_num_samples
        self.instance_indices = list(instance_indices) if instance_indices else None
        postfix_str = f'_{save_postfix}' if save_postfix else ''
        self.metrics_root = os.path.join(output_dir, f'eval_metrics{postfix_str}')
        self.results_root = os.path.join(output_dir, f'eval_results{postfix_str}')
        self.samples_root = os.path.join(output_dir, f'eval_samples{postfix_str}')
        self.model_cfgs = model_cfgs
        self.kwargs = kwargs
        
        if dist.is_initialized():
            # Multi-GPU params
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = dist.get_rank() % torch.cuda.device_count()
            self.is_master = self.rank == 0
        else:
            # Single-GPU params
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.is_master = True

        self.batch_size = batch_size if batch_size_per_gpu is None else batch_size_per_gpu * self.world_size
        self.batch_size_per_gpu = batch_size_per_gpu if batch_size_per_gpu is not None else batch_size // self.world_size
        assert self.batch_size % self.world_size == 0, 'Batch size must be divisible by the number of GPUs.'

        self.prepare_dataloader(**kwargs)
        
        # Build models
        self.models = {}
        self.loaded_paths = {}
        for name, cfg in model_cfgs.items():
            self.models[name] = build_model(cfg, device=self.device)

        # Load checkpoint
        if load_dir is not None and load_step is not None:
            self.load(model_cfgs, load_dir=load_dir, load_step=load_step, load_ema=load_ema)
        elif pretrained_ckpt is not None:
            self.load(model_cfgs, pretrained_ckpt=pretrained_ckpt)
        else:
            self.load(model_cfgs)
        
        if self.is_master:
            os.makedirs(self.samples_root, exist_ok=True)
            os.makedirs(self.metrics_root, exist_ok=True)

        if self.is_master:
            print('\n\nEvaluator initialized.')
            print(self)

    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Models:')
        for name, model in self.models.items():
            lines.append(f'    - {name}: {model.__class__.__name__}')
        lines.append(f'  - Loaded paths:')
        for name, path in self.loaded_paths.items():
            lines.append(f'    - {name}: {path}')
        lines.append(f'  - Dataset: {indent(str(self.dataset), 2)}')
        lines.append(f'  - Dataloader:')
        lines.append(f'    - Sampler: {self.dataloader.sampler.__class__.__name__}')
        lines.append(f'    - Num workers: {self.dataloader.num_workers}')
        lines.append(f'  - Number of GPUs: {self.world_size}')
        lines.append(f'  - Batch size: {self.batch_size}')
        lines.append(f'  - Batch size per GPU: {self.batch_size_per_gpu}')
        lines.append(f'  - FP16 mode: {self.fp16_mode}')
        return '\n'.join(lines)

    def prepare_dataloader(self, num_workers=8, **kwargs):
        """
        Prepare dataloader for evaluation.
        """
        dataset = IndexedDatasetWrapper(self.dataset, keep_indices=self.instance_indices)
        self.data_sampler = DistributedSampler(
            dataset, 
            num_replicas=self.world_size, 
            rank=self.rank, 
            shuffle=False,
            drop_last=True,
        )
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=num_workers > 0,
            sampler=self.data_sampler,
            collate_fn=dataset.collate_fn,
        )
        self.total_samples = len(dataset)
        if self.max_num_samples is not None:
            if self.is_master:
                print(f'Testing the first {self.max_num_samples} samples out of {self.total_samples} total samples.')
            self.total_samples = min(self.total_samples, self.max_num_samples)
        self.total_batches = self.total_samples // self.batch_size
        if self.total_samples % self.batch_size != 0:
            if self.is_master:
                print(f'Warning: Original total samples {self.total_samples} is not divisible by batchsize {self.batch_size}.')
                print('This will cause some samples in the last incomplete batch to be ignored in the evaluation.')
            self.total_samples = self.batch_size * self.total_batches

        # Initialize the getter for instance indices, view indices, and instances
        if self.dataset.is_eval and hasattr(self.dataset, 'get_instance_view_index'):
            self.get_instance_view_indices = lambda indices: self.dataset.get_instance_view_index(indices)
        else:
            self.get_instance_view_indices = lambda indices: (indices, np.full_like(indices, -1))
        self.instances = np.array(self.dataset.instances)

    def load(self, model_cfgs, load_dir=None, load_step=None, load_ema=None, pretrained_ckpt=None):
        """
        Load model weights from checkpoints.
        Models are already built in __init__; this only loads weights.
        Should be called by all processes.
        """
        if self.is_master:
            if load_dir is not None and load_step is not None:
                print(f'\nLoading checkpoint from step {load_step} in {load_dir}...', end='')
            if load_ema is not None:
                print(f'\nUsing EMA weights = {load_ema}...', end='')
            if pretrained_ckpt is not None:
                print('\nLoading pretrained weights from:')
                for name, path in pretrained_ckpt.items():
                    print(f'  - {name}: {path}')

        for name, model_cfg in model_cfgs.items():
            model = self.models[name]

            # from_pretrained models: already loaded during build
            if isinstance(model_cfg, str):
                self.loaded_paths[name] = model_cfg
                continue
            if 'name' not in model_cfg:
                # dict with 'path' only (e.g. {"path": "...", "compile": true})
                # already loaded via from_pretrained in build_model
                self.loaded_paths[name] = model_cfg['path']
                continue

            # Determine weight path
            if 'path' in model_cfg:
                load_path = model_cfg['path']
                if self.is_master:
                    print(f'\nLoading eval model weights: {name}: {load_path}')
            elif load_dir is not None and load_step is not None:
                if load_ema is not None:
                    filename = f'{name}_ema{load_ema}_step{load_step:07d}.pt'
                else:
                    filename = f'{name}_step{load_step:07d}.pt'
                load_path = os.path.join(load_dir, 'ckpts', filename)
            elif pretrained_ckpt is not None:
                load_path = pretrained_ckpt[name]
            else:
                load_path = None
                if self.is_master:
                    print(f'Warning: {name} not loaded')

            if load_path is not None:
                model_ckpt = load_ckpt_dist(load_path, map_location=self.device)
                unwrap_model(model).load_state_dict(model_ckpt)
                self.loaded_paths[name] = load_path

            if self.fp16_mode == 'inflat_all':
                unwrap_model(model).convert_to_fp16()

        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print(' Done.')

    @torch.no_grad()
    def visualize_sample(self, sample):
        """
        Convert a sample to an image.
        """
        if hasattr(self.dataset, 'visualize_sample'):
            return self.dataset.visualize_sample(sample)
        else:
            return sample

    @torch.no_grad()
    def save_samples(self, samples, batch_idx):
        """
        Save sample images from the model.
        NOTE: This function should be called by all processes.
        """
        suffix = f'batch_{batch_idx:07d}'

        # Preprocess images
        for key in list(samples.keys()):
            if samples[key]['type'] == 'sample':
                vis = self.visualize_sample(samples[key]['value'])
                if isinstance(vis, dict):
                    for k, v in vis.items():
                        samples[f'{key}_{k}'] = {'value': v, 'type': 'image'}
                    del samples[key]
                else:
                    samples[key] = {'value': vis, 'type': 'image'}

        # Gather results
        if self.world_size > 1:
            for key in samples.keys():
                samples[key]['value'] = samples[key]['value'].contiguous()
                if self.is_master:
                    all_images = [torch.empty_like(samples[key]['value']) for _ in range(self.world_size)]
                else:
                    all_images = []
                dist.gather(samples[key]['value'], all_images, dst=0)
                if self.is_master:
                    samples[key]['value'] = torch.cat(all_images, dim=0)

        # Save images
        if self.is_master:
            os.makedirs(os.path.join(self.samples_root, suffix), exist_ok=True)
            for key in samples.keys():
                if samples[key]['type'] == 'image':
                    utils.save_image(
                        samples[key]['value'],
                        os.path.join(self.samples_root, suffix, f'{key}_{suffix}.jpg'),
                        nrow=int(np.sqrt(samples[key]['value'].shape[0])),
                        normalize=True,
                        value_range=self.dataset.value_range,
                    )
                elif samples[key]['type'] == 'mv_image':
                    subgrids = make_mv_subgrids(samples[key]['value'], value_range=self.dataset.value_range)
                    utils.save_image(
                        subgrids,
                        os.path.join(self.samples_root, suffix, f'{key}_{suffix}.jpg'),
                        nrow=int(np.sqrt(subgrids.shape[0])),
                        normalize=False,
                    )
                elif samples[key]['type'] == 'number':
                    min = samples[key]['value'].min()
                    max = samples[key]['value'].max()
                    images = (samples[key]['value'] - min) / (max - min)
                    images = utils.make_grid(
                        images,
                        nrow=int(np.sqrt(samples[key]['value'].shape[0])),
                        normalize=False,
                    )
                    save_image_with_notes(
                        images,
                        os.path.join(self.samples_root, suffix, f'{key}_{suffix}.jpg'),
                        notes=f'{key} min: {min}, max: {max}',
                    )

    @abstractmethod
    def evaluation_metrics(self, indices, **mb_data):
        """
        Compute evaluation metrics.
        Return a 2-tuple of:
            - a dict of metrics: {name: scalar values of shape (batch_size_per_gpu,)}.
            - a dict of samples: {name: {'value': samples of shape (batch_size_per_gpu, ...), 'type': sample type}}.
        """
        pass

    def run(self):
        """
        Run evaluation.
        """
        if self.is_master:
            print('\nStarting evaluation...')
            print(f'Total samples: {self.total_samples}, Total batches per process: {self.total_batches}')

        # Initialize metrics collection
        all_dataframes = []
        batch_idx = 0
        
        # Create progress bar for master process
        if self.is_master:
            pbar = tqdm(total=self.total_batches, desc="Evaluating", unit="batch")
        
        time_start = time.time()
        amp_context = partial(torch.autocast, device_type='cuda') if self.fp16_mode == 'amp' else nullcontext
        
        for data, indices in self.dataloader:
            # Move data to device
            data = recursive_to_device(data, self.device, non_blocking=True)
            indices = recursive_to_device(indices, self.device, non_blocking=True)
            
            # Run evaluation step and collect metrics
            with amp_context(), torch.no_grad():
                metric_dict, sample_dict = self.evaluation_metrics(indices=indices, **data)
            
            # Create DataFrame for this batch
            batch_df = self.create_batch_dataframe(metric_dict, indices)
            all_dataframes.append(batch_df)
            
            # Save samples occasionally
            if batch_idx % self.i_sample == 0:
                self.save_samples(sample_dict, batch_idx)
            
            batch_idx += 1
            if self.is_master:
                pbar.update(1)
            if batch_idx >= self.total_batches:
                break
        
        if self.is_master:
            pbar.close()
        
        # Save per-process metrics DataFrame
        self.save_per_process_metrics(all_dataframes)
        
        # Ensure all processes wait for aggregation to complete
        if self.world_size > 1:
            dist.barrier()
        
        # Merge all process metrics and save final evaluation results
        if self.is_master:
            all_metrics_df = self.merge_all_metrics()
            self.save_metrics_statistics(all_metrics_df)
        
        time_end = time.time()
        if self.is_master:
            print(f'Evaluation finished in {time_end - time_start:.2f} seconds.')
    
    def create_batch_dataframe(self, metric_dict, indices):
        """
        Create a DataFrame for a batch with metrics and indices.
        
        Args:
            metric_dict: Dictionary of metrics with scalar values
            indices: Tensor of indices for this batch
            
        Returns:
            pd.DataFrame: DataFrame with index, root, sha256, and metric columns
        """
        # Convert indices to CPU numpy array
        indices_np = indices.cpu().numpy()
        
        # Get (root, sha256) pairs for each index
        instance_indices, view_indices = self.get_instance_view_indices(indices_np)
        
        # Extract root and sha256 arrays directly using numpy indexing
        root_sha256_pairs = self.instances[instance_indices]
        roots = root_sha256_pairs[:, 0]
        sha256s = root_sha256_pairs[:, 1]

        # Create DataFrame
        df_data = {
            'dataset_index': indices_np,
            'instance_index': instance_indices,
            'view_index': view_indices,
            'root': roots,
            'sha256': sha256s
        }
        
        # Add metric columns
        for metric_name, metric_values in metric_dict.items():
            assert isinstance(metric_values, torch.Tensor)
            df_data[metric_name] = metric_values.cpu().numpy()
        
        return pd.DataFrame(df_data)
    
    def save_per_process_metrics(self, all_dataframes):
        """
        Save the combined DataFrame from this process to a CSV file.
        
        Args:
            all_dataframes: List of DataFrames from all batches
        """
        # Concatenate all batch DataFrames
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Save to CSV
        csv_path = os.path.join(self.metrics_root, f'metrics_{self.rank:02d}.csv')
        combined_df.to_csv(csv_path, index=False)
        print(f'Process {self.rank} metrics saved to {csv_path}')
    
    def merge_all_metrics(self):
        """
        Merge metrics CSV files from all processes into a single file.
        This should only be called by the master process.
        """
        all_dfs = []
        
        # Read CSV files from all processes
        for rank in range(self.world_size):
            csv_path = os.path.join(self.metrics_root, f'metrics_{rank:02d}.csv')
            all_dfs.append(pd.read_csv(csv_path))
        
        # Concatenate all DataFrames
        final_df = pd.concat(all_dfs, ignore_index=True)
        
        # Sort by dataset_index for consistency
        final_df = final_df.sort_values('dataset_index').reset_index(drop=True)
        
        # Save merged DataFrame
        final_csv_path = os.path.join(self.metrics_root, 'metrics_all.csv')
        final_df.to_csv(final_csv_path, index=False)
        print(f'All process metrics merged and saved to {final_csv_path}')
        print(f'Total metrics entries: {len(final_df)}')

        return final_df
    
    _METADATA_COLUMNS = {'dataset_index', 'root', 'sha256', 'instance_index', 'view_index', 't'}

    @staticmethod
    def _compute_column_stats(values):
        """Compute standard summary statistics for a pandas Series."""
        return {
            'mean': float(values.mean()),
            'std': float(values.std()),
            'min': float(values.min()),
            '05%': float(values.quantile(0.05)),
            '25%': float(values.quantile(0.25)),
            '50%': float(values.quantile(0.5)),
            '75%': float(values.quantile(0.75)),
            '95%': float(values.quantile(0.95)),
            'max': float(values.max()),
            'count': int(len(values)),
        }

    def save_metrics_statistics(self, df):
        """
        Calculate statistics for all metrics and save to JSON file.

        Args:
            df: DataFrame containing all metrics
        """
        metric_columns = [c for c in df.columns if c not in self._METADATA_COLUMNS]

        if not metric_columns:
            print("No metric columns found in DataFrame")
            return

        # Calculate statistics for each metric
        metrics_stats = {}
        for metric_name in metric_columns:
            metrics_stats[metric_name] = self._compute_column_stats(df[metric_name])
        
        # Allow subclasses to reorganize metrics_stats (e.g., group per-view metrics)
        metrics_stats = self.postprocess_metrics_statistics(metrics_stats)

        # Add overall metadata
        results = {
            'metadata': {
                'total_samples': len(df),
                'total_metrics': len(metric_columns),
                'loaded_paths': self.loaded_paths
            },
            'statistics': metrics_stats,
            **self.get_extra_metrics_statistics(df)
        }

        # Save to JSON file
        json_path = os.path.join(self.metrics_root, 'stats.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f'Metrics statistics saved to {json_path}')
        print('Metrics summary:')
        for metric_name, stats in metrics_stats.items():
            if isinstance(stats, dict) and 'mean' in stats:
                print(f'  {metric_name}: mean={stats["mean"]:.5f}, std={stats["std"]:.5f}, min={stats["min"]:.5f}, 25%={stats["25%"]:.5f}, 50%={stats["50%"]:.5f}, 75%={stats["75%"]:.5f}, 95%={stats["95%"]:.5f}, max={stats["max"]:.5f}')
    
    def postprocess_metrics_statistics(self, metrics_stats):
        """
        Hook for subclasses to reorganize metrics_stats before saving.
        """
        return metrics_stats

    def get_extra_metrics_statistics(self, df):
        """
        Get extra metrics statistics.
        """
        return {}
        
