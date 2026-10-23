import os
import copy
import math
from functools import partial
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

from .utils import *
from .base import Trainer
from ..utils.general_utils import *
from ..utils.dist_utils import *
from ..utils import grad_clip_utils, elastic_utils, nan_diag


class BasicTrainer(Trainer):
    """
    Trainer for basic training loop.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        nan_diag (bool): Enable NaN/Inf forward-output probe hooks. When off (default),
            zero runtime cost. When on, hooks are attached to every module up to depth
            ``nan_diag_depth`` and dumped on non-finite-grad events (rate-limited).
        nan_diag_depth (int): Max dot-count in qualified module names to hook. 0=direct
            children only; 1 (default)=+ grandchildren (e.g. ``blocks.0``); 2=+ their
            submodules (e.g. ``blocks.0.attn``).
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.
    """

    def _fmt_wds(self, wds):
        """Format a weight-decay list with per-group labels when available."""
        if self._group_labels is not None and len(self._group_labels) == len(wds):
            return ', '.join(f'{lbl}={wd}' for lbl, wd in zip(self._group_labels, wds))
        return f'{wds[0]}' if len(wds) == 1 else str(wds)

    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Models:')
        for name, model in self.models.items():
            lines.append(f'    - {name}: {model.__class__.__name__}')
        lines.append(f'  - Dataset: {indent(str(self.dataset), 2)}')
        lines.append(f'  - Dataloader:')
        lines.append(f'    - Sampler: {self.dataloader.sampler.__class__.__name__}')
        lines.append(f'    - Num workers: {self.dataloader.num_workers}')
        lines.append(f'  - Number of steps: {self.max_steps}')
        lines.append(f'  - Number of GPUs: {self.world_size}')
        lines.append(f'  - Batch size: {self.batch_size}')
        lines.append(f'  - Batch size per GPU: {self.batch_size_per_gpu}')
        lines.append(f'  - Batch split: {self.batch_split}')
        lines.append(f'  - Optimizer: {self.optimizer.__class__.__name__}')
        lines.append(f'  - Learning rate: {self.optimizer.param_groups[0]["lr"]}')
        lines.append(f'  - Weight decay: {self._fmt_wds([pg["weight_decay"] for pg in self.optimizer.param_groups])}')
        if self.lr_scheduler_config is not None:
            lines.append(f'  - LR scheduler: {self.lr_scheduler.__class__.__name__}')
        if self.elastic_controller_config is not None:
            lines.append(f'  - Elastic memory: {indent(str(self.elastic_controller), 2)}')
        if self.grad_clip is not None:
            lines.append(f'  - Gradient clip: {indent(str(self.grad_clip), 2)}')
        lines.append(f'  - EMA rate: {self.ema_rate}')
        lines.append(f'  - FP16 mode: {self.fp16_mode}')
        return '\n'.join(lines)

    def init_models_and_more(self, **kwargs):
        """
        Initialize models and more.
        """
        if self.world_size > 1:
            # Prepare distributed data parallel
            self.training_models = {
                name: DDP(
                    model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    bucket_cap_mb=128,
                    find_unused_parameters=False
                )
                for name, model in self.models.items()
            }
        else:
            self.training_models = self.models

        # Build master params
        self.model_params = sum(
            [[p for p in unwrap_model(model).parameters() if p.requires_grad] for model in self.models.values()]
        , [])
        self.fused_master_copy_ctx = None
        self._group_indices = None
        self._group_weight_decays = None
        self._group_labels = None
        if self.fp16_mode == 'amp':
            self.master_params = self.model_params
            self.scaler = torch.GradScaler() if self.fp16_mode == 'amp' else None
        elif self.fp16_mode == 'inflat_all':
            # Per-parameter-type weight decay: split into grouped flat tensors
            wd = self.optimizer_config['args'].get('weight_decay', 0.0)
            wd_bias = self.optimizer_config['args'].get('weight_decay_bias', 0.0)
            wd_ln = self.optimizer_config['args'].get('weight_decay_layer_norm', 0.0)
            needs_grouping = (wd > 0 and (wd_bias != wd or wd_ln != wd)) or (wd == 0 and (wd_bias > 0 or wd_ln > 0))
            if needs_grouping:
                raw_groups = classify_params_by_decay_group(self.models, self.model_params, unwrap_model)
                self.master_params, self._group_indices = make_grouped_master_params(self.model_params, raw_groups)
                wd_for_raw = [wd, wd_bias, wd_ln]
                labels_for_raw = ['weight', 'bias', 'ln']
                self._group_weight_decays = [wd_for_raw[i] for i, gi in enumerate(raw_groups) if gi]
                self._group_labels = [labels_for_raw[i] for i, gi in enumerate(raw_groups) if gi]
            else:
                self.master_params = make_master_params(self.model_params)
            fused_master_copy = kwargs.get('fused_master_copy', True)
            self.fused_master_copy_ctx = init_fused_master_copy_context(
                self.model_params,
                self.master_params,
                group_indices=self._group_indices,
                enabled=fused_master_copy,
            )
            self.fp16_scale_growth = self.fp16_scale_growth
            self.log_scale = 20.0
        elif self.fp16_mode is None:
            self.master_params = self.model_params
        else:
            raise NotImplementedError(f'FP16 mode {self.fp16_mode} is not implemented.')

        # Build EMA params
        if self.is_master:
            self.ema_params = [copy.deepcopy(self.master_params) for _ in self.ema_rate]

        # Initialize optimizer
        if hasattr(torch.optim, self.optimizer_config['name']):
            if self.fp16_mode == 'inflat_all' and self._group_indices is not None:
                # inflat_all grouped: one optimizer param group per flat tensor
                optimizer_args = {k: v for k, v in self.optimizer_config['args'].items()
                                  if k not in ['weight_decay', 'weight_decay_bias', 'weight_decay_layer_norm']}
                opt_groups = [{'params': [mp], 'weight_decay': wd}
                              for mp, wd in zip(self.master_params, self._group_weight_decays)]
                self.optimizer = getattr(torch.optim, self.optimizer_config['name'])(opt_groups, **optimizer_args)
            elif 'weight_decay' in self.optimizer_config['args'] and self.optimizer_config['args']['weight_decay'] > 0:
                # Parameter grouping for per-type weight decay (non-inflat_all modes)
                weight_decay = self.optimizer_config['args']['weight_decay']
                weight_decay_bias = self.optimizer_config['args'].get('weight_decay_bias', 0.0)
                weight_decay_layer_norm = self.optimizer_config['args'].get('weight_decay_layer_norm', 0.0)
                raw_groups = classify_params_by_decay_group(self.models, self.master_params, unwrap_model)
                wd_for_raw = [weight_decay, weight_decay_bias, weight_decay_layer_norm]
                opt_groups = []
                for indices, wd in zip(raw_groups, wd_for_raw):
                    if indices:
                        opt_groups.append({'params': [self.master_params[i] for i in indices], 'weight_decay': wd})
                optimizer_args = {k: v for k, v in self.optimizer_config['args'].items()
                                  if k not in ['weight_decay', 'weight_decay_bias', 'weight_decay_layer_norm']}
                self.optimizer = getattr(torch.optim, self.optimizer_config['name'])(opt_groups, **optimizer_args)
            else:
                # Use simple approach (original TRELLIS behavior)
                # Filter out custom weight decay parameters for backward compatibility
                optimizer_args = {k: v for k, v in self.optimizer_config['args'].items() 
                                  if k not in ['weight_decay_bias', 'weight_decay_layer_norm']}
                self.optimizer = getattr(torch.optim, self.optimizer_config['name'])(self.master_params, **optimizer_args)
        else:
            self.optimizer = globals()[self.optimizer_config['name']](self.master_params, **self.optimizer_config['args'])
        
        # Initalize learning rate scheduler
        if self.lr_scheduler_config is not None:
            if hasattr(torch.optim.lr_scheduler, self.lr_scheduler_config['name']):
                self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_config['name'])(self.optimizer, **self.lr_scheduler_config['args'])
            else:
                self.lr_scheduler = globals()[self.lr_scheduler_config['name']](self.optimizer, **self.lr_scheduler_config['args'])

        # Initialize elastic memory controller. ``unwrap_model`` peels off DDP / torch.compile wrappers
        # so the isinstance check sees the actual module hierarchy (their wrappers are not ElasticModuleMixin subclasses).
        if self.elastic_controller_config is not None:
            assert any([isinstance(unwrap_model(model), (elastic_utils.ElasticModule, elastic_utils.ElasticModuleMixin)) for model in self.models.values()]), \
                'No elastic module found in models, please inherit from ElasticModule or ElasticModuleMixin'
            self.elastic_controller = getattr(elastic_utils, self.elastic_controller_config['name'])(**self.elastic_controller_config['args'])
            for model in self.models.values():
                unwrapped = unwrap_model(model)
                if isinstance(unwrapped, (elastic_utils.ElasticModule, elastic_utils.ElasticModuleMixin)):
                    unwrapped.register_memory_controller(self.elastic_controller)

        # Initialize gradient clipper
        if self.grad_clip is not None:
            if isinstance(self.grad_clip, (float, int)):
                self.grad_clip = float(self.grad_clip)
            else:
                self.grad_clip = getattr(grad_clip_utils, self.grad_clip['name'])(**self.grad_clip['args'])

        # NaN/Inf diagnostics: opt-in via `nan_diag=True` in trainer config.
        # Zero runtime cost when disabled (returns None, no hooks registered).
        self.nan_diag = None
        if self.nan_diag_enabled and self.is_master:
            self.nan_diag = nan_diag.NaNDiagnostic(self.training_models, depth=self.nan_diag_depth)

    def _master_params_to_state_dicts(self, master_params):
        """
        Convert master params to dict of state_dicts.
        """
        if self.fp16_mode == 'inflat_all':
            master_params = unflatten_master_params(self.model_params, master_params, group_indices=self._group_indices)
        state_dicts = {name: unwrap_model(model).state_dict() for name, model in self.models.items()}
        master_params_names = sum(
            [[(name, n) for n, p in unwrap_model(model).named_parameters() if p.requires_grad] for name, model in self.models.items()]
        , [])
        for i, (model_name, param_name) in enumerate(master_params_names):
            state_dicts[model_name][param_name] = master_params[i]
        return state_dicts

    def _state_dicts_to_master_params(self, master_params, state_dicts):
        """
        Convert a state_dict to master params.
        """
        master_params_names = sum(
            [[(name, n) for n, p in unwrap_model(model).named_parameters() if p.requires_grad] for name, model in self.models.items()]
        , [])
        params = [state_dicts[name][param_name] for name, param_name in master_params_names]
        if self.fp16_mode == 'inflat_all':
            model_params_to_master_params(params, master_params, group_indices=self._group_indices)
        else:
            for i, param in enumerate(params):
                master_params[i].data.copy_(param.data)

    def load(self, load_dir, step=0):
        """
        Load a checkpoint.
        Should be called by all processes.
        """
        if self.is_master:
            print(f'\nLoading checkpoint from step {step}...', end='')
            
        model_ckpts = {}
        for name, model in self.models.items():
            model_ckpt_path = os.path.join(load_dir, 'ckpts', f'{name}_step{step:07d}.pt')
            model_ckpt = load_ckpt_dist(model_ckpt_path, map_location=self.device)
            model_ckpts[name] = model_ckpt
            unwrap_model(model).load_state_dict(model_ckpt)
            if self.fp16_mode == 'inflat_all':
                unwrap_model(model).convert_to_fp16()
        self._state_dicts_to_master_params(self.master_params, model_ckpts)
        del model_ckpts

        if self.is_master:
            for i, ema_rate in enumerate(self.ema_rate):
                ema_ckpts = {}
                for name, model in self.models.items():
                    ema_ckpt_path = os.path.join(load_dir, 'ckpts', f'{name}_ema{ema_rate}_step{step:07d}.pt')
                    ema_ckpt = torch.load(ema_ckpt_path, map_location=self.device, weights_only=False)
                    ema_ckpts[name] = ema_ckpt
                self._state_dicts_to_master_params(self.ema_params[i], ema_ckpts)
                del ema_ckpts
        
        misc_ckpt_path = os.path.join(load_dir, 'ckpts', f'misc_step{step:07d}.pt')
        misc_ckpt = load_ckpt_dist(misc_ckpt_path, map_location=torch.device('cpu'))
        # Backward compat: remap optimizer state if param group count changed
        opt_ckpt = misc_ckpt['optimizer']
        # Capture weight_decay from the raw checkpoint so we can announce
        # changes even when remap has already rewritten them.
        ckpt_wds = [pg['weight_decay'] for pg in opt_ckpt['param_groups']]
        num_ckpt_groups = len(opt_ckpt['param_groups'])
        num_cur_groups = len(self.optimizer.param_groups)
        if num_ckpt_groups != num_cur_groups:
            if self._group_indices is not None and num_ckpt_groups == 1:
                param_numels = [p.numel() for p in self.model_params]
                opt_ckpt = remap_optimizer_state_single_to_grouped(
                    opt_ckpt, param_numels, self._group_indices, self._group_weight_decays)
            else:
                raise ValueError(
                    f"Checkpoint has {num_ckpt_groups} optimizer param groups but "
                    f"current model expects {num_cur_groups}. Cannot auto-migrate.")
        self.optimizer.load_state_dict(opt_ckpt)
        # Re-apply weight_decay from current config: load_state_dict restores the
        # checkpoint's per-group values, silently overriding any config change.
        # Also covers the 1→N remap path by comparing against the raw checkpoint.
        desired_wds = (self._group_weight_decays
                       if self._group_weight_decays is not None
                       else [self.optimizer_config['args'].get('weight_decay', 0.0)])
        if ckpt_wds != desired_wds:
            if self.is_master:
                print(f"[resume] weight_decay changed: {self._fmt_wds(ckpt_wds)} -> {self._fmt_wds(desired_wds)}")
            for pg, wd in zip(self.optimizer.param_groups, desired_wds):
                pg['weight_decay'] = wd
        # Same pattern as weight_decay: load_state_dict restored the checkpoint's eps, silently
        # overriding any config change. Re-apply from config.
        desired_eps = self.optimizer_config['args'].get('eps')
        ckpt_eps = [pg.get('eps') for pg in opt_ckpt['param_groups']]
        if desired_eps is not None and any(e != desired_eps for e in ckpt_eps):
            if self.is_master:
                print(f"[resume] eps changed: {ckpt_eps} -> {desired_eps}")
            for pg in self.optimizer.param_groups:
                pg['eps'] = desired_eps
        self.step = misc_ckpt['step']
        self.data_sampler.load_state_dict(misc_ckpt['data_sampler'])
        if self.fp16_mode == 'amp':
            self.scaler.load_state_dict(misc_ckpt['scaler'])
        elif self.fp16_mode == 'inflat_all':
            self.log_scale = misc_ckpt['log_scale']
        if self.lr_scheduler_config is not None:
            sched_ckpt = misc_ckpt['lr_scheduler']
            if self._group_indices is not None and num_ckpt_groups == 1:
                sched_ckpt = remap_lr_scheduler_state_single_to_grouped(
                    sched_ckpt, num_cur_groups)
            self.lr_scheduler.load_state_dict(sched_ckpt)
        if self.elastic_controller_config is not None:
            self.elastic_controller.load_state_dict(misc_ckpt['elastic_controller'])
        if self.grad_clip is not None and not isinstance(self.grad_clip, float):
            self.grad_clip.load_state_dict(misc_ckpt['grad_clip'])
        del misc_ckpt

        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print(' Done.')

        if self.world_size > 1:
            self.check_ddp()

    def save(self):
        """
        Save a checkpoint.
        Should be called only by the rank 0 process.
        """
        assert self.is_master, 'save() should be called only by the rank 0 process.'
        print(f'\nSaving checkpoint at step {self.step}...', end='')
        
        model_ckpts = self._master_params_to_state_dicts(self.master_params)
        for name, model_ckpt in model_ckpts.items():
            torch.save(model_ckpt, os.path.join(self.output_dir, 'ckpts', f'{name}_step{self.step:07d}.pt'))
        
        for i, ema_rate in enumerate(self.ema_rate):
            ema_ckpts = self._master_params_to_state_dicts(self.ema_params[i])
            for name, ema_ckpt in ema_ckpts.items():
                torch.save(ema_ckpt, os.path.join(self.output_dir, 'ckpts', f'{name}_ema{ema_rate}_step{self.step:07d}.pt'))

        misc_ckpt = {
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'data_sampler': self.data_sampler.state_dict(),
        }
        if self.fp16_mode == 'amp':
            misc_ckpt['scaler'] = self.scaler.state_dict()
        elif self.fp16_mode == 'inflat_all':
            misc_ckpt['log_scale'] = self.log_scale
        if self.lr_scheduler_config is not None:
            misc_ckpt['lr_scheduler'] = self.lr_scheduler.state_dict()
        if self.elastic_controller_config is not None:
            misc_ckpt['elastic_controller'] = self.elastic_controller.state_dict()
        if self.grad_clip is not None and not isinstance(self.grad_clip, float):
            misc_ckpt['grad_clip'] = self.grad_clip.state_dict()
        torch.save(misc_ckpt, os.path.join(self.output_dir, 'ckpts', f'misc_step{self.step:07d}.pt'))
        print(' Done.')

    def finetune_from(self, finetune_ckpt):
        """
        Finetune from a checkpoint.
        Should be called by all processes.
        """
        if self.is_master:
            print('\nFinetuning from:')
            for name, path in finetune_ckpt.items():
                print(f'  - {name}: {path}')
        
        model_ckpts = {}
        for name, model in self.models.items():
            model_state_dict = unwrap_model(model).state_dict()
            if name in finetune_ckpt:
                mismatch_strategy = 'skip'
                if isinstance(finetune_ckpt[name], dict):
                    finetune_ckpt_path = finetune_ckpt[name]['path']
                    if 'mismatch_strategy' in finetune_ckpt[name]:
                        mismatch_strategy = finetune_ckpt[name]['mismatch_strategy']
                else:
                    finetune_ckpt_path = finetune_ckpt[name]
                    if not isinstance(finetune_ckpt_path, str):
                        raise ValueError(f'finetune_ckpt[{name}] must be a string or a dict with "path" key.')
                model_ckpt = load_ckpt_dist(finetune_ckpt_path, map_location=self.device)
                shared_keys = set(model_ckpt.keys()) & set(model_state_dict.keys())
                for k in shared_keys:
                    if model_ckpt[k].shape != model_state_dict[k].shape:
                        if self.is_master:
                            print(f'Warning: {k} shape mismatch, {model_ckpt[k].shape} vs {model_state_dict[k].shape}, using {mismatch_strategy} strategy.')
                        if mismatch_strategy == 'skip':
                            model_ckpt[k] = model_state_dict[k]
                        elif mismatch_strategy == 'partial':
                            model_ckpt[k] = handle_partial_mismatch(model_ckpt[k], model_state_dict[k])
                        elif mismatch_strategy == 'zeropad':
                            model_ckpt[k] = handle_zeropad_mismatch(model_ckpt[k], model_state_dict[k])
                        elif mismatch_strategy == 'error':
                            raise ValueError(f'{k} shape mismatch, {model_ckpt[k].shape} vs {model_state_dict[k].shape}.')
                        else:
                            raise ValueError(f'mismatch behaviour {mismatch_strategy} is not implemented.')
                incompatible = unwrap_model(model).load_state_dict(model_ckpt, strict=False)
                if self.is_master:
                    for k in incompatible.missing_keys:
                        print(f'Warning: Model {name} missing key: {k}')
                    for k in incompatible.unexpected_keys:
                        print(f'Warning: Model {name} unexpected key: {k}')
                model_ckpts[name] = unwrap_model(model).state_dict()
                if self.fp16_mode == 'inflat_all':
                    unwrap_model(model).convert_to_fp16()
            else:
                if self.is_master:
                    print(f'Warning: {name} not found in finetune_ckpt, skipped.')
                model_ckpts[name] = model_state_dict
        self._state_dicts_to_master_params(self.master_params, model_ckpts)
        del model_ckpts

        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print('Done.')

        if self.world_size > 1:
            self.check_ddp()

    def update_ema(self):
        """
        Update exponential moving average.
        Should only be called by the rank 0 process.
        """
        assert self.is_master, 'update_ema() should be called only by the rank 0 process.'
        for i, ema_rate in enumerate(self.ema_rate):
            for master_param, ema_param in zip(self.master_params, self.ema_params[i]):
                ema_param.detach().mul_(ema_rate).add_(master_param, alpha=1.0 - ema_rate)

    def check_ddp(self):
        """
        Check if DDP is working properly.
        Should be called by all process.
        """
        if self.is_master:
            print('\nPerforming DDP check...')

        if self.is_master:
            print('Checking if parameters are consistent across processes...')
        dist.barrier()
        try:
            for p in self.master_params:
                # split to avoid OOM
                for i in range(0, p.numel(), 10000000):
                    sub_size = min(10000000, p.numel() - i)
                    sub_p = p.detach().view(-1)[i:i+sub_size]
                    # gather from all processes
                    sub_p_gather = [torch.empty_like(sub_p) for _ in range(self.world_size)]
                    dist.all_gather(sub_p_gather, sub_p)
                    # check if equal
                    assert all([torch.equal(sub_p, sub_p_gather[i]) for i in range(self.world_size)]), 'parameters are not consistent across processes'
        except AssertionError as e:
            if self.is_master:
                print(f'\n\033[91mError: {e}\033[0m')
                print('DDP check failed.')
            raise e

        dist.barrier()
        if self.is_master:
            print('Done.')

    def _all_grads_finite(self, params):
        """
        Check gradient finiteness with a single device-to-host sync.
        Use fused foreach non-finite check to avoid per-parameter Python dispatch.
        """
        grads = [p.grad for p in params]

        # Fused multi-tensor non-finite check (same primitive used by GradScaler).
        # Keep scale=1 so grads are unchanged while we still get one aggregated flag.
        if hasattr(torch, "_amp_foreach_non_finite_check_and_unscale_"):
            found_inf = torch.zeros((), dtype=torch.float32, device=self.device)
            inv_scale = torch.ones((), dtype=torch.float32, device=self.device)
            torch._amp_foreach_non_finite_check_and_unscale_(grads, found_inf, inv_scale)
            return bool(found_inf.item() == 0.0)

        # Fallback for environments without the fused primitive.
        all_finite = torch.ones((), dtype=torch.bool, device=self.device)
        for g in grads:
            all_finite.logical_and_(torch.isfinite(g).all())
        return bool(all_finite.item())

    def run_step(self, data_list):
        """
        Run a training step.
        """
        # Align logging with Trainer.run(): `self.step` is incremented after `run_step` returns.
        # So the "current" step number from Trainer.run()'s perspective is `self.step + 1`.
        should_log = self.is_master and ((self.step + 1) % self.i_log == 0 or (self.step + 1) % self.i_steplog == 0)

        step_log = {'loss': {}, 'status': {}} if should_log else None
        amp_context = partial(torch.autocast, device_type='cuda') if self.fp16_mode == 'amp' else nullcontext
        elastic_controller_context = self.elastic_controller.record if self.elastic_controller_config is not None else nullcontext

        # Train
        losses = [] if should_log else None
        statuses = [] if should_log else None
        elastic_controller_logs = [] if (should_log and self.elastic_controller_config is not None) else None
        zero_grad(self.model_params, fused_copy_ctx=self.fused_master_copy_ctx)
        if self.nan_diag is not None:
            self.nan_diag.reset()
        for i, mb_data in enumerate(data_list):
            ## sync at the end of each batch split
            sync_contexts = [self.training_models[name].no_sync for name in self.training_models] if i != len(data_list) - 1 and self.world_size > 1 else [nullcontext]
            with nested_contexts(*sync_contexts), elastic_controller_context():
                with amp_context():
                    loss, status = self.training_losses(**mb_data)
                    l = loss['loss'] / len(data_list)
                ## backward
                if self.fp16_mode == 'amp':
                    self.scaler.scale(l).backward()
                elif self.fp16_mode == 'inflat_all':
                    scaled_l = l * (2 ** self.log_scale)
                    scaled_l.backward()
                else:
                    l.backward()
            ## log (only on logging steps to avoid frequent GPU->CPU sync from `.item()`)
            if should_log:
                losses.append(dict_foreach(loss, lambda x: x.item() if isinstance(x, torch.Tensor) else x))
                statuses.append(dict_foreach(status, lambda x: x.item() if isinstance(x, torch.Tensor) else x))
                if self.elastic_controller_config is not None:
                    assert elastic_controller_logs is not None
                    elastic_controller_logs.append(self.elastic_controller.log())
        ## gradient clip
        if self.grad_clip is not None:
            if self.fp16_mode == 'amp':
                self.scaler.unscale_(self.optimizer)
            elif self.fp16_mode == 'inflat_all':
                model_grads_to_master_grads(
                    self.model_params,
                    self.master_params,
                    group_indices=self._group_indices,
                    fused_copy_ctx=self.fused_master_copy_ctx,
                )
                scale_inv = 1.0 / (2 ** self.log_scale)
                for mp in self.master_params:
                    mp.grad.mul_(scale_inv)
            if isinstance(self.grad_clip, float):
                grad_norm = torch.nn.utils.clip_grad_norm_(self.master_params, self.grad_clip)
            else:
                grad_norm = self.grad_clip(self.master_params)
            if should_log:
                gn = grad_norm.item()
                if math.isfinite(gn):
                    statuses[-1]['grad_norm'] = gn
        ## step
        did_step = False
        if self.fp16_mode == 'amp':
            prev_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # If inf/NaN gradients were found, GradScaler skips optimizer.step()
            # and reduces the scale. Only step LR scheduler when an optimizer step happened.
            did_step = self.scaler.get_scale() >= prev_scale
        elif self.fp16_mode == 'inflat_all':
            prev_scale = 2 ** self.log_scale
            # Gate the step on both fp16 model grads (overflow detector; drives halving) and
            # fp32 master grads (grad_clip can inject NaN via a NaN clip_coef, which the fp16
            # check cannot see). Without the master check, one such step poisons master
            # weights, EMA, and AdamW (m, v) -- forward then NaNs forever regardless of scale.
            fp16_ok = self._all_grads_finite(self.model_params)
            step_ok = fp16_ok and (self.grad_clip is None or self._all_grads_finite(self.master_params))
            if step_ok:
                if self.grad_clip is None:
                    model_grads_to_master_grads(
                        self.model_params, self.master_params,
                        group_indices=self._group_indices, fused_copy_ctx=self.fused_master_copy_ctx,
                    )
                    scale_inv = 1.0 / (2 ** self.log_scale)
                    for mp in self.master_params:
                        mp.grad.mul_(scale_inv)
                self.optimizer.step()
                master_params_to_model_params(
                    self.model_params, self.master_params,
                    group_indices=self._group_indices, fused_copy_ctx=self.fused_master_copy_ctx,
                )
                self.log_scale += self.fp16_scale_growth
                did_step = True
            else:
                # Halve only on fp16 overflow; a master-only NaN is a grad_clip artefact that
                # halving won't help. Floor at -20: below that fp16 grads underflow uniformly
                # and 1/(2**log_scale) would ZeroDivisionError.
                if not fp16_ok:
                    self.log_scale = max(self.log_scale - 1, -20)
                if self.nan_diag is not None:
                    reason = "fp16-overflow" if not fp16_ok else "master-grad-NaN"
                    self.nan_diag.dump(step=self.step + 1, tag=f"inflat_all log_scale={self.log_scale:.2f} {reason}")
                if not fp16_ok and self.log_scale <= -20:
                    raise RuntimeError(
                        "log_scale hit floor -20: fp16 dynamic range exhausted. Training diverged "
                        "-- restore an earlier checkpoint and/or switch to BF16."
                    )
        else:
            prev_scale = 1.0
            if self._all_grads_finite(self.model_params):
                self.optimizer.step()
                did_step = True
            else:
                print('\nWarning: NaN detected in gradients. Skipping update.')
                if self.nan_diag is not None:
                    self.nan_diag.dump(step=self.step + 1, tag="fp32")
        ## adjust learning rate
        if self.lr_scheduler_config is not None:
            if should_log:
                # Log the LR actually used for this (potential) update
                statuses[-1]['lr'] = self.optimizer.param_groups[0]['lr']
            if did_step:
                self.lr_scheduler.step()

        # Logs (only on logging steps)
        if should_log:
            step_log['loss'] = dict_reduce(losses, lambda x: np.mean(x))
            step_log['status'] = dict_reduce(
                statuses,
                lambda x: np.mean(x),
                special_func={'min': lambda x: np.min(x), 'max': lambda x: np.max(x)},
            )
            if self.elastic_controller_config is not None:
                assert elastic_controller_logs is not None
                step_log['elastic'] = dict_reduce(elastic_controller_logs, lambda x: np.mean(x))
            if self.grad_clip is not None:
                step_log['grad_clip'] = self.grad_clip if isinstance(self.grad_clip, float) else self.grad_clip.log()
            
        # Check grad and norm of each param
        if should_log and self.log_param_stats:
            param_norms = {}
            param_grads = {}
            for name, param in self.backbone.named_parameters():
                if param.requires_grad:
                    param_norms[name] = param.norm().item()
                    if param.grad is not None and torch.isfinite(param.grad).all():
                        param_grads[name] = param.grad.norm().item() / prev_scale
            step_log['param_norms'] = param_norms
            step_log['param_grads'] = param_grads

        # Update exponential moving average
        if self.is_master:
            self.update_ema()

        return step_log
