from typing import *
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict

from ..basic import BasicTrainer
from ...pipelines import samplers
from ...utils.general_utils import dict_reduce
from ...utils.data_utils import (
    ResumableSampler,
    VarViewBatchSampler,
    cycle,
    recursive_to_device,
)
from .mixins.flow_matching_core import FlowMatchingCoreMixin
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.image_conditioned import MultiVisualImageConditionedMixin


class FlowMatchingTrainer(FlowMatchingCoreMixin, BasicTrainer):
    """
    Trainer for diffusion model with flow matching objective.
    
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
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
    """
    def __init__(
        self,
        *args,
        t_schedule: dict = {
            'name': 'logitNormal',
            'args': {
                'mean': 0.0,
                'std': 1.0,
            }
        },
        sigma_min: float = 1e-5,
        **kwargs
    ):
        super().__init__(*args, t_schedule=t_schedule, sigma_min=sigma_min, **kwargs)

    def get_sampler(self, **kwargs) -> samplers.FlowEulerSampler:
        """
        Get the sampler for the diffusion process.
        """
        return samplers.FlowEulerSampler(self.sigma_min)
    
    def vis_cond(self, **kwargs):
        """
        Visualize the conditioning data.
        """
        return {}

    def training_losses(
        self,
        x_0: torch.Tensor,
        cond=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        noise = torch.randn_like(x_0)

        t = self.sample_t(x_0.shape[0], x_0.device)
        x_t = self.diffuse(x_0, t, noise)
        cond = self.get_training_cond(cond, **kwargs)

        pred = self.training_models['denoiser'](x_t, t * 1000, cond, **kwargs)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)
        terms = edict()

        if hasattr(self.dataset, 'suv_latent_channels'):
            x_pred = pred[:, :-self.dataset.suv_latent_channels]
            P_pred = pred[:, -self.dataset.suv_latent_channels:]
            x_target = target[:, :-self.dataset.suv_latent_channels]
            P_target = target[:, -self.dataset.suv_latent_channels:]
            mse_occ_per_instance = torch.square(x_pred - x_target).flatten(start_dim=1).mean(dim=1)
            mse_pose_per_instance = torch.square(P_pred - P_target).flatten(start_dim=1).mean(dim=1)
            terms["mse_occ"] = mse_occ_per_instance.mean()
            terms["mse_pose"] = mse_pose_per_instance.mean()
            
        mse_per_instance = torch.square(pred - target).flatten(start_dim=1).mean(dim=1)
        terms["mse"] = mse_per_instance.mean()
        terms["loss"] = terms["mse"]

        # log loss with time bins
        mse_per_instance = mse_per_instance.detach().cpu().numpy()
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}

        return terms, {}
    
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        # inference
        sampler = self.get_sampler()
        sample_gt = []
        sample = []
        cond_vis = []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            data = {k: v[:batch].cuda() if isinstance(v, torch.Tensor) else v[:batch] for k, v in data.items()}
            noise = torch.randn_like(data['x_0'])
            sample_gt.append(data['x_0'])
            cond_vis.append(self.vis_cond(**data))
            del data['x_0']
            args = self.get_inference_cond(**data)
            res = sampler.sample(
                self.models['denoiser'],
                noise=noise,
                **args,
                steps=50, cfg_strength=3.0, verbose=verbose,
            )
            sample.append(res.samples)

        sample_gt = torch.cat(sample_gt, dim=0)
        sample = torch.cat(sample, dim=0)
        sample_dict = {
            'sample_gt': {'value': sample_gt, 'type': 'sample'},
            'sample': {'value': sample, 'type': 'sample'},
        }
        sample_dict.update(dict_reduce(cond_vis, None, {
            'value': lambda x: torch.cat(x, dim=0),
            'type': lambda x: x[0],
        }))
        
        return sample_dict

    
class FlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, FlowMatchingTrainer):
    """
    Trainer for diffusion model with flow matching objective and classifier-free guidance.
    
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
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
    """
    pass


class MultiPoseFlowMatchingCFGTrainer(FlowMatchingCFGTrainer):
    """
    Trainer for multi-pose diffusion model with flow matching objective and classifier-free guidance.

    Args:
        t_coupling: How to couple shape/pose timesteps during training.
            - str: 'shared' | 'shape_pose' | 'per_pose' (deterministic).
            - dict: per-mode probabilities, e.g. {'shared': 0.5, 'per_pose': 0.5}.
              Missing keys are treated as 0; values are auto-normalized.
              Mode is sampled per-instance each step.
        var_view_microbatch: Optional dict ``{V: B_per_microbatch}`` enabling
            variable-view gradient accumulation. One micro-batch per entry per
            macro-step, with ``B`` instances at ``V`` views each. When set,
            ``batch_size_per_gpu`` and ``batch_split`` are derived from this dict
            and any user-provided values are ignored. The dataset must be
            configured with ``num_pose_views >= max(V)`` so the loader can serve
            the largest case (and ``num_pose_views`` keeps governing eval-time V).
    """
    MODES = ('shared', 'shape_pose', 'per_pose')

    def __init__(
        self,
        *args,
        t_coupling: Union[str, Dict[str, float]] = 'shared',
        var_view_microbatch: Optional[Dict[Union[int, str], int]] = None,
        **kwargs,
    ):
        # Parse var_view_microbatch BEFORE super().__init__ so the base Trainer's
        # batch_size_per_gpu / batch_split divisibility asserts see the overrides.
        self.var_view_microbatch: Optional[Dict[int, int]] = None
        if var_view_microbatch is not None:
            self.var_view_microbatch = {int(v): int(b) for v, b in var_view_microbatch.items()}
            num_mb = len(self.var_view_microbatch)
            total_instances = sum(self.var_view_microbatch.values())
            if kwargs.get('batch_size_per_gpu') is not None or kwargs.get('batch_size') is not None:
                print(f"[var_view_microbatch] ignoring batch_size/batch_size_per_gpu; "
                      f"deriving batch_size_per_gpu={total_instances}, batch_split={num_mb} "
                      f"from {self.var_view_microbatch}")
            kwargs['batch_size'] = None
            kwargs['batch_size_per_gpu'] = total_instances
            kwargs['batch_split'] = num_mb

            # Switch dynamo's DDP graph-split implementation. The default
            # "ddp_optimizer" splitter mishandles the shape variation that
            # variable V introduces, raising "'int' has no attribute 'meta'"
            # in AOT autograd's pre_compile. "python_reducer" is a newer
            # split implementation that usually sidesteps that bug while
            # keeping the forward/allreduce overlap. Fall back to False
            # (single graph, no bucket overlap) if python_reducer still fails.
            current = torch._dynamo.config.optimize_ddp
            if current in (True, "ddp_optimizer"):
                print("[var_view_microbatch] torch._dynamo.config.optimize_ddp: "
                      f"{current!r} -> 'python_reducer' (avoids DDP+compile bug "
                      "under variable V; keeps forward/allreduce overlap)")
                torch._dynamo.config.optimize_ddp = "python_reducer"

        super().__init__(*args, **kwargs)

        if isinstance(t_coupling, str):
            t_coupling = {t_coupling: 1.0}
        assert set(t_coupling) <= set(self.MODES), \
            f"t_coupling keys must be in {self.MODES}, got {set(t_coupling)}"
        probs = [t_coupling.get(m, 0.0) for m in self.MODES]
        total = sum(probs)
        assert total > 0, "t_coupling probabilities must sum to > 0"
        self.t_coupling_probs = torch.tensor([p / total for p in probs])

        if self.var_view_microbatch is not None:
            assert hasattr(self.dataset, 'num_pose_views'), \
                "var_view_microbatch requires a MultiViewMixin-based dataset"
            max_v = max(self.var_view_microbatch)
            total_views = self.dataset.num_ref_views + self.dataset.num_cond_views
            assert max_v <= self.dataset.num_pose_views <= total_views, (
                f"max V ({max_v}) <= dataset.num_pose_views ({self.dataset.num_pose_views}) "
                f"<= total_views ({total_views}) must hold"
            )

    def prepare_dataloader(self, num_workers=8, **kwargs):
        if self.var_view_microbatch is None:
            return super().prepare_dataloader(num_workers=num_workers, **kwargs)
        self.data_sampler = ResumableSampler(self.dataset, shuffle=True)
        batch_sampler = VarViewBatchSampler(self.data_sampler, self.var_view_microbatch)
        self.dataloader = DataLoader(
            self.dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )
        self.data_iterator = cycle(self.dataloader)

    def load_data(self):
        if self.var_view_microbatch is None:
            return super().load_data()

        def _fetch_macro_step():
            return [
                recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
                for _ in range(self.batch_split)
            ]

        if self.prefetch_data:
            if self._data_prefetched is None:
                self._data_prefetched = _fetch_macro_step()
            data_list = self._data_prefetched
            self._data_prefetched = _fetch_macro_step()
        else:
            data_list = _fetch_macro_step()
        return data_list

    def get_sampler(self, **kwargs) -> samplers.MultiPoseFlowEulerCfgSampler:
        """
        Get the sampler for the diffusion process.
        """
        return samplers.MultiPoseFlowEulerCfgSampler(self.sigma_min)

    def training_losses(
        self,
        x_0: torch.Tensor,
        P_0: torch.Tensor,
        cond=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            P_0: The [N x N_views x C_pose x ...] tensor of noiseless poses.
            cond: The [N x N_views x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        assert P_0.ndim == x_0.ndim + 1, f"P_0 must have one more dimension than x_0, got {P_0.ndim} and {x_0.ndim}"

        x_noise = torch.randn_like(x_0)
        P_noise = torch.randn_like(P_0)

        N, V = x_0.shape[0], P_0.shape[1]
        device = x_0.device

        t_x        = self.sample_t(N, device)                  # [N]
        t_P_shared = self.sample_t(N, device)                  # [N]
        t_P_per    = self.sample_t(N * V, device).view(N, V)   # [N, V]

        # Canonical mode order: 'shared', 'shape_pose', 'per_pose'
        candidates = torch.stack([
            t_x.unsqueeze(1).expand(N, V),
            t_P_shared.unsqueeze(1).expand(N, V),
            t_P_per,
        ], dim=0)  # [3, N, V]

        probs = self.t_coupling_probs.to(device)
        mode_idx = torch.multinomial(probs, num_samples=N, replacement=True)  # [N]
        batch_idx = torch.arange(N, device=device)
        t_P = candidates[mode_idx, batch_idx]  # [N, V]

        x_t = self.diffuse(x_0, t_x, x_noise)
        P_t = self.diffuse(P_0, t_P, P_noise)
        cond = self.get_training_cond(cond, **kwargs)

        x_pred, P_pred = self.training_models['denoiser'](x_t, P_t, t_x * 1000, t_P * 1000, cond, **kwargs)
        assert x_pred.shape == x_noise.shape == x_0.shape
        assert P_pred.shape == P_noise.shape == P_0.shape
        x_target = self.get_v(x_0, x_noise, t_x)
        P_target = self.get_v(P_0, P_noise, t_P)

        terms = edict()
        mse_occ_per_instance = torch.square(x_pred - x_target).flatten(start_dim=1).mean(dim=1)
        mse_pose_per_instance = torch.square(P_pred - P_target).flatten(start_dim=1).mean(dim=1)
        mse_per_instance = (mse_occ_per_instance + mse_pose_per_instance) / 2
        terms["mse_occ"] = mse_occ_per_instance.mean()
        terms["mse_pose"] = mse_pose_per_instance.mean()
        terms["mse"] = mse_per_instance.mean()
        terms["loss"] = terms["mse"]

        # log loss with time bins
        mse_per_instance = mse_per_instance.detach().cpu().numpy()
        time_bin = np.digitize(t_x.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}

        return terms, {}

    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        # inference
        sampler = self.get_sampler()
        sample_gt = []
        sample = []
        cond_vis = []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            data = {k: v[:batch].cuda() if isinstance(v, torch.Tensor) else v[:batch] for k, v in data.items()}
            x_noise = torch.randn_like(data['x_0'])
            P_noise = torch.randn_like(data['P_0'])
            sample_gt.append(data['x_0'])
            cond_vis.append(self.vis_cond(**data))
            del data['x_0'], data['P_0']
            args = self.get_inference_cond(**data)
            res = sampler.sample(
                self.models['denoiser'],
                noise=(x_noise, P_noise),
                **args,
                steps=25, cfg_strength=3.0, verbose=verbose,
            )
            sample.append(res.samples[0])

        sample_gt = torch.cat(sample_gt, dim=0)
        sample = torch.cat(sample, dim=0)
        sample_dict = {
            'sample_gt': {'value': sample_gt, 'type': 'sample'},
            'sample': {'value': sample, 'type': 'sample'},
        }
        sample_dict.update(dict_reduce(cond_vis, None, {
            'value': lambda x: torch.cat(x, dim=0),
            'type': lambda x: x[0],
        }))
        
        return sample_dict


class VisualImageConditionedMultiPoseFlowMatchingCFGTrainer(MultiVisualImageConditionedMixin, MultiPoseFlowMatchingCFGTrainer):
    """
    Trainer for visual + DINOv2 multi-pose diffusion model. Feeds denoiser
    cond = {'visual_cond': [B,V,C,H,W], 'dino_cond': [B,V,L,C_dino]}.
    """
    pass
