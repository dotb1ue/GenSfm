from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin, MultiPoseGuidanceIntervalSamplerMixin
from ...utils.voxel_utils import project_cv


class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart(self, x_t, t, v):
        assert x_t.shape == v.shape
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0

    def _v_to_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        return eps

    def _batchify_t(self, t, batch_size, device):
        if isinstance(t, Iterable):
            t = [[1000 * t_i for t_i in t]]
        elif isinstance(t, float):
            t = [1000 * t]
        else:
            raise ValueError(f"Unknown type of t: {type(t)}")
        return torch.tensor(t * batch_size, device=device, dtype=torch.float32)

    def _batchify_cond(self, cond, batch_size):
        if cond is not None and batch_size > 1:
            if isinstance(cond, torch.Tensor) and cond.shape[0] == 1:
                cond = cond.repeat(batch_size, *([1] * (len(cond.shape) - 1)))
            elif isinstance(cond, dict):
                cond = {k: (
                    v.repeat(batch_size, *([1] * (len(v.shape) - 1)))
                    if isinstance(v, torch.Tensor) and v.shape[0] == 1 else v
                ) for k, v in cond.items()}
        return cond

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = self._batchify_t(t, x_t.shape[0], x_t.device)
        cond = self._batchify_cond(cond, x_t.shape[0])
        return model(x_t, t, cond, **kwargs)

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            sample: x_{t-1}.
            out: a dict containing the following
            - 'pred_x_t': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_v = self._inference_model(model, x_t, t, cond=cond, **kwargs)
        pred_x_0 = self._v_to_xstart(x_t=x_t, t=t, v=pred_v)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return pred_x_prev, edict({"pred_x_t": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None})
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            sample, out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            for key in out.keys():
                ret.setdefault(key, []).append(out[key])
        ret.samples = sample
        return ret


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, **kwargs)


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)


class MultiPoseFlowEulerSampler(FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling for multi-pose flow.
    """
    def _inference_model(self, model, x_t_and_P_t, t_x_and_t_P, cond=None, **kwargs):
        """Run one denoising step, caching ``fused_cond`` across steps.

        For models exposing ``encode_cond`` (output depends only on cond, not on
        timestep), we call it once per distinct (cond tensor, batch size) via a
        size-2 LRU. Size 2 covers CFG (alternating cond / neg_cond). The cache
        key is the pre-batchify cond identity plus the post-batchify batch size,
        so that a B=1 cond broadcast to B>1 noise gets a correctly shaped fused
        state. The cache is cleared at sample() boundaries (see sample() override)
        so long-lived sampler instances do not retain large fused tensors.
        """
        x_t, P_t = x_t_and_P_t
        t_x, t_P = t_x_and_t_P
        assert x_t.shape[0] == P_t.shape[0]
        B = x_t.shape[0]
        t_x = self._batchify_t(t_x, B, x_t.device)
        t_P = self._batchify_t(t_P, B, P_t.device)
        cond_b = self._batchify_cond(cond, B)

        if cond is not None and hasattr(model, "encode_cond"):
            cache = getattr(self, "_fused_cond_cache", ())
            fused = next((v for (c_key, b_key), v in cache
                          if c_key is cond and b_key == B), None)
            if fused is None:
                fused = model.encode_cond(cond_b)
                self._fused_cond_cache = (((cond, B), fused),) + cache[:1]  # LRU-2
            return model(x_t, P_t, t_x, t_P, None, fused_cond=fused, **kwargs)

        return model(x_t, P_t, t_x, t_P, cond_b, **kwargs)

    @torch.no_grad()
    def sample(self, *args, **kwargs):
        self._fused_cond_cache = ()
        try:
            return super().sample(*args, **kwargs)
        finally:
            self._fused_cond_cache = ()

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t_and_P_t: Tuple[torch.Tensor, torch.Tensor],
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t_and_P_t: A tuple of (x_t, P_t), where
             - x_t: The [N x C x ...] tensor of noisy inputs at time t.
             - P_t: The [N x N_views x C_pose x ...] tensor of noisy poses at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        x_t, P_t = x_t_and_P_t

        # Use the same timestep for x and P
        t_x, t_P = t, [t] * P_t.shape[1]
        t_prev_x, t_prev_P = t_prev, [t_prev] * P_t.shape[1]

        pred_v_x, pred_v_P = self._inference_model(model, x_t_and_P_t, (t_x, t_P), cond=cond, **kwargs)
        t_P = torch.tensor(t_P, dtype=torch.float32, device=P_t.device)[None, :, None, None, None, None]
        t_prev_P = torch.tensor(t_prev_P, dtype=torch.float32, device=P_t.device)[None, :, None, None, None, None]
        pred_x_0 = self._v_to_xstart(x_t=x_t, t=t_x, v=pred_v_x)
        pred_P_0 = self._v_to_xstart(x_t=P_t, t=t_P, v=pred_v_P)
        pred_x_prev = x_t - (t_x - t_prev_x) * pred_v_x
        pred_P_prev = P_t - (t_P - t_prev_P) * pred_v_P

        sample = (pred_x_prev, pred_P_prev)
        out = edict({"pred_x_t": pred_x_prev, "pred_x_0": pred_x_0, "pred_P_t": pred_P_prev, "pred_P_0": pred_P_0})
        return sample, out


class MultiPoseFlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, MultiPoseFlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance for multi-pose flow.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
            - 'pred_P_t': a list of prediction of P_t.
            - 'pred_P_0': a list of prediction of P_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, **kwargs)


class MultiPoseFlowEulerGuidanceIntervalSampler(MultiPoseGuidanceIntervalSamplerMixin, MultiPoseFlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval for multi-pose flow.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
            - 'pred_P_t': a list of prediction of P_t.
            - 'pred_P_0': a list of prediction of P_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)
