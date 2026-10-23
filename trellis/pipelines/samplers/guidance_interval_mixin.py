from typing import *


class GuidanceIntervalSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance with interval.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
        assert isinstance(t, float), "t must be a float"
        if cfg_interval[0] <= t <= cfg_interval[1]:
            pred = super()._inference_model(model, x_t, t, cond, **kwargs)
            neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
            return (1 + cfg_strength) * pred - cfg_strength * neg_pred
        else:
            return super()._inference_model(model, x_t, t, cond, **kwargs)


class MultiPoseGuidanceIntervalSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance with interval for multi-pose flow.
    """
    def _inference_model(self, model, x_t_and_P_t, t_x_and_t_P, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
        t_x, t_P = t_x_and_t_P
        assert isinstance(t_x, float), "t_x must be a float"
        assert all([t_P[i] == t_x for i in range(len(t_P))]), "t_P must be the same for all views"
        if cfg_interval[0] <= t_x <= cfg_interval[1]:
            x_pred, P_pred = super()._inference_model(model, x_t_and_P_t, t_x_and_t_P, cond, **kwargs)
            x_neg_pred, P_neg_pred = super()._inference_model(model, x_t_and_P_t, t_x_and_t_P, neg_cond, **kwargs)
            x_pred = (1 + cfg_strength) * x_pred - cfg_strength * x_neg_pred
            P_pred = (1 + cfg_strength) * P_pred - cfg_strength * P_neg_pred
            return x_pred, P_pred
        else:
            return super()._inference_model(model, x_t_and_P_t, t_x_and_t_P, cond, **kwargs)
