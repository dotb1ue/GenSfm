import torch
from ....utils.general_utils import dicts_foreach
from ....pipelines import samplers


class ClassifierFreeGuidanceMixin:
    def __init__(self, *args, p_uncond: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_uncond = p_uncond

    def get_training_cond(self, cond, neg_cond=None, **kwargs):
        """
        Get the conditioning data.
        """
        assert neg_cond is not None, "neg_cond must be provided for classifier-free guidance" 

        if self.p_uncond > 0:
            # randomly drop the class label
            def get_batch_size(cond):
                if isinstance(cond, torch.Tensor):
                    return cond.shape[0]
                elif isinstance(cond, list):
                    return len(cond)
                else:
                    raise ValueError(f"Unsupported type of cond: {type(cond)}")
                
            ref_cond = cond if not isinstance(cond, dict) else cond[list(cond.keys())[0]]
            B = get_batch_size(ref_cond)
            
            def select(cond, neg_cond, mask):
                if isinstance(cond, torch.Tensor):
                    if isinstance(mask, torch.Tensor):
                        mask_t = mask
                    else:
                        mask_t = torch.as_tensor(mask, device=cond.device)
                    mask_t = mask_t.reshape(-1, *[1] * (cond.ndim - 1))
                    return torch.where(mask_t, neg_cond, cond)
                elif isinstance(cond, list):
                    return [nc if m else c for c, nc, m in zip(cond, neg_cond, mask)]
                else:
                    raise ValueError(f"Unsupported type of cond: {type(cond)}")

            if isinstance(ref_cond, torch.Tensor):
                mask = torch.rand(B, device=ref_cond.device) < self.p_uncond
            else:
                mask = (torch.rand(B) < self.p_uncond).tolist()
            if not isinstance(cond, dict):
                cond = select(cond, neg_cond, mask)
            else:
                cond = dicts_foreach([cond, neg_cond], lambda x: select(x[0], x[1], mask))
    
        return cond

    def get_inference_cond(self, cond, neg_cond=None, **kwargs):
        """
        Get the conditioning data for inference.
        """
        assert neg_cond is not None, "neg_cond must be provided for classifier-free guidance"
        return {'cond': cond, 'neg_cond': neg_cond, **kwargs}
    
    def get_sampler(self, **kwargs) -> samplers.FlowEulerCfgSampler:
        """
        Get the sampler for the diffusion process.
        """
        return samplers.FlowEulerCfgSampler(self.sigma_min)
