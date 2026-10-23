import torch


class FlowMatchingCoreMixin:
    """
    Mixin providing core flow matching math shared by trainers and evaluators.

    Args:
        t_schedule (dict): Time schedule config with 'name' and 'args'.
        sigma_min (float): Minimum noise level.
    """

    def __init__(self, 
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
        super().__init__(*args, **kwargs)
        self.t_schedule = t_schedule
        self.sigma_min = sigma_min

    def diffuse(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            t: The [N] (or [N x V] for multi-pose) tensor of diffusion steps [0-1].
            noise: The [N x C x ...] tensor of noise.

        Returns:
            x_t, the noisy version of x_0 under timestep t.
        """
        assert noise.shape == x_0.shape, "noise must have same shape as x_0"
        t = t.view(*t.shape, *[1 for _ in range(x_0.dim() - t.dim())])
        x_t = (1 - t) * x_0 + (self.sigma_min + (1 - self.sigma_min) * t) * noise
        return x_t

    def reverse_diffuse(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Get original image from noisy version under timestep t.
        """
        assert noise.shape == x_t.shape, "noise must have same shape as x_t"
        t = t.view(*t.shape, *[1 for _ in range(x_t.dim() - t.dim())])
        x_0 = (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * noise) / (1 - t)
        return x_0

    def get_v(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the velocity of the diffusion process at time t.
        """
        return (1 - self.sigma_min) * noise - x_0

    def sample_t(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """
        Sample timesteps.

        Args:
            batch_size: Number of timesteps to sample.
            device: Device for the output tensor. If None, uses CPU.
        """
        if self.t_schedule['name'] == 'uniform':
            t = torch.rand(batch_size, device=device, dtype=torch.float32)
        elif self.t_schedule['name'] == 'logitNormal':
            mean = self.t_schedule['args']['mean']
            std = self.t_schedule['args']['std']
            t = torch.randn(batch_size, device=device, dtype=torch.float32)
            t = torch.sigmoid(t * std + mean)
        elif self.t_schedule['name'] == 'fixed':
            t = torch.full((batch_size,), self.t_schedule['args']['t'], device=device, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown t_schedule: {self.t_schedule['name']}")
        return t

    def get_training_cond(self, cond, **kwargs):
        """
        Get the conditioning data for training.
        """
        return cond

    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        """
        return {'cond': cond, **kwargs}
