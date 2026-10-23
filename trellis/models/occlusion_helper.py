import torch
import torch.nn.functional as F


class OcclusionMaskPatcher:
    def __init__(self, input_resolution: int = 518, patch_size: int = 14, num_prefix_tokens: int = 5):
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.num_prefix_tokens = num_prefix_tokens

    def patchify_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Patchify an occlusion mask into per-patch non-occluded ratios.

        Accepts any number of leading dimensions, with spatial dims as the last two.
        Typical inputs: [B, H, W] (single view) or [B, V, H, W] (multi view).
        Output has the same leading dims, with spatial dims replaced by patch grid.
        """
        leading_shape = mask.shape[:-2]
        H, W = mask.shape[-2:]
        # Collapse leading dims, add a singleton channel for conv ops
        mask_flat = mask.reshape(-1, 1, H, W).float()  # [N, 1, H, W]
        if (H, W) != (self.input_resolution, self.input_resolution):
            mask_flat = F.interpolate(
                mask_flat,
                size=(self.input_resolution, self.input_resolution),
                mode='bilinear',
                antialias=True,
            )
        # Non-occluded ratio per patch
        patch_weights = 1 - F.avg_pool2d(
            mask_flat,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )  # [N, 1, h, w]
        h, w = patch_weights.shape[-2:]
        return patch_weights.reshape(*leading_shape, h, w)  # (..., h, w)

    def tokenize_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Tokenize an occlusion mask into per-token weights (prefix tokens + patch tokens).

        Accepts any number of leading dimensions with spatial dims as the last two.
        Output shape: (..., num_prefix_tokens + h*w).
        """
        patch_weights = self.patchify_mask(mask)  # (..., h, w)
        leading_shape = patch_weights.shape[:-2]
        patch_weights = patch_weights.reshape(*leading_shape, -1)  # (..., h*w)
        prefix_weights = torch.ones(
            (*leading_shape, self.num_prefix_tokens),
            device=mask.device,
            dtype=patch_weights.dtype,
        )
        return torch.cat([prefix_weights, patch_weights], dim=-1)  # (..., L)

