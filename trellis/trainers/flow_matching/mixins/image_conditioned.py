from typing import *
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from ....utils import dist_utils


class ImageConditionedMixin:
    """
    Mixin for image-conditioned models.
    
    Args:
        image_cond_model: The image conditioning model.
        image_cond_model_compile: If True, compile the DINOv2 model with torch.compile.
            If a dict, use it as kwargs for torch.compile(model, **kwargs).
    """
    def __init__(self,
        *args, 
        image_cond_model: str = 'dinov2_vitl14_reg',
        image_cond_model_compile: Union[bool, Dict[str, Any], None] = None, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.image_cond_model_name = image_cond_model
        self.image_cond_model_compile = image_cond_model_compile
        self.image_cond_model = None     # the model is init lazily
        self._image_norm_cache = {}

    def _get_normalize_stats(self, device: torch.device, dtype: torch.dtype):
        key = (str(device), dtype)
        if key not in self._image_norm_cache:
            mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)
            self._image_norm_cache[key] = (mean, std)
        return self._image_norm_cache[key]

    def _build_image_transform(self):
        """
        Build model-owned transform callable. Keep this here so model-specific
        transform logic can evolve without touching encode_image().
        """
        def transform(image: torch.Tensor) -> torch.Tensor:
            mean, std = self._get_normalize_stats(image.device, image.dtype)
            return (image - mean) / std
        return transform
        
    def _init_image_cond_model(self):
        """
        Initialize the image conditioning model.
        """
        with dist_utils.local_master_first():
            dinov2_model = torch.hub.load('facebookresearch/dinov2', self.image_cond_model_name, pretrained=True)
        dinov2_model.eval().cuda()
        if self.image_cond_model_compile:
            compile_kwargs = self.image_cond_model_compile if isinstance(self.image_cond_model_compile, dict) else {}
            dinov2_model = torch.compile(dinov2_model, **compile_kwargs)
        self.image_cond_model = {
            'model': dinov2_model,
            'transform': self._build_image_transform(),
        }
    
    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).cuda(non_blocking=True)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        if self.image_cond_model is None:
            self._init_image_cond_model()
        image = self.image_cond_model['transform'](image)
        features = self.image_cond_model['model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
        
    def prepare_cond(self, cond, kwargs):
        cond = self.encode_image(cond)
        kwargs['neg_cond'] = torch.zeros_like(cond)
        return cond, kwargs

    def get_training_cond(self, cond, **kwargs):
        cond, kwargs = self.prepare_cond(cond, kwargs)
        return super().get_training_cond(cond, **kwargs)

    def get_inference_cond(self, cond, **kwargs):
        cond, kwargs = self.prepare_cond(cond, kwargs)
        return super().get_inference_cond(cond, **kwargs)

    def vis_cond(self, cond, **kwargs):
        """
        Visualize the conditioning data.
        """
        return {'image': {'value': cond, 'type': 'image'}}


class MultiImageConditionedMixin(ImageConditionedMixin):
    """
    Mixin for multi-image-conditioned models.
    Handles multi-view [B, V, 3, H, W] inputs.
    """
    def encode_image(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        assert isinstance(image, torch.Tensor) and image.ndim == 5
        B, V = image.shape[:2]
        cond = super().encode_image(image.reshape(B * V, *image.shape[2:]))
        return cond.reshape(B, V, *cond.shape[1:])

    def vis_cond(self, cond, **kwargs):
        return {'image': {'value': cond, 'type': 'mv_image'}}


class VisualImageConditionedMixin(ImageConditionedMixin):
    """
    Mixin for visual image-conditioned models.

    Args:
        image_cond_model: The image conditioning model.
    """
    def prepare_cond(self, cond, kwargs):
        cond = {
            'visual_cond': cond,
            'dino_cond': self.encode_image(cond),
        }
        kwargs['neg_cond'] = {k: torch.zeros_like(v) for k, v in cond.items()}
        return cond, kwargs


class MultiVisualImageConditionedMixin(MultiImageConditionedMixin, VisualImageConditionedMixin):
    """
    Mixin for multi-view visual + DINOv2-conditioned models.
    Produces cond = {'visual_cond': (B,V,C,H,W), 'dino_cond': (B,V,N,D)}.
    """
    def vis_cond(self, cond, **kwargs):
        return {'image': {'value': cond, 'type': 'mv_image'}}
