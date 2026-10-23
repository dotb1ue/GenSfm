"""Export mesh, Gaussian and camera artifacts for native evaluation."""
import json
import os

import torch

from . import postprocessing_utils
from .general_utils import json_safe


def save_outputs_for_instance(
    outputs,
    output_dir,
    index: int = 0,
    **kwargs,
):
    """
    Save a textured mesh, Gaussians, and cameras for one evaluated object.
    """
    # GLB files can be extracted from the outputs
    with torch.enable_grad():
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][index],
            outputs['mesh'][index],
            # Optional parameters
            simplify=0.95,          # Ratio of triangles to remove in the simplification process
            texture_size=1024,      # Size of the texture used for the GLB
            **kwargs,
        )
    os.makedirs(f'{output_dir}', exist_ok=True)
    glb_path = f'{output_dir}/mesh.glb'
    glb.export(glb_path)
    
    # Save Gaussians as PLY files
    gaussian_path = f'{output_dir}/gs.ply'
    outputs['gaussian'][index].save_ply(gaussian_path)
    
    pose_dict = outputs['pose'][index]

    # --- Save metadata to JSON ---
    output_data = {
        'glb_path': glb_path,
        'gaussian_path': gaussian_path,
        'pose': json_safe(pose_dict),
    }
    
    json_path = f'{output_dir}/metadata.json'
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=4)
