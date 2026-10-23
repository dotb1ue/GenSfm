import torch
import torch.nn.functional as F
import math

from ..base import Evaluator
from ...utils.pose_utils import *
from ...utils.voxel_utils import dilate_binary_volume
from ...datasets.sparse_uv_structure import dilate_sparse_grid


class SparseUVStructureVaeEvaluator(Evaluator):
    """
    Evaluator for Sparse UV Structure VAE.
    
    Args:
        models (dict[str, nn.Module]): Models to evaluate.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        load_step (int): Step to load.
        pretrained_ckpt (dict): Pretrained checkpoint.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        fp16_mode (str): FP16 mode.
        i_sample (int): Sample interval.
        input_ssuv (bool): Whether to use the sparse UV structure (partial voxels) as input. Default is True.
        input_uv_volume (bool): Whether to use the uv volume as input. Default is True.
        input_depth (bool): Whether to use the depth volume as input. Default is False.
        predict_ssuv (bool): Whether to predict the sparse UV structure. Default is True.
        evaluate_pose (bool): Whether to evaluate the pose related metrics. Default is True.
        use_ss_as_weights (bool): Whether to use the sparse 3D structure as weights in pose evaluation. Default is False.
        dilation_sigma (float): Standard deviation of Gaussian kernel for post-processing dilation. Default is 0.
        dilation_min_weight (float): Minimum weight to include in the dilated grid. Default is 0.05.
        evaluate_suv_latent (bool): Whether to evaluate the stored suv latent instead of the output of encoder. Default is False.
    """
    
    def __init__(
        self,
        *args,
        input_ssuv=True,
        input_uv_volume=True,
        input_depth=False,
        predict_ssuv=True,
        evaluate_pose=True,
        use_ss_as_weights=False,
        dilation_sigma=0.0,
        dilation_min_weight=0.05,
        evaluate_suv_latent=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.input_ssuv = input_ssuv
        self.input_uv_volume = input_uv_volume
        self.input_depth = input_depth
        self.predict_ssuv = predict_ssuv
        self.evaluate_pose = evaluate_pose
        self.use_ss_as_weights = use_ss_as_weights
        self.dilation_sigma = dilation_sigma
        self.dilation_min_weight = dilation_min_weight
        self.evaluate_suv_latent = evaluate_suv_latent
        self._xyz_voxels = None

    def evaluation_metrics(self, indices, **data):
        """
        Compute evaluation metrics for sparse UV structure VAE.
        
        Args: 
            indices: [N] tensor of indices for this batch.
            **data may contain the following keys:
            - 'ss': The [N x 1 x H x W x D] tensor of binary sparse structure.
            - 'ssuv': The [N x 1 x H x W x D] tensor of binary sparse structure with UV.
            - 'uv_volume': The [N x 2 x H x W x D] tensor of UV coordinates.
            - 'depth_volume': The [N x 1 x H x W x D] tensor of depth volume.
            - 'cond': The [N x 3 x ImageHeight x ImageWidth] tensor of image condition.
            - 'intrinsics': The [N x 3 x 3] tensor of intrinsics.
            - 'extrinsics': The [N x 4 x 4] tensor of extrinsics.
                
        Returns:
            Tuple of (metrics_dict, samples_dict):
                - metrics_dict: Dictionary of scalar metrics per sample
                - samples_dict: Dictionary of samples for visualization
        """
        # Prepare input volume and get predictions
        preds = self._get_predictions(**data)
        
        # Compute metrics for each component
        metrics = {}
        samples = {}
        
        # SSUV metrics
        if self.predict_ssuv:
            ssuv_metrics, ssuv_samples = self._evaluate_occupancy('ssuv', data['ssuv'], preds['ssuv_logits'], preds['ssuv_recon'])
            metrics.update(ssuv_metrics)
            samples.update(ssuv_samples)
        
        # UV volume metrics
        uv_metrics, uv_samples = self._evaluate_uv_volume(**data, **preds)
        metrics.update(uv_metrics)
        samples.update(uv_samples)

        # Pose related metrics
        if self.evaluate_pose:
            pose_metrics, pose_samples = self._evaluate_poses(**data, **preds)
            metrics.update(pose_metrics)
            samples.update(pose_samples)
        
        return metrics, samples

    def _get_predictions(self, **data):
        """Get model predictions for all components."""
        if self.evaluate_suv_latent:
            if 'P_0' in data:
                assert data['P_0'].shape[1] == 1, "P_0 should be a single-view UV latent"
                z = data['P_0'].squeeze(1)
            else:
                z = data['x_0'][:, -self.models['decoder'].latent_channels:]
        else:
            inputs = []
            if self.input_ssuv:
                inputs.append(data['ssuv'].float())
            if self.input_uv_volume:
                inputs.append(data['uv_volume'])
            if self.input_depth:
                inputs.append(data['depth_volume'])
            input_volume = torch.cat(inputs, dim=1)
            z = self.models['encoder'](input_volume, sample_posterior=False)
        logits = self.models['decoder'](z)

        ssuv_logits, ssuv_recon = None, None
        if self.predict_ssuv:
            ssuv_logits, logits = logits[:, :1, :, :, :], logits[:, 1:, :, :, :]
            ssuv_recon = (ssuv_logits > 0).int()
        uv_volume_pred = torch.sigmoid(logits[:, :2, :, :, :])
        
        return {
            'ssuv_logits': ssuv_logits,
            'ssuv_recon': ssuv_recon,
            'uv_volume_pred': uv_volume_pred,
        }

    def _evaluate_occupancy(self, prefix, occupancy, occupancy_logits, occupancy_recon):
        """Evaluate the occupancy of sparse structure metrics."""
        metrics = {}
        samples = {}
        
        # BCE loss for occupancy
        bce_loss = F.binary_cross_entropy_with_logits(occupancy_logits, occupancy.float(), reduction='none')
        bce_loss = bce_loss.mean(dim=[2, 3, 4])  # Average over spatial dimensions
        metrics[f'{prefix}_bce_loss'] = bce_loss.squeeze(1)  # [batch_size]
        
        # Binary accuracy and IoU (Intersection over Union)
        accuracy = (occupancy_recon == occupancy).float().mean(dim=[2, 3, 4])
        metrics[f'{prefix}_accuracy'] = accuracy.squeeze(1)  # [batch_size]
        intersection = (occupancy_recon * occupancy).sum(dim=[2, 3, 4])
        union = (occupancy_recon + occupancy).clamp(0, 1).sum(dim=[2, 3, 4])
        iou = (intersection + 1e-8) / (union + 1e-8)  # Add small epsilon to avoid division by zero
        metrics[f'{prefix}_iou'] = iou.squeeze(1)  # [batch_size]
        
        # Ratio of activated occupancy voxels (occupancy_recon = True)
        activated_count = occupancy_recon.sum(dim=[2, 3, 4])  # [N x 1]
        total_voxels = occupancy_recon.numel() // occupancy_recon.shape[0]  # Total voxels per sample
        activate_ratio = activated_count.float() / total_voxels
        metrics[f'{prefix}_activate_ratio'] = activate_ratio.squeeze(1)  # [batch_size]
        
        # Samples for visualization
        samples[f'{prefix}_gt'] = {'value': occupancy, 'type': 'sample'}
        samples[f'{prefix}_recon'] = {'value': occupancy_recon, 'type': 'sample'}
        samples[f'{prefix}_diff'] = {'value': (occupancy_recon - occupancy).abs(), 'type': 'sample'}
        
        return metrics, samples

    def _evaluate_uv_volume(self, uv_volume, ssuv, uv_volume_pred, **kwargs):
        """Evaluate UV volume metrics."""
        metrics = {}
        samples = {}
        
        ssuv_sum = ssuv.sum(dim=[1, 2, 3, 4])  # [N]
        
        # L1 and MSE loss for UV volume (weighted by ground truth uv weight)
        uv_l1_loss = F.l1_loss(uv_volume_pred, uv_volume, reduction='none')
        uv_mse_loss = F.mse_loss(uv_volume_pred, uv_volume, reduction='none')
        metrics['uv_l1_loss'] = (uv_l1_loss * ssuv).sum(dim=[1, 2, 3, 4]) / ssuv_sum
        metrics['uv_mse_loss'] = (uv_mse_loss * ssuv).sum(dim=[1, 2, 3, 4]) / ssuv_sum

        # UV distance (weighted by ground truth uv weight)
        uv_distance = torch.norm(uv_volume - uv_volume_pred, dim=1, keepdim=True)
        metrics['uv_distance'] = (uv_distance * ssuv).sum(dim=[1, 2, 3, 4]) / ssuv_sum
        
        # Samples for visualization
        samples['uv_volume_gt'] = {'value': {'uv_volume': uv_volume, 'ssuv': ssuv}, 'type': 'sample'}
        samples['uv_volume_pred'] = {'value': {'uv_volume': uv_volume_pred, 'ssuv': ssuv}, 'type': 'sample'}
        
        # UV volume difference visualization
        uv_volume_diff = (uv_volume - uv_volume_pred).norm(dim=1, keepdim=True)
        uv_volume_diff = uv_volume_diff / uv_volume_diff.max().clamp_min_(1e-2)
        samples['uv_volume_diff'] = {'value': {'uv_volume': uv_volume_diff, 'ssuv': ssuv}, 'type': 'sample'}
        
        return metrics, samples 

    def _evaluate_poses(self, uv_volume, ssuv, uv_volume_pred, ssuv_recon, intrinsics, extrinsics, **kwargs):
        """Evaluate transforms metrics."""
        metrics = {}
        samples = {}
        
        # Get the center xyz of the voxels
        if self._xyz_voxels is None:
            x = torch.arange(uv_volume_pred.shape[-1], dtype=torch.float32, device=uv_volume_pred.device)
            y = torch.arange(uv_volume_pred.shape[-2], dtype=torch.float32, device=uv_volume_pred.device)
            z = torch.arange(uv_volume_pred.shape[-3], dtype=torch.float32, device=uv_volume_pred.device)
            x = (x + 0.5) / uv_volume_pred.shape[-1] - 0.5
            y = (y + 0.5) / uv_volume_pred.shape[-2] - 0.5
            z = (z + 0.5) / uv_volume_pred.shape[-3] - 0.5
            xyz_voxels = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=0)  # [3 x 64 x 64 x 64]
            self._xyz_voxels = xyz_voxels.view(1, 3, -1)  # [1 x 3 x 64*64*64]

        # Get the uv and weights of the voxels
        uv_voxels = uv_volume_pred.flatten(start_dim=2)  # [N x 2 x 64*64*64]
        uv_voxels_gt = uv_volume.flatten(start_dim=2)  # [N x 2 x 64*64*64]
        if self.use_ss_as_weights:
            voxel_weights = kwargs['ss'].flatten(start_dim=1)  # [N x 64*64*64]
            voxel_weights_gt = kwargs['ss'].flatten(start_dim=1)  # [N x 64*64*64]
        else:
            voxel_weights = (ssuv_recon if self.predict_ssuv else ssuv).flatten(start_dim=1)  # [N x 64*64*64]
            voxel_weights_gt = ssuv.flatten(start_dim=1)  # [N x 64*64*64]
        
        # Compute the transforms
        # transforms_pred, A_pred = dlt_calibration(self._xyz_voxels, uv_voxels, weights=voxel_weights)
        # transforms_best, A_best = dlt_calibration(self._xyz_voxels, uv_voxels_gt, weights=voxel_weights_gt)

        extrinisc_best = []
        intrinsic_best = []
        reproj_best = []
        extrinisc_pred = []
        intrinsic_pred = []
        reproj_pred = []
        for i in range(len(uv_voxels)):
            valid_voxel = (voxel_weights[i] > 0)
            valid_gt = (voxel_weights_gt[i] > 0)

            if self.dilation_sigma > 0:
                valid_voxel = dilate_binary_volume(valid_voxel.view(64,64,64), sigma=self.dilation_sigma, min_weight=self.dilation_min_weight).flatten()
                valid_gt = dilate_binary_volume(valid_gt.view(64,64,64), sigma=self.dilation_sigma, min_weight=self.dilation_min_weight).flatten()

            pred_xyzs = self._xyz_voxels[:, :, valid_voxel].permute(0,2,1).contiguous()
            pred_uvs = uv_voxels[i:i+1, :, valid_voxel].permute(0,2,1).contiguous()
            gt_xyzs = self._xyz_voxels[:, :, valid_gt].permute(0,2,1).contiguous()
            gt_uvs = uv_voxels_gt[i:i+1, :, valid_gt].permute(0,2,1).contiguous()
            
            K, RT, RE = calibrate_camera_dlt(gt_xyzs, gt_uvs)
            intrinsic_best.append(K)
            extrinisc_best.append(RT)
            reproj_best.append(RE)

            K, RT, RE = calibrate_camera_dlt(pred_xyzs, pred_uvs)
            intrinsic_pred.append(K)
            extrinisc_pred.append(RT)
            reproj_pred.append(RE)

        extrinisc_best = torch.concat(extrinisc_best, dim=0)
        intrinsic_best = torch.concat(intrinsic_best, dim=0)
        extrinisc_pred = torch.concat(extrinisc_pred, dim=0)
        intrinsic_pred = torch.concat(intrinsic_pred, dim=0)
        reproj_pred = torch.stack(reproj_pred)
        reproj_best = torch.stack(reproj_best)

        for prefix, ext, K, reproj in [('',        extrinisc_pred, intrinsic_pred, reproj_pred),
                                       ('oracle_', extrinisc_best, intrinsic_best, reproj_best)]:
            for k, v in compute_pose_diff_metrics(ext, K, extrinsics, intrinsics).items():
                metrics[f'{prefix}{k}'] = v
            metrics[f'{prefix}reproj'] = reproj.reshape(-1).clamp_max(1.0)

        return metrics, samples