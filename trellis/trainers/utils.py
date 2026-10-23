import math
import torch
import torch.nn as nn
from dataclasses import dataclass

# FP16 utils
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


@dataclass
class _FusedCopyCtx:
    enabled: bool
    master_views: list | None = None
    grad_views: list | None = None
    flat_grad: torch.Tensor | list | None = None
    unpack_graph: torch.cuda.CUDAGraph | None = None
    pack_graph: torch.cuda.CUDAGraph | None = None
    pack_grad_ptrs: tuple | None = None
    zero_graph: torch.cuda.CUDAGraph | None = None
    zero_grad_ptrs: tuple | None = None
    failed: bool = False
    warned: bool = False
    group_indices: list | None = None


def init_fused_master_copy_context(model_params, master_params, group_indices=None, enabled=True):
    """
    Prepare reusable views/graphs for model<->master copy in inflat_all mode.
    """
    active = (
        enabled
        and torch.cuda.is_available()
        and master_params[0].is_cuda
        and hasattr(torch.cuda, "CUDAGraph")
    )
    ctx = _FusedCopyCtx(enabled=bool(active), group_indices=group_indices)
    if active:
        if group_indices is not None:
            # Grouped mode: build views from multiple flat tensors, interleaved to model_params order
            ctx.master_views = [None] * len(model_params)
            ctx.grad_views = [None] * len(model_params)
            ctx.flat_grad = []  # list of flat grad tensors, one per group
            for g, indices in enumerate(group_indices):
                templates = [model_params[i] for i in indices]
                m_views = _unflatten_dense_tensors(master_params[g].detach(), templates)
                flat_g = torch.empty_like(master_params[g].detach())
                g_views = _unflatten_dense_tensors(flat_g, templates)
                ctx.flat_grad.append(flat_g)
                for j, idx in enumerate(indices):
                    ctx.master_views[idx] = m_views[j]
                    ctx.grad_views[idx] = g_views[j]
        else:
            # Single flat tensor mode (original path)
            flat_master = master_params[0].detach()
            ctx.master_views = _unflatten_dense_tensors(flat_master, model_params)
            ctx.flat_grad = torch.empty_like(flat_master)
            ctx.grad_views = _unflatten_dense_tensors(ctx.flat_grad, model_params)
    return ctx


def _disable_fused_ctx(ctx, message):
    ctx.failed = True
    if not ctx.warned:
        print(f"\n\033[93mWarning: disabling fused master copy path: {message}\033[0m")
        ctx.warned = True


def _capture_unpack_graph(ctx, model_params):
    assert ctx.master_views is not None
    # Warm up before graph capture.
    for _ in range(2):
        for param, master_view in zip(model_params, ctx.master_views):
            param.detach().copy_(master_view)
    torch.cuda.synchronize()
    ctx.unpack_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(ctx.unpack_graph):
        for param, master_view in zip(model_params, ctx.master_views):
            param.detach().copy_(master_view)


def _capture_pack_graph(ctx, model_params):
    assert ctx.grad_views is not None
    grad_ptrs = tuple(
        param.grad.data_ptr() if param.grad is not None else 0 for param in model_params
    )
    if any(ptr == 0 for ptr in grad_ptrs):
        return False
    # Warm up before graph capture.
    for _ in range(2):
        for grad_view, param in zip(ctx.grad_views, model_params):
            grad_view.copy_(param.grad.detach())
    torch.cuda.synchronize()
    ctx.pack_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(ctx.pack_graph):
        for grad_view, param in zip(ctx.grad_views, model_params):
            grad_view.copy_(param.grad.detach())
    ctx.pack_grad_ptrs = grad_ptrs
    return True


def _prepare_grad_for_zero(grad: torch.Tensor):
    if grad.grad_fn is not None:
        grad.detach_()
    else:
        grad.requires_grad_(False)


def _capture_zero_graph(ctx, model_params):
    grads = [param.grad for param in model_params if param.grad is not None]
    if len(grads) == 0:
        return False
    for grad in grads:
        _prepare_grad_for_zero(grad)
    grad_ptrs = tuple(grad.data_ptr() for grad in grads)
    # Warm up before graph capture.
    for _ in range(2):
        for grad in grads:
            grad.zero_()
    torch.cuda.synchronize()
    ctx.zero_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(ctx.zero_graph):
        for grad in grads:
            grad.zero_()
    ctx.zero_grad_ptrs = grad_ptrs
    return True

def make_master_params(model_params):
    """
    Copy model parameters into a inflated tensor of full-precision parameters.
    """
    master_params = _flatten_dense_tensors(
        [param.detach().float() for param in model_params]
    )
    master_params = nn.Parameter(master_params)
    master_params.requires_grad = True
    return [master_params]


def classify_params_by_decay_group(models, model_params, unwrap_fn):
    """
    Classify model_params into (decay, bias, layernorm) groups.
    LayerNorm params (including LN bias) follow weight_decay_layer_norm.
    Non-LN biases follow weight_decay_bias. All others follow weight_decay.

    Returns:
        group_indices: list of 3 lists of ints — indices into model_params
            [decay_indices, bias_indices, layernorm_indices]
    """
    bias_ids, layer_norm_ids = set(), set()
    for model in models.values():
        for module in unwrap_fn(model).modules():
            if isinstance(module, nn.LayerNorm):
                for p in module.parameters():
                    if p.requires_grad:
                        layer_norm_ids.add(id(p))
            elif hasattr(module, 'bias') and module.bias is not None:
                if module.bias.requires_grad:
                    bias_ids.add(id(module.bias))

    decay_indices, bias_indices, ln_indices = [], [], []
    for i, p in enumerate(model_params):
        pid = id(p)
        if pid in layer_norm_ids:
            ln_indices.append(i)
        elif pid in bias_ids:
            bias_indices.append(i)
        else:
            decay_indices.append(i)

    # Assert strict partition
    total = len(decay_indices) + len(bias_indices) + len(ln_indices)
    assert total == len(model_params), \
        f"Partition error: {total} classified but {len(model_params)} params exist"

    return [decay_indices, bias_indices, ln_indices]


def make_grouped_master_params(model_params, group_indices):
    """
    Create one flat FP32 nn.Parameter per non-empty group.

    Args:
        model_params: list of model params in original order
        group_indices: list of lists of ints (indices into model_params)
    Returns:
        master_params: list of nn.Parameter (one per non-empty group)
        effective_group_indices: list of lists of ints (no empty groups)
    """
    master_params = []
    effective_group_indices = []
    for indices in group_indices:
        if not indices:
            continue
        flat = _flatten_dense_tensors(
            [model_params[i].detach().float() for i in indices]
        )
        param = nn.Parameter(flat)
        param.requires_grad = True
        master_params.append(param)
        effective_group_indices.append(indices)
    return master_params, effective_group_indices


def unflatten_master_params(model_params, master_params, group_indices=None):
    """
    Unflatten the master parameters to look like model_params.
    """
    if group_indices is not None:
        result = [None] * len(model_params)
        for g, indices in enumerate(group_indices):
            templates = [model_params[i] for i in indices]
            views = _unflatten_dense_tensors(master_params[g].detach(), templates)
            for j, idx in enumerate(indices):
                result[idx] = views[j]
        return result
    return _unflatten_dense_tensors(master_params[0].detach(), model_params)


def model_params_to_master_params(model_params, master_params, group_indices=None):
    """
    Copy the model parameter data into the master parameters.
    """
    if group_indices is not None:
        for g, indices in enumerate(group_indices):
            master_params[g].detach().copy_(
                _flatten_dense_tensors([model_params[i].detach().float() for i in indices])
            )
        return
    master_params[0].detach().copy_(
        _flatten_dense_tensors([param.detach().float() for param in model_params])
    )


def master_params_to_model_params(model_params, master_params, group_indices=None, fused_copy_ctx=None):
    """
    Copy the master parameter data back into the model parameters.
    """
    ctx = fused_copy_ctx
    if ctx is not None and ctx.enabled and not ctx.failed:
        try:
            if ctx.unpack_graph is None:
                _capture_unpack_graph(ctx, model_params)
            ctx.unpack_graph.replay()
            return
        except Exception as err:
            _disable_fused_ctx(ctx, str(err))
    if group_indices is not None:
        for g, indices in enumerate(group_indices):
            templates = [model_params[i] for i in indices]
            views = _unflatten_dense_tensors(master_params[g].detach(), templates)
            for j, idx in enumerate(indices):
                model_params[idx].detach().copy_(views[j])
        return
    for param, master_param in zip(
        model_params, _unflatten_dense_tensors(master_params[0].detach(), model_params)
    ):
        param.detach().copy_(master_param)


def model_grads_to_master_grads(model_params, master_params, group_indices=None, fused_copy_ctx=None):
    """
    Copy the gradients from the model parameters into the master parameters.
    """
    ctx = fused_copy_ctx
    if ctx is not None and ctx.enabled and not ctx.failed:
        try:
            if ctx.flat_grad is None:
                raise RuntimeError("fused grad buffer is not initialized")
            if ctx.group_indices is not None:
                # Grouped mode: bind each master_params[g].grad to its flat_grad buffer
                for g in range(len(ctx.group_indices)):
                    fg = ctx.flat_grad[g]
                    if master_params[g].grad is None or master_params[g].grad.data_ptr() != fg.data_ptr():
                        master_params[g].grad = fg
            else:
                if (
                    master_params[0].grad is None
                    or master_params[0].grad.data_ptr() != ctx.flat_grad.data_ptr()
                ):
                    master_params[0].grad = ctx.flat_grad
            grad_ptrs = tuple(
                param.grad.data_ptr() if param.grad is not None else 0
                for param in model_params
            )
            if any(ptr == 0 for ptr in grad_ptrs):
                raise RuntimeError("found None gradient in fused pack path")
            if ctx.pack_graph is None or ctx.pack_grad_ptrs != grad_ptrs:
                ok = _capture_pack_graph(ctx, model_params)
                if not ok:
                    raise RuntimeError(
                        "unable to capture pack graph due to None gradients"
                    )
            ctx.pack_graph.replay()
            return
        except Exception as err:
            _disable_fused_ctx(ctx, str(err))

    if group_indices is not None:
        for g, indices in enumerate(group_indices):
            master_params[g].grad = _flatten_dense_tensors(
                [model_params[i].grad.data.detach().float() for i in indices]
            )
    else:
        master_params[0].grad = _flatten_dense_tensors(
            [param.grad.data.detach().float() for param in model_params]
        )
    

def zero_grad(model_params, fused_copy_ctx=None):
    ctx = fused_copy_ctx
    if ctx is not None and ctx.enabled and not ctx.failed:
        try:
            grads = [param.grad for param in model_params if param.grad is not None]
            if len(grads) == 0:
                return
            for grad in grads:
                _prepare_grad_for_zero(grad)
            grad_ptrs = tuple(grad.data_ptr() for grad in grads)
            if ctx.zero_graph is None or ctx.zero_grad_ptrs != grad_ptrs:
                ok = _capture_zero_graph(ctx, model_params)
                if not ok:
                    return
            ctx.zero_graph.replay()
            return
        except Exception as err:
            _disable_fused_ctx(ctx, str(err))

    for param in model_params:
        if param.grad is not None:
            if param.grad.grad_fn is not None:
                param.grad.detach_()
            else:
                param.grad.requires_grad_(False)
            param.grad.zero_()


# Backward compat helpers for grouped master params

def remap_optimizer_state_single_to_grouped(opt_state_dict, param_numels, group_indices, group_weight_decays):
    """
    Remap optimizer state from a 1-param-group checkpoint to a multi-param-group format.
    Generic: splits any tensor state matching [total_numel], copies scalars as-is.
    """
    if 0 not in opt_state_dict['state']:
        # No optimizer state yet (checkpoint saved before first step) — just rebuild param_groups
        new_param_groups = []
        old_group = opt_state_dict['param_groups'][0]
        for g, wd in enumerate(group_weight_decays):
            pg = {k: v for k, v in old_group.items() if k != 'params'}
            pg['params'] = [g]
            pg['weight_decay'] = wd
            new_param_groups.append(pg)
        return {'state': {}, 'param_groups': new_param_groups}

    old_state = opt_state_dict['state'][0]
    old_group = opt_state_dict['param_groups'][0]
    total_numel = sum(param_numels)

    # Precompute cumulative offsets for each param in original model_params order
    cum_offsets = [0]
    for n in param_numels:
        cum_offsets.append(cum_offsets[-1] + n)

    new_state = {}
    for g, indices in enumerate(group_indices):
        group_state = {}
        for key, val in old_state.items():
            if isinstance(val, torch.Tensor) and val.shape == (total_numel,):
                # Split flat tensor: gather slices for this group's params
                slices = [val[cum_offsets[i]:cum_offsets[i] + param_numels[i]] for i in indices]
                group_state[key] = torch.cat(slices)
            else:
                # Scalar state (e.g. step): copy as-is
                group_state[key] = val
        new_state[g] = group_state

    # Rebuild param_groups: copy hyperparams from old group, override weight_decay per group
    new_param_groups = []
    for g, wd in enumerate(group_weight_decays):
        pg = {k: v for k, v in old_group.items() if k != 'params'}
        pg['params'] = [g]
        pg['weight_decay'] = wd
        new_param_groups.append(pg)

    return {'state': new_state, 'param_groups': new_param_groups}


def remap_lr_scheduler_state_single_to_grouped(scheduler_state_dict, num_groups):
    """
    Expand LR scheduler state from 1-group to N-group format.
    Replicates base_lrs and _last_lr entries to match num_groups.
    """
    state = dict(scheduler_state_dict)
    if 'base_lrs' in state and len(state['base_lrs']) == 1:
        state['base_lrs'] = state['base_lrs'] * num_groups
    if '_last_lr' in state and len(state['_last_lr']) == 1:
        state['_last_lr'] = state['_last_lr'] * num_groups
    return state


def handle_partial_mismatch(ckpt_tensor, model_tensor):
    """
    Handle partial mismatch by keeping the common part from checkpoint and 
    the excess part from the current model state dict.
    
    Args:
        ckpt_tensor: tensor from checkpoint
        model_tensor: tensor from current model state dict
        
    Returns:
        tensor: combined tensor with common part from checkpoint and excess part from model
    """
    if ckpt_tensor.ndim != model_tensor.ndim:
        raise ValueError(f'Tensor dimensions must match for partial loading. Got {ckpt_tensor.ndim} vs {model_tensor.ndim}')
    
    # Create a copy of the model tensor
    result = model_tensor.clone()
    
    # Create slicing indices for the common part
    slices = []
    for i in range(ckpt_tensor.ndim):
        if ckpt_tensor.shape[i] <= model_tensor.shape[i]:
            slices.append(slice(0, ckpt_tensor.shape[i]))
        else:
            raise ValueError(f'Checkpoint tensor dimension {i} ({ckpt_tensor.shape[i]}) is larger than model tensor dimension {i} ({model_tensor.shape[i]})')
    
    # Load the common part from checkpoint
    result[tuple(slices)] = ckpt_tensor
    
    return result


def handle_zeropad_mismatch(ckpt_tensor, model_tensor):
    """
    Handle zeropad mismatch by keeping the common part from checkpoint and 
    filling the excess part with zeros.
    
    Args:
        ckpt_tensor: tensor from checkpoint
        model_tensor: tensor from current model state dict
        
    Returns:
        tensor: combined tensor with common part from checkpoint and excess part filled with zeros
    """
    if ckpt_tensor.ndim != model_tensor.ndim:
        raise ValueError(f'Tensor dimensions must match for zeropad loading. Got {ckpt_tensor.ndim} vs {model_tensor.ndim}')
    
    # Create a zero tensor with the model's shape
    result = torch.zeros_like(model_tensor)
    
    # Create slicing indices for the common part
    slices = []
    for i in range(ckpt_tensor.ndim):
        if ckpt_tensor.shape[i] <= model_tensor.shape[i]:
            slices.append(slice(0, ckpt_tensor.shape[i]))
        else:
            raise ValueError(f'Checkpoint tensor dimension {i} ({ckpt_tensor.shape[i]}) is larger than model tensor dimension {i} ({model_tensor.shape[i]})')
    
    # Load the common part from checkpoint
    result[tuple(slices)] = ckpt_tensor
    
    return result 
            

# LR Schedulers
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

class LinearWarmupLRScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(LinearWarmupLRScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)
        
    def lr_lambda(self, current_step):
        if current_step < self.warmup_steps:
            return float(current_step + 1) / self.warmup_steps
        return 1.0


class CosineWarmupScheduler(LRScheduler):
    """
    Cosine warmup scheduler with cosine annealing.
    
    Args:
        optimizer: The optimizer to schedule
        warmup_steps: Number of warmup steps
        max_steps: Total number of training steps
        initial_lr: Initial learning rate for warmup (default: 1e-10)
        end_lr: End learning rate for cosine annealing (default: 0)
        last_epoch: Last epoch (default: -1)
    """
    def __init__(self, optimizer, warmup_steps, max_steps, initial_lr=1e-10, end_lr=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self._step_count <= self.warmup_steps:
            # Linear warmup
            return [
                self.initial_lr + (base_lr - self.initial_lr) * self._step_count / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            cos_iter = self._step_count - self.warmup_steps
            cos_max_iter = self.max_steps - self.warmup_steps
            cos_theta = cos_iter / cos_max_iter * math.pi
            cos_lr = [
                self.end_lr + (base_lr - self.end_lr) * (1 + math.cos(cos_theta)) / 2
                for base_lr in self.base_lrs
            ]
            return cos_lr        
