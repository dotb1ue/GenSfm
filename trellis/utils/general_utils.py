import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import contextlib


# Dictionary utils
def _dict_merge(dicta, dictb, prefix=''):
    """
    Merge two dictionaries.
    """
    assert isinstance(dicta, dict), 'input must be a dictionary'
    assert isinstance(dictb, dict), 'input must be a dictionary'
    dict_ = {}
    all_keys = set(dicta.keys()).union(set(dictb.keys()))
    for key in all_keys:
        if key in dicta.keys() and key in dictb.keys():
            if isinstance(dicta[key], dict) and isinstance(dictb[key], dict):
                dict_[key] = _dict_merge(dicta[key], dictb[key], prefix=f'{prefix}.{key}')
            else:
                raise ValueError(f'Duplicate key {prefix}.{key} found in both dictionaries. Types: {type(dicta[key])}, {type(dictb[key])}')
        elif key in dicta.keys():
            dict_[key] = dicta[key]
        else:
            dict_[key] = dictb[key]
    return dict_


def dict_merge(dicta, dictb):
    """
    Merge two dictionaries.
    """
    return _dict_merge(dicta, dictb, prefix='')


def dict_foreach(dic, func, special_func={}):
    """
    Recursively apply a function to all non-dictionary leaf values in a dictionary.
    """
    assert isinstance(dic, dict), 'input must be a dictionary'
    for key in dic.keys():
        if isinstance(dic[key], dict):
            dic[key] = dict_foreach(dic[key], func, special_func)
        else:
            if key in special_func.keys():
                dic[key] = special_func[key](dic[key])
            else:
                dic[key] = func(dic[key])
    return dic


def dicts_foreach(dict_list, func, special_func={}):
    """
    Recursively apply a function to all non-dictionary leaf values in a list of dictionaries.
    """
    assert isinstance(dict_list, list), 'input must be a list of dictionaries'
    assert all([isinstance(d, dict) for d in dict_list]), 'input must be a list of dictionaries'
    assert len(dict_list) > 0, 'input must be a non-empty list of dictionaries'

    dic = {}
    for key in dict_list[0].keys():
        if isinstance(dict_list[0][key], dict):
            dic[key] = dicts_foreach([d[key] for d in dict_list], func, special_func)
        else:
            if key in special_func.keys():
                dic[key] = special_func[key]([d[key] for d in dict_list])
            else:
                dic[key] = func([d[key] for d in dict_list])
    return dic


def dict_reduce(dicts, func, special_func={}):
    """
    Reduce a list of dictionaries. Leaf values must be scalars.
    """
    assert isinstance(dicts, list), 'input must be a list of dictionaries'
    assert all([isinstance(d, dict) for d in dicts]), 'input must be a list of dictionaries'
    assert len(dicts) > 0, 'input must be a non-empty list of dictionaries'
    all_keys = set([key for dict_ in dicts for key in dict_.keys()])
    reduced_dict = {}
    for key in all_keys:
        vlist = [dict_[key] for dict_ in dicts if key in dict_.keys()]
        if isinstance(vlist[0], dict):
            reduced_dict[key] = dict_reduce(vlist, func, special_func)
        else:
            if key in special_func.keys():
                reduced_dict[key] = special_func[key](vlist)
            else:
                reduced_dict[key] = func(vlist)
    return reduced_dict


def dict_any(dic, func):
    """
    Recursively apply a function to all non-dictionary leaf values in a dictionary.
    """
    assert isinstance(dic, dict), 'input must be a dictionary'
    for key in dic.keys():
        if isinstance(dic[key], dict):
            if dict_any(dic[key], func):
                return True
        else:
            if func(dic[key]):
                return True
    return False


def dict_all(dic, func):
    """
    Recursively apply a function to all non-dictionary leaf values in a dictionary.
    """
    assert isinstance(dic, dict), 'input must be a dictionary'
    for key in dic.keys():
        if isinstance(dic[key], dict):
            if not dict_all(dic[key], func):
                return False
        else:
            if not func(dic[key]):
                return False
    return True


def dict_flatten(dic, sep='.'):
    """
    Flatten a nested dictionary into a dictionary with no nested dictionaries.
    """
    assert isinstance(dic, dict), 'input must be a dictionary'
    flat_dict = {}
    for key in dic.keys():
        if isinstance(dic[key], dict):
            sub_dict = dict_flatten(dic[key], sep=sep)
            for sub_key in sub_dict.keys():
                flat_dict[str(key) + sep + str(sub_key)] = sub_dict[sub_key]
        else:
            flat_dict[key] = dic[key]
    return flat_dict


def json_safe(value):
    """
    Convert tensors/numpy values to JSON-serializable Python types.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(val) for val in value]
    return value


# Config utils
def apply_config_overrides(config, overrides):
    """
    Apply dot-separated key=value overrides to a nested config dict.
    Values are auto-cast to int, float, bool, None, or kept as str.
    Example: "models.denoiser.args.num_view_embeds=4"
    """
    def _cast(v):
        if v.lower() == 'true': return True
        if v.lower() == 'false': return False
        if v.lower() == 'none' or v.lower() == 'null': return None
        try: return int(v)
        except ValueError: pass
        try: return float(v)
        except ValueError: pass
        return v

    for override in overrides:
        if '=' not in override:
            raise ValueError(f'Invalid override format: {override}. Expected key=value')
        key, value = override.split('=', 1)
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = _cast(value)
    return config


# RNG utils
def setup_rng(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed_all(rank)
    np.random.seed(rank)
    random.seed(rank)


# Context utils
@contextlib.contextmanager
def nested_contexts(*contexts):
    with contextlib.ExitStack() as stack:
        for ctx in contexts:
            stack.enter_context(ctx())
        yield


# Image utils
def make_grid(images, nrow=None, ncol=None, aspect_ratio=None):
    num_images = len(images)
    if nrow is None and ncol is None:
        if aspect_ratio is not None:
            nrow = int(np.round(np.sqrt(num_images / aspect_ratio)))
        else:
            nrow = int(np.sqrt(num_images))
        ncol = (num_images + nrow - 1) // nrow
    elif nrow is None and ncol is not None:
        nrow = (num_images + ncol - 1) // ncol
    elif nrow is not None and ncol is None:
        ncol = (num_images + nrow - 1) // nrow
    else:
        assert nrow * ncol >= num_images, 'nrow * ncol must be greater than or equal to the number of images'
    
    if images[0].ndim == 2:
        grid = np.zeros((nrow * images[0].shape[0], ncol * images[0].shape[1]), dtype=images[0].dtype)
    else:
        grid = np.zeros((nrow * images[0].shape[0], ncol * images[0].shape[1], images[0].shape[2]), dtype=images[0].dtype)
    for i, img in enumerate(images):
        row = i // ncol
        col = i % ncol
        grid[row * img.shape[0]:(row + 1) * img.shape[0], col * img.shape[1]:(col + 1) * img.shape[1]] = img
    return grid


def notes_on_image(img, notes=None):
    img = np.pad(img, ((0, 32), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if notes is not None:
        img = cv2.putText(img, notes, (0, img.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image_with_notes(img, path, notes=None):
    """
    Save an image with notes.
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy().transpose(1, 2, 0)
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img = notes_on_image(img, notes)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def make_mv_subgrids(val, value_range=(0, 1)):
    """
    Convert a [B, V, C, H, W] multi-view tensor into a [B, C, H', W'] subgrid tensor.
    Each sample's V views are arranged in a square grid; empty slots are filled with
    a random solid colour so that borders between samples remain distinguishable.
    """
    from torchvision import utils as tv_utils
    B, V, C, H, W = val.shape
    sub_nrow = int(np.ceil(np.sqrt(V)))
    pad_count = sub_nrow * sub_nrow - V
    if pad_count > 0:
        rand_bg = torch.rand(B, 1, C, 1, 1, device=val.device).expand(B, pad_count, C, H, W)
        val = torch.cat([val, rand_bg], dim=1)
    subgrids = torch.stack(
        [tv_utils.make_grid(val[i], nrow=sub_nrow, normalize=True, value_range=value_range)
         for i in range(B)],
        dim=0,
    )
    return subgrids  # [B, C, sub_H, sub_W]


# debug utils

def atol(x, y):
    """
    Absolute tolerance.
    """
    return torch.abs(x - y)


def rtol(x, y):
    """
    Relative tolerance.
    """
    return torch.abs(x - y) / torch.clamp_min(torch.maximum(torch.abs(x), torch.abs(y)), 1e-12)


# print utils
def indent(s, n=4):
    """
    Indent a string.
    """
    lines = s.split('\n')
    for i in range(1, len(lines)):
        lines[i] = ' ' * n + lines[i]
    return '\n'.join(lines)


# Model utils

def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Unwrap DDP and torch.compile wrappers for checkpoint / parameter access.
    """
    from torch.nn.parallel import DistributedDataParallel as DDP
    if isinstance(model, DDP):
        model = model.module
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
    return model


def get_model_summary(model):
    """
    Get a summary string of model parameters, handling torch.compile wrappers.
    """
    summary = 'Parameters:\n'
    summary += '=' * 128 + '\n'
    summary += f'{"Name":<{72}}{"Shape":<{32}}{"Type":<{16}}{"Grad"}\n'
    num_params = 0
    num_trainable_params = 0
    raw = unwrap_model(model)
    is_compiled = raw is not model
    for name, param in raw.named_parameters():
        summary += f'{name:<{72}}{str(param.shape):<{32}}{str(param.dtype):<{16}}{param.requires_grad}\n'
        num_params += param.numel()
        if param.requires_grad:
            num_trainable_params += param.numel()
    summary += '\n'
    suffix = ' (torch.compile model)' if is_compiled else ''
    summary += f'Number of parameters: {num_params}{suffix}\n'
    summary += f'Number of trainable parameters: {num_trainable_params}{suffix}\n'
    return summary


def build_model(model_cfg, *, device='cuda'):
    """
    Build a model from config with optional torch.compile.

    Supports three config forms:
      - str: load via models.from_pretrained(model_cfg)
      - dict with 'name'/'args': instantiate from class name and constructor args
      - dict with 'path': load a pretrained model from path (via models.from_pretrained)

    Optional 'compile' key (for dict forms): True for default compile, or dict of torch.compile kwargs.
    """
    from trellis import models

    if isinstance(model_cfg, str):
        return models.from_pretrained(model_cfg).to(device)

    if 'name' in model_cfg:
        model = getattr(models, model_cfg['name'])(**model_cfg['args']).to(device)
    else:
        model = models.from_pretrained(model_cfg['path']).to(device)

    compile_cfg = model_cfg.get('compile', False)
    if compile_cfg:
        if not hasattr(torch, 'compile'):
            raise RuntimeError('torch.compile requested but not available in this torch build')
        if isinstance(compile_cfg, dict):
            model = torch.compile(model, **compile_cfg)
        else:
            model = torch.compile(model)

    return model

