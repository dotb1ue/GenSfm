import importlib


_PIPELINE_MODULES = {
    'GenSfMPipeline': '.gensfm',
}
__all__ = ['samplers', *_PIPELINE_MODULES, 'from_pretrained']


def __getattr__(name):
    if name == 'samplers':
        value = importlib.import_module('.samplers', __name__)
    elif name in _PIPELINE_MODULES:
        value = getattr(importlib.import_module(_PIPELINE_MODULES[name], __name__), name)
    else:
        raise AttributeError(f'module {__name__} has no attribute {name}')
    globals()[name] = value
    return value


def from_pretrained(path: str, revision=None):
    """Load a native GenSfM pipeline from a local bundle or Hugging Face."""
    return __getattr__('GenSfMPipeline').from_pretrained(path, revision=revision)
