import importlib

__attributes = {
    'SparseUVStructureVaeEvaluator': 'vae.sparse_uv_structure_vae',

    'FlowMatchingEvaluator': 'flow_matching.flow_matching',
    'MultiPoseFlowMatchingEvaluator': 'flow_matching.flow_matching',
    'MultiPoseFlowMatchingCFGEvaluator': 'flow_matching.flow_matching',

    'SparseFlowMatchingEvaluator': 'flow_matching.sparse_flow_matching',
    'MultiPoseSparseFlowMatchingEvaluator': 'flow_matching.sparse_flow_matching',
    'MultiPoseSparseFlowMatchingCFGEvaluator': 'flow_matching.sparse_flow_matching',
    'VisualImageConditionedMultiPoseSparseFlowMatchingEvaluator': 'flow_matching.sparse_flow_matching',
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


# For Pylance
if __name__ == '__main__':
    from .vae.sparse_uv_structure_vae import SparseUVStructureVaeEvaluator

    from .flow_matching.flow_matching import (
        FlowMatchingEvaluator,
        MultiPoseFlowMatchingEvaluator,
        MultiPoseFlowMatchingCFGEvaluator,
    )

    from .flow_matching.sparse_flow_matching import (
        SparseFlowMatchingEvaluator,
        MultiPoseSparseFlowMatchingEvaluator,
        MultiPoseSparseFlowMatchingCFGEvaluator,
        VisualImageConditionedMultiPoseSparseFlowMatchingEvaluator,
    )
