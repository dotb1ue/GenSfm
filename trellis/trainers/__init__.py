import importlib

__attributes = {
    'BasicTrainer': 'basic',
    
    'SparseUVStructureVaeTrainer': 'vae.sparse_uv_structure_vae',
    
    
    'FlowMatchingTrainer': 'flow_matching.flow_matching',
    'FlowMatchingCFGTrainer': 'flow_matching.flow_matching',
    'MultiPoseFlowMatchingCFGTrainer': 'flow_matching.flow_matching',
    'VisualImageConditionedMultiPoseFlowMatchingCFGTrainer': 'flow_matching.flow_matching',
    
    'SparseFlowMatchingTrainer': 'flow_matching.sparse_flow_matching',
    'SparseFlowMatchingCFGTrainer': 'flow_matching.sparse_flow_matching',
    'MultiVisualImageConditionedSparseFlowMatchingCFGTrainer': 'flow_matching.sparse_flow_matching',
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
    from .basic import BasicTrainer

    from .vae.sparse_uv_structure_vae import SparseUVStructureVaeTrainer
    
    from .flow_matching.flow_matching import (
        FlowMatchingTrainer,
        FlowMatchingCFGTrainer,
        MultiPoseFlowMatchingCFGTrainer,
        VisualImageConditionedMultiPoseFlowMatchingCFGTrainer,
    )
    
    from .flow_matching.sparse_flow_matching import (
        SparseFlowMatchingTrainer,
        SparseFlowMatchingCFGTrainer,
        MultiVisualImageConditionedSparseFlowMatchingCFGTrainer,
    )
