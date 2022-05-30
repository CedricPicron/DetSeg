"""
Function building registered models.
"""
from copy import deepcopy

from mmcv.cnn import initialize
from mmcv.utils import build_from_cfg, Registry
from mmdet.core.anchor.builder import PRIOR_GENERATORS
from mmdet.core.bbox.builder import BBOX_ASSIGNERS, BBOX_SAMPLERS, BBOX_CODERS
from mmdet.core.bbox.iou_calculators.builder import IOU_CALCULATORS
from mmdet.core.bbox.match_costs.builder import MATCH_COST
from mmdet.models.builder import MODELS as MMDET_MODELS
from torch import nn


def build_model_from_cfg(cfg, registry, sequential=False, **kwargs):
    """
    Build model from one or multiple configuration dictionaries.

    It extends the default 'build_from_cfg' from MMCV by:
        1) allowing a list of configuration dictionaries specifying various sub-modules to be concatenated;
        2) allowing a 'num_layers' key specifying multiple consecutive instances of the same sub-module.

    Args:
        cfg (Dict, List[Dict]): One or multiple configuration dictionaries specifying the model to be built.
        registry (Registry): A registry containing various model types.
        sequential (bool): Boolean indicating whether the model should be an instance of Sequential (default=False).
        kwargs (Dict): Dictionary of keyword arguments passed to the underlying 'build_from_cfg' function from MMCV.

    Returns:
        model (nn.Module): Model built from the given configuration dictionary.
    """

    # Some preparation
    cfg = [cfg] if not isinstance(cfg, list) else cfg
    modules = []

    # Build sub-modules
    for cfg_i in cfg:
        cfg_i = deepcopy(cfg_i)
        init_cfg = cfg_i.pop('init_cfg', None)

        for _ in range(cfg_i.pop('num_layers', 1)):
            module = build_from_cfg(cfg_i, registry, kwargs)
            initialize(module, init_cfg) if init_cfg is not None else None
            modules.append(module)

    # Concatenate sub-modules if needed
    seq_module = registry.get('Sequential')
    model = seq_module(*modules) if (len(modules) > 1 or sequential) else modules[0]

    return model


# Create registry
MODELS = Registry('models', build_func=build_model_from_cfg)

# Add modules from MMDetection
[MMDET_MODELS.register_module(module) for module in PRIOR_GENERATORS.module_dict.values()]
[MMDET_MODELS.register_module(module) for module in BBOX_ASSIGNERS.module_dict.values()]
[MMDET_MODELS.register_module(module) for module in BBOX_SAMPLERS.module_dict.values()]
[MMDET_MODELS.register_module(module) for module in BBOX_CODERS.module_dict.values()]
[MMDET_MODELS.register_module(module) for module in IOU_CALCULATORS.module_dict.values()]
[MMDET_MODELS.register_module(module) for module in MATCH_COST.module_dict.values()]
MODELS._add_children(MMDET_MODELS)

# Add modules from torch.nn
NN_MODELS = Registry('models')
NN_MODELS._scope = 'nn'
[NN_MODELS.register_module(nn.modules.__dict__[name]) for name in nn.modules.__all__]
MODELS._add_children(NN_MODELS)


def build_model(cfg, sequential=False, **kwargs):
    """
    Function building a model from the given configuration dictionary.

    The model type must registered in the MODELS registry.

    Args:
        cfg (Dict, List[Dict]): One or multiple configuration dictionaries specifying the model to be built.
        sequential (bool): Boolean indicating whether the model should be an instance of Sequential (default=False).
        kwargs (Dict): Dictionary of keyword arguments passed to the build function of the MODELS registry.

    Returns:
        model (nn.Module): Model built from the given configuration dictionary.
    """

    # Build model from given configuration
    model = MODELS.build(cfg, sequential=sequential, **kwargs)

    return model
