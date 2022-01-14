"""
Function building registered models.
"""

from mmcv.utils import build_from_cfg, Registry
from mmdet.models.builder import MODELS as MMDET_MODELS
from torch import nn


def build_model_from_cfg(cfg, registry, default_args=None):
    """
    Build model from one or multiple configuration dictionaries.

    It extends the default 'build_from_cfg' from 'mmcv' by:
        1) allowing a list of configuration dictionaries specifying various sub-modules to be concatenated;
        2) allowing a 'num_layers' key specifying multiple consecutive instances of the same sub-module.

    Args:
        cfg (Dict, List[Dict]): One or multiple configuration dictionaries specifying the model to be built.
        registry (Registry): A registry containing various model types.
        default_args (Dict): Default arguments to build the model (default=None).

    Returns:
        model (nn.Module): Model built from the given configuration dictionary.
    """

    # Build sub-modules
    build_args = (registry, default_args)
    cfg = [cfg] if not isinstance(cfg, list) else cfg
    modules = [build_from_cfg(cfg_i, *build_args) for cfg_i in cfg for _ in range(cfg_i.pop('num_layers', 1))]

    # Concatenate sub-modules if needed
    seq_module = registry.get('Sequential')
    model = seq_module(*modules) if len(modules) > 1 else modules[0]

    return model


# Create registry
MODELS = Registry('models', build_func=build_model_from_cfg)

# Add modules from MMDetection
MODELS._add_children(MMDET_MODELS)

# Add modules from torch.nn
NN_MODELS = Registry('models')
NN_MODELS._scope = 'nn'
[NN_MODELS.register_module(nn.modules.__dict__[name]) for name in nn.modules.__all__]
MODELS._add_children(NN_MODELS)


def build_model(cfg):
    """
    Function building a model from the given configuration dictionary.

    The model type must registered in the MODELS registry.

    Args:
        cfg (Dict, List[Dict]): One or multiple configuration dictionaries specifying the model to be built.

    Returns:
        model (nn.Module): Model built from the given configuration dictionary.
    """

    # Build model from given configuration
    model = MODELS.build(cfg)

    return model
