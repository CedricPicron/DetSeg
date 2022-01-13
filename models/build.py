"""
Function building registered models.
"""

from mmdet.models.builder import MODELS as MMDET_MODELS
from mmcv.utils import Registry

# Create registry
MODELS = Registry('models')

# Add modules from MMDetection
MODELS._add_children(MMDET_MODELS)


def build_model(cfg):
    """
    Function building a model from the given configuration dictionary.

    The model type must registered in the MODELS registry.

    Args:
        cfg (Dict): Configuration dictionary specifying the model to be built.

    Returns:
        model (nn.Module): Model built from the given configuration dictionary.
    """

    # Build model from configuration dictionary
    model = MODELS.build(cfg)

    return model
