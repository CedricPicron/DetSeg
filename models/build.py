"""
Function building registered models.
"""

from mmcv.utils import Registry


MODELS = Registry('models')


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
