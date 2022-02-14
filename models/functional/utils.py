"""
Collection of utility functions.
"""

from models.functional.autograd import CustomClamp, CustomOnes


def custom_clamp(in_tensor, min=None, max=None):
    """
    Custom clamp function altering the backward pass by computing gradients as if no clamping occurred in the forward
    pass.

    Args:
        in_tensor (FloatTensor): Input tensor of arbitrary shape.
        min (float): Minimum value of clamped tensor (default=None).
        max (float): Maximum value of clamped tensor (default=None).

    Returns:
        out_tensor (FloatTensor): Clamped output tensor of same shape as input tensor.
    """

    # Apply custom clamp function
    out_tensor = CustomClamp.apply(in_tensor, min, max)

    return out_tensor


def custom_ones(in_tensor):
    """
    Custom ones function altering the backward pass by computing gradients as if the ones operation in the forward pass
    did not occur.

    Args:
        in_tensor (FloatTensor): Input tensor of arbitrary shape.

    Returns:
        out_tensor (FloatTensor): Output tensor filled with ones of same shape as input tensor.
    """

    # Apply custom ones function
    out_tensor = CustomOnes.apply(in_tensor)

    return out_tensor
