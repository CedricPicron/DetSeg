"""
Collection of activation functions.
"""

from models.functional.autograd import CustomReLU, CustomStep


def custom_relu(in_tensor, zero_grad_thr=0.0):
    """
    Custom ReLU activation function altering the backward pass by only setting an input gradient to zero if the input
    is lower than the zero gradient threshold and higher input values would result in a higher loss.

    Args:
        in_tensor (FloatTensor): Input tensor of arbitrary shape.
        zero_grad_thr (float): Value containing the zero gradient threshold (default=0.0).

    Returns:
        out_tensor (FloatTensor): Output tensor of same shape as input tensor.
    """

    # Apply custom ReLU
    out_tensor = CustomReLU.apply(in_tensor, zero_grad_thr)

    return out_tensor


def custom_step(in_tensor, left_zero_grad_thr=0.0, right_zero_grad_thr=0.0):
    """
    Custom step activation function altering the backward pass by only setting an input gradient to zero if the input
    is lower than the left zero gradient threshold and higher input values would result in a higher loss, or if the
    input is higher than the right zero gradient threshold and lower input values would result in a higher loss.

    Args:
        in_tensor (FloatTensor): Input tensor of arbitrary shape.
        left_zero_grad_thr (float): Value containing the left zero gradient threshold (default=0.0).
        right_zero_grad_thr (float): Value containing the right zero gradient threshold (default=0.0).

    Returns:
        out_tensor (FloatTensor): Output tensor of same shape as input tensor.
    """

    # Apply custom step function
    out_tensor = CustomStep.apply(in_tensor, left_zero_grad_thr, right_zero_grad_thr)

    return out_tensor
