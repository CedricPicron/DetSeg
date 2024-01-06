"""
Collection of modules based on PyTorch functions.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from models.build import MODELS


@MODELS.register_module()
class Add(nn.Module):
    """
    Class implementing the Add module.

    Attributes:
        in_keys (List): List of strings with keys to retrieve input tensors from storage dictionary.
        out_key (str): String with key to store addition output tensor in storage dictionary.
    """

    def __init__(self, in_keys, out_key):
        """
        Initializes the Add module.

        Args:
            in_keys (List): List of strings with keys to retrieve input tensors from storage dictionary.
            out_key (str): String with key to store addition output tensor in storage dictionary.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set additional attributes
        self.in_keys = in_keys
        self.out_key = out_key

    def forward(self, storage_dict, **kwargs):
        """
        Forward method of the Add module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - {in_key} (Tensor): input tensor to be added of shape [*].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional key:
                - {self.out_key} (Tensor): addition output tensor of shape [*].
        """

        # Retrieve input tensors from storage dictionary
        in_list = [storage_dict[in_key] for in_key in self.in_keys]

        # Get output tensor
        out_tensor = sum(in_list)

        # Store output tensor in storage dictionary
        storage_dict[self.out_key] = out_tensor

        return storage_dict


@MODELS.register_module()
class Cat(nn.Module):
    """
    Class implementing the Cat module.

    Attributes:
        in_keys (List): List of strings with keys to retrieve input tensors from storage dictionary.
        out_key (str): String with key to store concatenated output tensor in storage dictionary.
        dim (int): Integer containing the dimension along which to concatenate.
    """

    def __init__(self, in_keys, out_key, dim=0):
        """
        Initializes the Cat module.

        Args:
            in_keys (List): List of strings with keys to retrieve input tensors from storage dictionary.
            out_key (str): String with key to store concatenated output tensor in storage dictionary.
            dim (int): Integer containing the dimension along which to concatenate (default=0).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set additional attributes
        self.in_keys = in_keys
        self.out_key = out_key
        self.dim = dim

    def forward(self, storage_dict, **kwargs):
        """
        Forward method of the Cat module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - {in_key} (Tensor): input tensor to be concatenated of shape [*, in_size, *].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional key:
                - {self.out_key} (Tensor): concatenated output tensor of shape [*, sum(in_size), *].
        """

        # Retrieve input tensors from storage dictionary
        in_list = [storage_dict[in_key] for in_key in self.in_keys]

        # Get output tensor
        out_tensor = torch.cat(in_list, dim=self.dim)

        # Store output tensor in storage dictionary
        storage_dict[self.out_key] = out_tensor

        return storage_dict


@MODELS.register_module()
class Exp(nn.Module):
    """
    Class implementing the Exp module computing the exponential of the elements of the input tensor.
    """

    def __init__(self):
        """
        Initializes the Exp module.
        """

        # Initialization of default nn.Module
        super().__init__()

    def forward(self, in_tensor):
        """
        Forward method of the Exp module.

        Args:
            in_tensor (FloatTensor): Input tensor of arbitrary shape.

        Returns:
            out_tensor (FloatTensor): Output tensor of same shape as input tensor.
        """

        # Get output tensor
        out_tensor = torch.exp(in_tensor)

        return out_tensor


@MODELS.register_module()
class Float(nn.Module):
    """
    Class implementing the Float module.
    """

    def __init__(self):
        """
        Initializes the Float module.
        """

        # Initialization of default nn.Module
        super().__init__()

    def forward(self, in_tensor, **kwargs):
        """
        Forward method of the Float module.

        Args:
            in_tensor (Tensor): Input tensor with arbitrary data type.
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            out_tensor (FloatTensor): Output tensor with float data type.
        """

        # Get output tensor
        out_tensor = in_tensor.float()

        return out_tensor


@MODELS.register_module()
class Interpolate(nn.Module):
    """
    Class implementing the Interpolate module.

    Attributes:
        size (int or tuple): Integer or tuple containing the output spatial size.
        scale_factor (float or Tuple): Value or tuple of values scaling the spatial.
        mode (str): String containing the interpolation mode.
        align_corners (bool): Boolean indicating whether values at corners are preserved.
    """

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        """
        Initializes the Interpolate module.

        Args:
            size (int or tuple): Integer or tuple containing the output spatial size (default=None).
            scale_factor (float or Tuple): Value or tuple of values scaling the spatial (default=None).
            mode (str): String containing the interpolation mode (default='nearest').
            align_corners (bool): Boolean indicating whether values at corners are preserved (default=False).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attributes
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, in_tensor, **kwargs):
        """
        Forward method of the Interpolate module.

        Args:
            in_tensor (FloatTensor): Input tensor to interpolate.
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            out_tensor (FloatTensor): Output tensor obtained by interpolating the input tensor.
        """

        # Get output tensor by interpolating input tensor
        out_tensor = F.interpolate(in_tensor, self.size, self.scale_factor, self.mode, self.align_corners)

        return out_tensor


@MODELS.register_module()
class Mul(nn.Module):
    """
    Class implementing the Mul module performing element-wise multiplication with optional addition.

    Attributes:
        factor (Parameter): Tensor containing the mulitplication factor of shape [1] or [feat_size].
        bias (Parameter): Tensor containing the optional addition term of shape [1] or [feat_size] (None when missing).
    """

    def __init__(self, feat_dependent=False, feat_size=None, init_factor=1.0, learn_factor=True, bias=False,
                 init_bias=0.0, learn_bias=True):
        """
        Initializes the Mul module.

        Args:
            feat_dependent (bool): Boolean indicating whether parameters should be feature dependent (default=False).
            feat_size (float): Value containing the expected input feature size (default=None).
            init_factor (float): Value containing the initial multiplication factor (default=1.0).
            learn_factor (bool): Boolean indicating whether factor parameter should be learned (default=True).
            bias (bool): Boolean indicating whether bias should be added after multiplication (default=False).
            init_bias (float): Value containing the initial bias value (default=0.0).
            learn_bias (bool): Boolean indicating whether bias parameter should be learned (default=True).

        Raises:
            ValueError: Error when 'feat_dependent' is True and no feature size is provided.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Get factor and optional bias parameters
        if not feat_dependent:
            self.factor = Parameter(torch.tensor([init_factor]), requires_grad=learn_factor)
            self.bias = Parameter(torch.tensor([init_bias]), requires_grad=learn_bias) if bias else None

        elif feat_size is not None:
            self.factor = Parameter(torch.full((feat_size,), init_factor), requires_grad=learn_factor)
            self.bias = Parameter(torch.full((feat_size,), init_bias), requires_grad=learn_bias) if bias else None

        else:
            error_msg = "A feature size must be provided when 'feat_dependent' is True."
            raise ValueError(error_msg)

    def forward(self, in_tensor):
        """
        Forward method of the Mul module.

        Args:
            in_tensor (FloatTensor): Input tensor of shape [*, feat_size].

        Returns:
            out_tensor (FloatTensor): Output tensor of shape [*, feat_size].
        """

        # Get output tensor
        if self.bias is None:
            out_tensor = torch.mul(in_tensor, self.factor)
        else:
            out_tensor = torch.addcmul(self.bias, in_tensor, self.factor)

        return out_tensor


@MODELS.register_module()
class Permute(nn.Module):
    """
    Class implementing the Permute module.

    Attributes:
        dims (List): List of integers containing the output ordering of dimensions.
    """

    def __init__(self, dims):
        """
        Initializes the Permute module.

        Args:
            dims (List): List of integers containing the output ordering of dimensions.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set dims attribute
        self.dims = dims

    def forward(self, in_tensor, **kwargs):
        """
        Forward method of the Permute module.

        Args:
            in_tensor (Tensor): Input tensor to be permuted.
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            out_tensor (Tensor): Permuted output tensor.
        """

        # Get output tensor
        out_tensor = torch.permute(in_tensor, self.dims)

        return out_tensor


@MODELS.register_module()
class Topk(nn.Module):
    """
    Class implementing the Topk module.

    Attributes:
        in_key (str): String with key to retrieve input tensor from storage dictionary.
        out_vals_key (str): String with key to store topk output values in storage dictionary (or None).
        out_ids_key (str): String with key to store topk output indices in storage dictionary (or None).
        topk_kwargs (Dict): Dictionary of keyword arguments specifying the topk operation.
    """

    def __init__(self, in_key, topk_kwargs, out_vals_key=None, out_ids_key=None):
        """
        Initializes the Topk module.

        Args:
            in_key (str): String with key to retrieve input tensor from storage dictionary.
            topk_kwargs (Dict): Dictionary of keyword arguments specifying the topk operation.
            out_vals_key (str): String with key to store topk output values in storage dictionary (default=None).
            out_ids_key (str): String with key to store topk output indices in storage dictionary (default=None).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set additional attributes
        self.in_key = in_key
        self.out_vals_key = out_vals_key
        self.out_ids_key = out_ids_key
        self.topk_kwargs = topk_kwargs

    def forward(self, storage_dict, **kwargs):
        """
        Forward method of the Topk module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following key:
                - {self.in_key} (Tensor): input tensor on which to apply the topk operation.

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            storage_dict (Dict): Storage dictionary possibly containing following additional keys:
                - {self.out_vals_key} (Tensor): output tensor with topk values;
                - {self.out_ids_key} (LongTensor): output tensor with indices corresponding to the topk values.
        """

        # Retrieve input tensor from storage dictionary
        in_tensor = storage_dict[self.in_key]

        # Get topk values and indices
        out_vals, out_ids = torch.topk(in_tensor, **self.topk_kwargs)

        # Store outputs in storage dictionary if needed
        if self.out_vals_key is not None:
            storage_dict[self.out_vals_key] = out_vals

        if self.out_ids_key is not None:
            storage_dict[self.out_ids_key] = out_ids

        return storage_dict


@MODELS.register_module()
class Transpose(nn.Module):
    """
    Class implementing the Transpose module.

    Attributes:
        dim0 (int): Integer containing the first dimension to be transposed.
        dim1 (int): Integer containing the second dimension to be transposed.
    """

    def __init__(self, dim0, dim1):
        """
        Initializes the Transpose module.

        Args:
            dim0 (int): Integer containing the first dimension to be transposed.
            dim1 (int): Integer containing the second dimension to be transposed.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set additional attributes
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, in_tensor, **kwargs):
        """
        Forward method of the Transpose module.

        Args:
            in_tensor (Tensor): Input tensor to be transposed.
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            out_tensor (Tensor): Transposed output tensor.
        """

        # Get output tensor
        out_tensor = torch.transpose(in_tensor, self.dim0, self.dim1)

        return out_tensor


@MODELS.register_module()
class Unsqueeze(nn.Module):
    """
    Class implementing the Unsqueeze module.

    Attributes:
        dim (int): Integer containing the dimension at which to insert singleton dimension.
    """

    def __init__(self, dim):
        """
        Initializes the Unsqueeze module.

        Args:
            dim (int): Integer containing the dimension at which to insert singleton dimension.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set dim attribute
        self.dim = dim

    def forward(self, in_tensor, **kwargs):
        """
        Forward method of the Unsqueeze module.

        Args:
            in_tensor (Tensor): Input tensor for which to add a singleton dimension.
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            out_tensor (FloatTensor): Output tensor containing an additional singleton dimension.
        """

        # Get output tensor
        out_tensor = torch.unsqueeze(in_tensor, dim=self.dim)

        return out_tensor


@MODELS.register_module()
class View(nn.Module):
    """
    Class implementing the View module.

    Attributes:
        out_shape (Tuple): Tuple containing the output shape.
    """

    def __init__(self, out_shape):
        """
        Initializes the View module.

        Args:
            out_shape (Tuple): Tuple containing the output shape.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set output shape attribute
        self.out_shape = out_shape

    def forward(self, in_tensor, **kwargs):
        """
        Forward method of the View module.

        Args:
            in_tensor (Tensor): Input tensor of shape [*in_shape].
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            out_tensor (Tensor): Output tensor of shape [*out_shape].
        """

        # Get output tensor
        out_tensor = in_tensor.view(*self.out_shape)

        return out_tensor
