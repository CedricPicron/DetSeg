"""
Collection of feature maps to feature maps operations.
"""

import torch
from torch import nn
import torch.nn.functional as F


def initialize_add(operation, feat_sizes):
    """
    Initializes adding operation for forward computation.

    Args:
        operation (dict): Dictionary specifying the operation to be performed.
        feat_sizes (List): List of feature sizes of feature maps to be expected during forward computation.

    Returns:
        sub_operations (List): List of sub-operation dictionaries specifying their behavior during forward computation.
        feat_sizes (List): Updated list of feature sizes of feature maps to be generated during forward computation.

    Raises:
        ValueError: Error when the number of inputs of an add sub-operation is different from two.
    """

    # Check operation dictionary
    for map_ids in operation['in']:
        if len(map_ids) != 2:
            raise ValueError(f"Exactly two inputs are required per add sub-operation (got {len(map_ids)}).")

    # Get sub-operation keyword arguments and initialize empty list of sub-operations
    allowed_keys = ['alpha']
    sub_operation_kwargs = {k: v for k, v in operation.items() if k in allowed_keys}
    sub_operations = []

    # Initialize every add operation
    for map_ids in operation['in']:
        sub_operation = {'function': torch.add, 'in_map_ids': map_ids, **sub_operation_kwargs}
        feat_size = feat_sizes[map_ids[0]]

        sub_operations.append(sub_operation)
        feat_sizes.append(feat_size)

    return sub_operations, feat_sizes


def initialize_conv2d(operation, feat_sizes, modules_offset):
    """
    Initializes 2D convolution operation with corresponding modules for forward computation.

    Args:
        operation (dict): Dictionary specifying the operation to be performed.
        feat_sizes (List): List of feature sizes of feature maps to be expected during forward computation.
        modules_offset (int): Integer containing the offset to be added to the module indices.

    Returns:
        modules (List): List of initialized modules used by this operation.
        sub_operations (List): List of sub-operation dictionaries specifying their behavior during forward computation.
        feat_sizes (List): Updated list of feature sizes of feature maps to be generated during forward computation.

    Raises:
        ValueError: Error when incorrect output channels input is provided.
    """

    # Get lists of number input and output channels
    in_channels_list = [feat_sizes[i] for i in operation['in']]
    out_channels_list = [operation['out_channels']]

    if len(in_channels_list) > 1 and len(out_channels_list) == 1:
        out_channels_list = [out_channels_list[0]] * len(in_channels_list)
    elif len(in_channels_list) != len(out_channels_list) and len(out_channels_list) > 1:
        raise ValueError("Size of input and output channels list must match when multiple output channels are given.")

    # Get 2D convolution keyword arguments
    allowed_keys = ['kernel_size', 'stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode']
    sub_operation_kwargs = {k: v for k, v in operation.items() if k in allowed_keys}

    # Initialize list of modules and sub-operations
    modules = []
    sub_operations = []

    # Initialize every 2D convolution sub-operation
    for map_id, in_channels, out_channels in zip(operation['in'], in_channels_list, out_channels_list):
        module = nn.Conv2d(in_channels, out_channels, **sub_operation_kwargs)
        sub_operation = {'in_map_ids': [map_id], 'module_id': modules_offset+len(modules)}

        if 'weight_init' in operation:
            init_kwargs = {k.replace('weight_init_', ''): v for k, v in operation.items() if 'weight_init_' in k}
            getattr(nn.init, operation['weight_init'])(module.weight, **init_kwargs)

        if 'bias_init' in operation:
            init_kwargs = {k.replace('bias_init_', ''): v for k, v in operation.items() if 'bias_init_' in k}
            getattr(nn.init, operation['bias_init'])(module.bias, **init_kwargs)

        modules.append(module)
        sub_operations.append(sub_operation)
        feat_sizes.append(out_channels)

    return modules, sub_operations, feat_sizes


def initialize_interpolate(operation, feat_sizes):
    """
    Initializes interpolate operation for forward computation.

    Args:
        operation (dict): Dictionary specifying the operation to be performed.
        feat_sizes (List): List of feature sizes of feature maps to be expected during forward computation.

    Returns:
        sub_operations (List): List of sub-operation dictionaries specifying their behavior during forward computation.
        feat_sizes (List): Updated list of feature sizes of feature maps to be generated during forward computation.

    Raises:
        ValueError: Error when sizes of the 'in' and 'shape' keys from the operation dictionary do not match.
        ValueError: Error when both 'shape' and 'size' keys are present in the operation dictionary.
    """

    # Pre-process and check operation dictionary when 'shape' key is present
    if 'shape' in operation:
        if isinstance(operation['shape'], int):
            operation['shape'] = [operation['shape']]

        if len(operation['in']) != len(operation['shape']):
            raise ValueError(f"Sizes of 'in' and 'shape' keys from operation dictionary {operation} must match.")

        if 'size' in operation:
            raise ValueError(f"The 'shape' and 'size' keys shouldn't be in the same operation dictionary {operation}.")

    # Get sub-operation keyword arguments and initialize empty list of sub-operations
    allowed_keys = ['size', 'scale_factor', 'mode', 'align_corners', 'recompute_scale_factor']
    sub_operation_kwargs = {k: v for k, v in operation.items() if k in allowed_keys}
    sub_operations = []

    # Initialize every interpolate operation
    for i, map_id in enumerate(operation['in']):
        if 'shape' in operation:
            def shape_interpolate(x, y, **kwargs):
                return F.interpolate(x, size=y.shape[-2:], **kwargs)

            function = shape_interpolate
            in_map_ids = [map_id, operation['shape'][i]]

        else:
            function = F.interpolate
            in_map_ids = [map_id]

        sub_operation = {'function': function, 'in_map_ids': in_map_ids, **sub_operation_kwargs}
        feat_size = feat_sizes[map_id]

        sub_operations.append(sub_operation)
        feat_sizes.append(feat_size)

    return sub_operations, feat_sizes


def initialize_relu(operation, feat_sizes):
    """
    Initializes ReLU operation for forward computation.

    Args:
        operation (dict): Dictionary specifying the operation to be performed.
        feat_sizes (List): List of feature sizes of feature maps to be expected during forward computation.

    Returns:
        sub_operations (List): List of sub-operation dictionaries specifying their behavior during forward computation.
        feat_sizes (List): Updated list of feature sizes of feature maps to be generated during forward computation.
    """

    # Get sub-operation keyword arguments and initialize empty list of sub-operations
    allowed_keys = ['inplace']
    sub_operation_kwargs = {k: v for k, v in operation.items() if k in allowed_keys}
    sub_operations = []

    # Initialize every ReLU operation
    for map_id in operation['in']:
        sub_operation = {'function': F.relu, 'in_map_ids': [map_id], **sub_operation_kwargs}
        feat_size = feat_sizes[map_id]

        sub_operations.append(sub_operation)
        feat_sizes.append(feat_size)

    return sub_operations, feat_sizes


def initialize_operation(operation, feat_sizes, modules_offset):
    """
    Initializes operation with corresponding modules for forward computation.

    Args:
        operation (dict): Dictionary specifying the operation to be performed.
        feat_sizes (List): List of feature sizes of feature maps to be expected during forward computation.
        modules_offset (int): Integer containing the offset to be added to the module indices.

    Returns:
        modules (List): List of initialized modules used by this operation.
        sub_operations (List): List of sub-operation dictionaries specifying their behavior during forward computation.
        feat_sizes (List): Updated list of feature sizes of feature maps to be generated during forward computation.

    Raises:
        ValueError: Error when sizes of the 'in' and 'out' keys from the operation dictionary do not match.
        ValueError: Error when unknown operation type was provided.
    """

    # Transform 'in' and 'out' keys to lists if integers
    if isinstance(operation['in'], int):
        operation['in'] = [operation['in']]

    if isinstance(operation['out'], int):
        operation['out'] = [operation['out']]

    # Check operation dictionary
    if len(operation['in']) != len(operation['out']):
        raise ValueError(f"Sizes of 'in' and 'out' keys from operation dictionary {operation} must match.")

    # Get operation type and initialize empty list of modules
    operation_type = operation['type']
    modules = []

    # Initialize operation
    if operation_type == 'add':
        sub_operations, feat_sizes = initialize_add(operation, feat_sizes)
    elif operation_type == 'conv2d':
        modules, sub_operations, feat_sizes = initialize_conv2d(operation, feat_sizes, modules_offset)
    elif operation_type == 'interpolate':
        sub_operations, feat_sizes = initialize_interpolate(operation, feat_sizes)
    elif operation_type == 'relu':
        sub_operations, feat_sizes = initialize_relu(operation, feat_sizes)
    else:
        raise ValueError(f'Unknown operation type {operation_type} was provided.')

    return modules, sub_operations, feat_sizes
