"""
Function constructing network module from network dictionary.
"""
from copy import deepcopy

from models.modules.attention import DeformableAttn, ParticleAttn, SelfAttn1d
from models.modules.container import Sequential
from models.modules.mlp import OneStepMLP, TwoStepMLP


def get_net(net_dict):
    """
    Get network from network dictionary.

    Args:
        net_dict (Dict): Network dictionary, possibly containing following keys:
            - type (str): string containing the type of network;
            - layers (int): integer containing the number of network layers;
            - in_size (int): input feature size of the network;
            - sample_size (int): sample feature size of the network;
            - hidden_size (int): hidden feature size of the network;
            - out_size (int): output feature size of the network;
            - norm (str): string containing the type of normalization of the network;
            - act_fn (str): string containing the type of activation function of the network;
            - skip (bool): boolean indicating whether layers of the network contain skip connections;
            - version (int): integer containing the version of the network;
            - num_heads (int): integer containing the number of attention heads of the network;
            - num_levels (int): integer containing the number of map levels for the network to sample from;
            - num_points (int): integer containing the number of points of the network;
            - rad_pts (int): integer containing the number of radial points of the network;
            - ang_pts (int): integer containing the number of angular points of the network;
            - lvl_pts (int): integer containing the number of level points of the network;
            - dup_pts (int): integer containing the number of duplicate points of the network;
            - qk_size (int): query and key feature size of the network;
            - val_size (int): value feature size of the network;
            - val_with_pos (bool): boolean indicating whether position info is added to value features;
            - norm_z (float): factor normalizing the sample offsets in the Z-direction;
            - step_size (float): size of the sample steps relative to the sample step normalization;
            - step_norm_xy (str): string containing the normalization type of sample steps in the XY-direction;
            - step_norm_z (float): value normalizing the sample steps in the Z-direction;
            - num_particles (int): integer containing the number of particles per head;
            - sample_insert (bool): boolean indicating whether to insert sample information in a maps structure;
            - insert_size (int): integer containing the size of features to be inserted during sample insertion.

    Returns:
        net (Sequential): Module implementing the network specified by the given network dictionary.

    Raises:
        ValueError: Error when unsupported type of network is provided.
        ValueError: Error when the number of layers in non-positive.
    """

    if net_dict['type'] == 'deformable_attn':
        net_args = (net_dict['in_size'], net_dict['sample_size'], net_dict['out_size'])
        net_keys = ('norm', 'act_fn', 'skip', 'version', 'num_heads', 'num_levels', 'num_points', 'rad_pts', 'ang_pts')
        net_keys = (*net_keys, 'lvl_pts', 'dup_pts', 'qk_size', 'val_size', 'val_with_pos', 'norm_z', 'sample_insert')
        net_keys = (*net_keys, 'insert_size')
        net_kwargs = {k: v for k, v in net_dict.items() if k in net_keys}
        net_layer = DeformableAttn(*net_args, **net_kwargs)

    elif net_dict['type'] == 'one_step_mlp':
        net_args = (net_dict['in_size'], net_dict['out_size'])
        net_kwargs = {k: v for k, v in net_dict.items() if k in ('norm', 'act_fn', 'skip')}
        net_layer = OneStepMLP(*net_args, **net_kwargs)

    elif net_dict['type'] == 'particle_attn':
        net_args = (net_dict['in_size'], net_dict['sample_size'], net_dict['out_size'])
        net_keys = ('norm', 'act_fn', 'skip', 'version', 'num_heads', 'num_levels', 'num_points', 'qk_size')
        net_keys = (*net_keys, 'val_size', 'val_with_pos', 'step_size', 'step_norm_xy', 'step_norm_z')
        net_keys = (*net_keys, 'num_particles', 'sample_insert', 'insert_size')
        net_kwargs = {k: v for k, v in net_dict.items() if k in net_keys}
        net_layer = ParticleAttn(*net_args, **net_kwargs)

    elif net_dict['type'] == 'self_attn_1d':
        net_args = (net_dict['in_size'], net_dict['out_size'])
        net_kwargs = {k: v for k, v in net_dict.items() if k in ('norm', 'act_fn', 'skip', 'num_heads')}
        net_layer = SelfAttn1d(*net_args, **net_kwargs)

    elif net_dict['type'] == 'two_step_mlp':
        net_args = (net_dict['in_size'], net_dict['hidden_size'], net_dict['out_size'])
        net_kwargs = {'norm1': net_dict['norm'], 'act_fn2': net_dict['act_fn'], 'skip': net_dict['skip']}
        net_layer = TwoStepMLP(*net_args, **net_kwargs)

    else:
        error_msg = f"The provided network type '{net_dict['type']}' is not supported."
        raise ValueError(error_msg)

    if net_dict['layers'] > 0:
        net = Sequential(*[deepcopy(net_layer) for _ in range(net_dict['layers'])])
    else:
        error_msg = f"The number of network layers must be positive, but got {net_dict['layers']}."
        raise ValueError(error_msg)

    return net
