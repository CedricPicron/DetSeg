"""
Modeling utilities.
"""

import torch
from torch import nn
import torch.nn.functional as F

from models.functional.position import sine_pos_encodings
from structures.boxes import Boxes


def downsample_index_maps(index_maps, map_sizes):
    """
    Method downsampling the given full-resolution index maps to maps with the given map sizes.

    Args:
        index_maps (LongTensor): Padded full-resolution index maps of shape [*, max_iH, max_iW].
        map_sizes (List): List of size [num_core_maps] with tuples of requested map sizes (fH, fW).

    Returns:
        maps_list (List): List of size [num_core_maps] with downsampled index maps of shape [*, fH, fW].
    """

    # Save original size of index maps and convert index maps into desired format
    original_size = index_maps.shape
    maps = index_maps.view(-1, *original_size[-2:])

    # Initialize list of downsampled index maps
    maps_list = [torch.zeros(*original_size[:-2], *map_size).to(index_maps) for map_size in map_sizes]

    # Compute list of downsampled index maps
    for i in range(len(maps_list)):
        while maps.shape[-2:] != map_sizes[i]:
            maps = maps[:, ::2, ::2]

        maps_list[i] = maps.view(*original_size[:-2], *maps.shape[-2:]).contiguous()

    return maps_list


def downsample_masks(masks, map_sizes):
    """
    Method downsampling the given full-resolution masks to maps with the given map sizes.

    Args:
        masks (ByteTensor): Padded full-resolution masks of shape [*, max_iH, max_iW].
        map_sizes (List): List of size [num_core_maps] with tuples of requested map sizes (fH, fW).

    Returns:
        maps_list (List): List of size [num_core_maps] with downsampled FloatTensor maps of shape [*, fH, fW].
    """

    # Save original size of masks and get initial full-resolution maps
    original_size = masks.shape
    maps = masks.float().view(-1, 1, *original_size[-2:])

    # Get averaging convolutional kernel
    device = masks.device
    average_kernel = torch.ones(1, 1, 3, 3, device=device)/9

    # Initialize list of downsampled output maps
    maps_list = [torch.zeros(*original_size[:-2], *map_size, device=device) for map_size in map_sizes]

    # Compute list of downsampled output maps
    for i in range(len(maps_list)):
        while maps.shape[-2:] != map_sizes[i]:
            maps = F.pad(maps, (1, 1, 1, 1), mode='replicate')
            maps = F.conv2d(maps, average_kernel, stride=2, padding=0)

        maps_list[i] = maps.view(*original_size[:-2], *maps.shape[-2:])

    return maps_list


def get_feat_boxes(feat_maps):
    """
    Function computing feature boxes based on map sizes of given feature maps.

    Args:
        feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

    Returns:
        feat_boxes (Boxes): Axis-aligned bounding boxes related to features of given feature maps of size [num_feats].
    """

    # Get feature centers
    _, feat_cts_maps = sine_pos_encodings(feat_maps, normalize=True)
    feat_cts = torch.cat([feat_cts_map.flatten(1).t() for feat_cts_map in feat_cts_maps], dim=0)

    # Get feature widths and heights
    map_numel = torch.tensor([feat_map.flatten(2).shape[-1] for feat_map in feat_maps]).to(feat_cts.device)
    feat_wh = torch.tensor([[1/s for s in feat_map.shape[:1:-1]] for feat_map in feat_maps]).to(feat_cts)
    feat_wh = torch.repeat_interleave(feat_wh, map_numel, dim=0)

    # Concatenate feature centers, widths and heights to get feature boxes
    feat_boxes = torch.cat([feat_cts, feat_wh], dim=1)
    feat_boxes = Boxes(feat_boxes, format='cxcywh', normalized='img_with_padding')

    return feat_boxes


class MLP(nn.Module):
    """
    Module implementing a simple multi-layer perceptron (MLP).

    Attributes:
        layers (nn.ModuleList): List of nn.Linear modules, which will be split by ReLU activation functions.
        num_layers (int): Number of linear layers in the multi-layer perceptron.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        """
        Initializes the MLP module.

        Args:
            input_dim (int): Expected dimension of input features.
            hidden_dim (int): Feature dimension in hidden layers.
            output_dim (int): Dimension of output features.
            num_layers (int): Number of linear layers in the multi-layer perceptron.
        """

        super().__init__()
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.num_layers = num_layers

    def forward(self, x):
        """
        Forward method of the MLP module.
        """

        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers-1 else layer(x)

        return x
