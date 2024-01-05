"""
Collection of sampling modules.
"""

from mmcv.ops import point_sample
from mmdet.models.utils.point_sample import get_uncertain_point_coords_with_randomness
import torch
from torch import nn
import torch.nn.functional as F

from models.build import MODELS


@MODELS.register_module()
class BaseSampler2d(nn.Module):
    """
    Class implementing the BaseSampler2d module.

    Attributes:
        in_map_key (str): String with key to retrieve input map from storage dictionary.
        in_pts_key (str): String with key to retrieve input sample points from storage dictionary.
        out_key (str): String with key to store output sampled features in storage dictionary.
        grid_sample_kwargs (Dict): Dictionary with grid_sample keyword arguments.
    """

    def __init__(self, in_map_key, in_pts_key, out_key, grid_sample_kwargs=None):
        """
        Initializes the BaseSampler2d module.

        Args:
            in_map_key (str): String with key to retrieve input map from storage dictionary.
            in_pts_key (str): String with key to retrieve input sample points from storage dictionary.
            out_key (str): String with key to store output sampled features in storage dictionary.
            grid_sample_kwargs (Dict): Dictionary with grid_sample keyword arguments (default=None).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set additional attributes
        self.in_map_key = in_map_key
        self.in_pts_key = in_pts_key
        self.out_key = out_key
        self.grid_sample_kwargs = grid_sample_kwargs if grid_sample_kwargs is not None else {}

    def forward(self, storage_dict, **kwargs):
        """
        Forward method of the BaseSampler2d module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - {self.in_map_key} (FloatTensor): map to sample from of shape [num_groups, feat_size, fH, fW];
                - {self.in_pts_key} (FloatTensor): normalized sample points of shape [num_groups, num_pts, 2].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional key:
                - {self.out_key} (FloatTensor): sampled features of shape [num_groups, num_pts, feat_size].
        """

        # Retrieve desired items from storage dictionary
        sample_map = storage_dict[self.in_map_key]
        sample_pts = storage_dict[self.in_pts_key]

        # Get sampled features
        sample_pts = 2*sample_pts - 1
        sample_pts = sample_pts.unsqueeze(dim=1)

        sample_feats = F.grid_sample(sample_map, sample_pts, **self.grid_sample_kwargs)
        sample_feats = sample_feats.squeeze(dim=2).transpose(1, 2)

        # Store sampled features in storage dictionary
        storage_dict[self.out_key] = sample_feats

        return storage_dict


@MODELS.register_module()
class MapSampler2d(nn.Module):
    """
    Class implementing the MapSampler2d module.

    Attributes:
        in_map_key (str): String with key to retrieve input map from storage dictionary.
        in_pts_key (str): String with key to retrieve input sample points from storage dictionary.
        batch_ids_key (str): String with key to retrieve group batch indices from storage dictionary.
        out_key (str): String with key to store output sampled features in storage dictionary.
        sampler (BaseSampler): Module sampling the features at the desired locations.
    """

    def __init__(self, in_map_key, in_pts_key, batch_ids_key, out_key, **kwargs):
        """
        Initializes the MapSampler2d module.

        Args:
            in_map_key (str): String with key to retrieve input map from storage dictionary.
            in_pts_key (str): String with key to retrieve input sample points from storage dictionary.
            batch_ids_key (str): String with key to retrieve group batch indices from storage dictionary.
            out_key (str): String with key to store output sampled features in storage dictionary.
            kwargs (Dict): Dictionary of keyword arguments passed to the BaseSampler2d __init__ method.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build underlying sampler
        self.sampler = BaseSampler2d('in_map', 'in_pts', 'out_feats', **kwargs)

        # Set additional attributes
        self.in_map_key = in_map_key
        self.in_pts_key = in_pts_key
        self.batch_ids_key = batch_ids_key
        self.out_key = out_key

    def forward(self, storage_dict, **kwargs):
        """
        Forward method of the MapSampler2d module.

        Args:
            storage_dict (Dict): Storage dictionary (possibly) containing following keys:
                - {self.in_map_key} (FloatTensor): feature map to sample from of shape [batch_size, feat_size, fH, fW];
                - {self.in_pts_key} (FloatTensor): normalized sample points of shape [num_groups, num_pts, 2];
                - {self.batch_ids_key} (LongTensor): batch indices corresponding to each group of shape [num_groups].

            kwargs (Dict): Dictionary of keyword arguments passed to the underlying sampler module.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional key:
                - {self.out_key} (FloatTensor): sampled features of shape [num_groups, num_pts, feat_size].
        """

        # Retrieve desired items from storage dictionary
        sample_map = storage_dict[self.in_map_key]
        sample_pts = storage_dict[self.in_pts_key]
        batch_ids = storage_dict[self.batch_ids_key]

        # Get tensor shapes and device
        batch_size, feat_size = sample_map.size()[:2]
        num_groups, num_pts = sample_pts.size()[:2]
        device = sample_map.device

        # Get sampled features
        sample_feats = torch.empty(num_groups, num_pts, feat_size, dtype=torch.float, device=device)

        for i in range(batch_size):
            batch_mask = batch_ids == i
            num_groups_i = batch_mask.sum().item()

            if num_groups_i > 0:
                storage_dict['in_map'] = sample_map[i, None]
                storage_dict['in_pts'] = sample_pts[batch_mask].view(1, num_groups_i*num_pts, 2)

                storage_dict = self.sampler(storage_dict, **kwargs)
                out_feats = storage_dict.pop('out_feats')
                sample_feats[batch_mask] = out_feats.view(num_groups_i, num_pts, feat_size)

        storage_dict.pop('in_map', None)
        storage_dict.pop('in_pts', None)

        # Store sampled features in storage dictionary
        storage_dict[self.out_key] = sample_feats

        return storage_dict


@MODELS.register_module()
class PointRendSampling(nn.Module):
    """
    Class implementing the PointRendSampling module.

    Attributes:
        num_points (int): Integer containing the number of points to sample.
        oversample_ratio (float): Value containing the oversample ratio.
        importance_ratio (float): Value containing the importance sampling ratio.
    """

    def __init__(self, num_points=12544, oversample_ratio=3.0, importance_ratio=0.75):
        """
        Initializes the PointRendSampling module.

        Args:
            num_points (int): Integer containing the number of points to sample (default=12544).
            oversample_ratio (float): Value containing the oversample ratio (default=3.0).
            importance_ratio (float): Value containing the importance sampling ratio (default=0.75).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attributes
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_ratio = importance_ratio

    def forward(self, in_mask_logits, in_mask_targets, **kwargs):
        """
        Forward method of the PointRendSampling module.

        Args:
            in_mask_logits (FloatTensor): Tensor containing the input mask logits of shape [num_masks, mH, mW].
            in_mask_targets (FloatTensor): Tensor containing the target mask logits of shape [num_masks, mH, mW].
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            out_mask_logits (FloatTensor): Tensor containing the output mask logits of shape [num_masks, num_points].
            out_mask_targets (FloatTensor): Tensor containing the output mask targets of shape [num_masks, num_points].
        """

        # Add channel dimension to inputs
        in_mask_logits = in_mask_logits.unsqueeze(dim=1)
        in_mask_targets = in_mask_targets.unsqueeze(dim=1)

        # Get sample points
        sample_kwargs = {'num_points': self.num_points, 'oversample_ratio': self.oversample_ratio}
        sample_kwargs['importance_sample_ratio'] = self.importance_ratio
        sample_pts = get_uncertain_point_coords_with_randomness(in_mask_logits, labels=None, **sample_kwargs)

        # Get output mask logits and targets
        out_mask_logits = point_sample(in_mask_logits, sample_pts, align_corners=False).squeeze(dim=1)
        out_mask_targets = point_sample(in_mask_targets, sample_pts, align_corners=False).squeeze(dim=1)

        return out_mask_logits, out_mask_targets
