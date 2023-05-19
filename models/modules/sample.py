"""
Collection of sampling modules.
"""

from mmcv.ops import point_sample
from mmdet.models.utils.point_sample import get_uncertain_point_coords_with_randomness
from torch import nn

from models.build import MODELS


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
