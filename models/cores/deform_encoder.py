"""
Deformable encoder core.
"""

from torch import nn

from models.build import MODELS


@MODELS.register_module()
class DeformEncoder(nn.Module):
    """
    Class implementing the DeformEncoder module.
    """

    def __init__(self):
        """
        Initializes the DeformEncoder module.
        """

    def forward(self, in_feat_maps):
        """
        Forward method the DeformEncoder module.

        Args:
            in_feat_maps (List): Input feature maps [num_maps] of shape [batch_size, feat_size, fH, fW].

        Returns:
            out_feat_maps (List): Output feature maps [num_maps] of shape [batch_size, feat_size, fH, fW].
        """
