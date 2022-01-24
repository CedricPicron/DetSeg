"""
Collection of modules related to position encodings.
"""

from torch import nn

from models.build import build_model, MODELS


@MODELS.register_module()
class PosEncoder(nn.Module):
    """
    Class implementing the PosEncoder module.

    Attributes
        net (nn.Module): Module implementing the position encoding network.
        normalize (bool): Boolean indicating whether to normalize the (x, y) position coordinates.
    """

    def __init__(self, net_cfg, normalize=False):
        """
        Initializes the PosEncoder module.

        Args:
            net_cfg (Dict): Configuration dictionary specifying the position encoding network.
            normalize (bool): Boolean indicating whether to normalize the (x, y) position coordinates (default=False).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build position encoding network
        self.net = build_model(net_cfg)

        # Set normalize attribute
        self.normalize = normalize

    def forward(self, pos_xy):
        """
        Forward method of the PosEncoder module.

        Args:
            pos_xy (FloatTensor): Tensor with position coordinates in (x, y) format of shape [num_pts, 2].

        Returns:
            pos_feats (FloatTensor): Tensor with position features of shape [num_pts, feat_size].
        """

        # Normalize position coordinates if requested
        if self.normalize:
            pos_xy = pos_xy / pos_xy.abs().amax(dim=0)

        # Get position features
        pos_feats = self.net(pos_xy)

        return pos_feats
