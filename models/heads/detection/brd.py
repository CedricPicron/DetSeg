"""
Base Reinforced Decoder (BRD) detection head.
"""

from torch import nn

from models.modules.policy import PolicyNet


class BRD(nn.Module):
    """
    Class implementing the Base Reinforced Decoder (BRD) module.

    Attributes:
        policy (PolicyNet): Policy network computing action masks and initial action losses.
    """

    def __init__(self, feat_size, policy_dict, decoder_dict, ffn_dict):
        """
        Initializes the BRD module.

        Args:
            feat_size (int): Integer containing the feature size.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Initialize policy network
        self.policy = PolicyNet(feat_size, input_type='map', **policy_dict)

    def forward(self, feat_maps, **kwargs):
        """
        Forward method of the BRD module.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].
        """

        # Get action masks and initial action losses from policy network
        action_masks, action_losses = self.policy(feat_maps)

        return action_masks, action_losses
