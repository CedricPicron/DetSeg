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

            policy_dict (Dict): Policy dictionary, potentially containing following keys:
                - num_hidden_layers (int): number of hidden layers of the policy head;
                - inference_samples (int): maximum number of samples during inference;
                - num_groups (int): number of groups used for group normalization.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Initialize policy network
        self.policy = PolicyNet(feat_size, input_type='pyramid', **policy_dict)

    def forward(self, feat_maps, **kwargs):
        """
        Forward method of the BRD module.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].
        """

        if self.training:
            sample_masks, action_losses = self.policy(feat_maps)

            return sample_masks, action_losses

        else:
            sample_ids = self.policy(feat_maps)

            return sample_ids
