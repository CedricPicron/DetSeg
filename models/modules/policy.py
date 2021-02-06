"""
Collection of policy-based modules.
"""
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    """
    Class implementing the PolicyNet module.

    Attributes:
        input_type (str): String containing the input type chosen from {'map'}.
        head (nn.Sequential): Module computing the action logits from its input.
    """

    def __init__(self, feat_size, input_type, num_hidden_layers=1, **kwargs):
        """
        Initializes the PolicyNet module.

        Args:
            feat_size (int): Integer containing the feature size.
            input_type (str): String containing the input type chosen from {'map'}.
            num_hidden_layers (int): Number of hidden layers (default=1).

            kwargs (Dict): Dictionary containing additional keywords arguments:
                - num_groups (int): number of groups used for group normalization.

        Raises:
            ValueError: Raised when unknown input type is provided.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set input type attribute
        self.input_type = input_type

        # Initialize network based on input type
        if input_type == 'map':

            # Get normalization and activation modules
            num_groups = kwargs.setdefault('num_groups', 8)
            norm = nn.GroupNorm(num_groups, feat_size)
            activation = nn.Relu(inplace=True)

            # Get hidden block module
            hidden_weight = nn.Conv1d(feat_size, feat_size, kernel_size=1)
            hidden_block = nn.Sequential(norm, activation, hidden_weight)

            # Get final block module
            final_weight = nn.Conv1d(feat_size, 1, kernel_size=1)
            final_block = nn.Sequential(norm, activation, final_weight)

            # Get policy head module
            hidden_blocks = [deepcopy(hidden_block) for _ in range(num_hidden_layers)]
            self.head = nn.Sequential(*hidden_blocks, final_block)

        else:
            raise ValueError(f"Unknown input type '{input_type}' was provided.")

    def forward(self, input):
        """
        Forward method of the PolicyNet module.

        Args:
            If input_type is 'map':
                input (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

        Returns
            If input_type is 'map':
                action_masks (List): List of size [num_maps] with masks of shape [batch_size, feat_size, fH, fW].
                action_losses (FloatTensor): Initial pre-reward action losses of shape [num_selected_features].
        """

        # Process input based on input type
        if self.input_type == 'map':

            # Get action logit maps
            logit_maps = [self.head(feat_map) for feat_map in input]

            # Get action masks
            with torch.no_grad():
                prob_maps = [F.sigmoid(logit_map) for logit_map in logit_maps]
                action_masks = [torch.bernoulli(prob_map).to(torch.bool) for prob_map in prob_maps]

            # Get initial pre-reward action losses
            sampled_logits = torch.cat([map[mask] for map, mask in zip(logit_maps, action_masks)])
            action_losses = F.logsigmoid(sampled_logits)

        return action_masks, action_losses
