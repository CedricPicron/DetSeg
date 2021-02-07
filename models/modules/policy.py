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
        input_type (str): String containing the input type chosen from {'pyramid'}.
        inference_samples (int): Maximum number of samples during inference.
        head (nn.Sequential): Module computing the action logits from its input.
    """

    def __init__(self, feat_size, input_type, num_hidden_layers=1, inference_samples=100, **kwargs):
        """
        Initializes the PolicyNet module.

        Args:
            feat_size (int): Integer containing the feature size.
            input_type (str): String containing the input type chosen from {'pyramid'}.
            num_hidden_layers (int): Number of hidden layers of the policy head (default=1).
            inference_samples (int): Maximum number of samples during inference (default=100).

            kwargs (Dict): Dictionary of keyword arguments, potentially containing following key:
                - num_groups (int): number of groups used for group normalization.

        Raises:
            ValueError: Raised when unknown input type is provided.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set input type and number of samples attributes
        self.input_type = input_type
        self.inference_samples = inference_samples

        # Initialize network based on input type
        if input_type == 'pyramid':

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
            If input_type is 'pyramid':
                input (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

        Returns
            If module in training mode and input_type is 'pyramid':
                sample_masks (BoolTensor): Masks of features to sampled of shape [batch_size, fH*fW].
                action_losses (List): List of size [batch_size] with initial action losses of shape [train_samples].

            If module in inference mode:
                sample_ids (LongTensor): Indices of features to be sampled of shape [batch_size, inference_samples].
        """

        # Process input based on input type
        if self.input_type == 'pyramid':

            # Get action logits
            logit_maps = [self.head(feat_map) for feat_map in input]
            logits = torch.cat([logit_map.flatten(1) for logit_map in logit_maps], dim=1)

            # Get sample masks and action losses during training
            if self.training:

                # Get sample masks
                with torch.no_grad():
                    action_probs = F.sigmoid(logits)
                    sample_masks = torch.bernoulli(action_probs).to(torch.bool)

                # Get initial pre-reward action losses
                batch_size = len(logits)
                sampled_logits = [logits[i][sample_masks[i]] for i in range(batch_size)]
                action_losses = [F.logsigmoid(sampled_logits[i]) for i in range(batch_size)]

                return sample_masks, action_losses

            # Get indices of most promising features during inference
            else:

                # Get indices of most promising features
                sample_ids = torch.argsort(logits, dim=1, descending=True)[:, :self.inference_samples]

                return sample_ids
