"""
Collection of modules computing accuracy metrics.
"""

import torch
from torch import nn

from models.build import MODELS


@MODELS.register_module()
class BinaryAccuracy(nn.Module):
    """
    Class implementing the BinaryAccuracy module.

    Attributes:
        pred_key (str): String with key to retrieve predictions from storage dictionary.
        tgt_key (str): String with key to retrieve targets from storage dictionary.
        out_key (str): String with key to store output accuracy in storage dictionary.
        as_logits (bool): Boolean indicating whether input predictions are in logits space.
        as_percentage (bool): Boolean indicating whether accuracy should be returned as percentage.
    """

    def __init__(self, pred_key, tgt_key, out_key, as_logits=True, as_percentage=True):
        """
        Initializes the BinaryAccuracy module.

        Args:
            pred_key (str): String with key to retrieve predictions from storage dictionary.
            tgt_key (str): String with key to retrieve targets from storage dictionary.
            out_key (str): String with key to store output accuracy in storage dictionary.
            as_logits (bool): Boolean indicating whether input predictions are in logits space (default=True).
            as_percentage (bool): Boolean indicating whether accuracy should be returned as percentage (default=True).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set additional attributes
        self.pred_key = pred_key
        self.tgt_key = tgt_key
        self.out_key = out_key
        self.as_logits = as_logits
        self.as_percentage = as_percentage

    @torch.no_grad()
    def forward(self, storage_dict, **kwargs):
        """
        Forward method of the BinaryAccuracy module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - {self.pred_key} (FloatTensor): tensor with predictions of shape [*];
                - {self.tgt_key} (FloatTensor): tensor with targets of shape [*].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional key:
                - {self.out_key}: output mask accuracy of shape [].
        """

        # Retrieve desired items from storage dictionary
        preds = storage_dict[self.pred_key]
        targets = storage_dict[self.tgt_key]

        # Get accuracy
        preds = preds > 0 if self.as_logits else preds > 0.5
        targets = targets.bool()
        acc = (preds == targets).sum() / max(preds.numel(), 1)

        if self.as_percentage:
            acc = 100 * acc

        # Store accuracy in storage dictionary
        storage_dict[self.out_key] = acc

        return storage_dict
