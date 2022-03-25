"""
Collection of classification heads.
"""

from torch import nn
import torch.nn.functional as F

from models.build import build_model, MODELS


@MODELS.register_module()
class BaseClsHead(nn.Module):
    """
    Class implementing the BaseClsHead module.

    Attributes:
        logits (nn.Module): Module computing the classification logits.
        loss (nn.Module): Module computing the classification loss.
        matcher (nn.Module): Optional module determining the target classification labels.
    """

    def __init__(self, logits_cfg, loss_cfg, matcher_cfg=None):
        """
        Initializes the BaseClsHead module.

        Args:
            logits_cfg (Dict): Configuration dictionary specifying the logits module.
            loss_cfg (Dict): Configuration dictionary specifying the loss module.
            matcher_cfg (Dict): Configuration dictionary specifying the matcher module (default=None).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build logits module
        self.logits = build_model(logits_cfg)

        # Build loss module
        self.loss = build_model(loss_cfg)

        # Build matcher module if needed
        if matcher_cfg is not None:
            self.matcher = build_model(matcher_cfg)

    def forward_pred(self, in_feats, storage_dict, **kwargs):
        """
        Prediction forward method of the BaseClsHead module.

        Args:
            in_feats (FloatTensor): Input features of shape [num_feats, in_feat_size].
            storage_dict (Dict): Dictionary storing all kinds of key-value pairs of interest.
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional key:
                - cls_logits (FloatTensor): classification logits of shape [num_feats, num_labels].
        """

        # Get classification logits
        cls_logits = self.logits(in_feats)
        storage_dict['cls_logits'] = cls_logits

        return storage_dict

    def forward_loss(self, storage_dict, loss_dict, analysis_dict=None, id=None, **kwargs):
        """
        Forward method of the BaseClsHead module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys (after matching):
                - cls_logits (FloatTensor): classification logits of shape [num_feats, num_labels];
                - cls_targets (LongTensor): target classification labels of shape [num_feats].

            loss_dict (Dict): Dictionary containing different weighted loss terms.
            analysis_dict (Dict): Dictionary containing different analyses (default=None).
            id (int): Integer containing the head id (default=None).
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            loss_dict (Dict): Loss dictionary containing following additional key:
                - cls_loss (FloatTensor): tensor containing the weighted classification loss of shape [1].

            analysis_dict (Dict): Analysis dictionary containing following additional key (if not None):
                - cls_acc (FloatTensor): tensor containing the classification accuracy of shape [1].
        """

        # Perform matching if matcher is available
        if hasattr(self, 'matcher'):
            self.matcher(storage_dict=storage_dict, analysis_dict=analysis_dict)

        # Retrieve classification logits and targets
        cls_logits = storage_dict['cls_logits']
        cls_targets = storage_dict['cls_targets']

        # Get classification loss
        num_labels = cls_logits.size(dim=1)
        cls_targets_oh = F.one_hot(cls_targets, num_classes=num_labels)
        cls_loss = self.loss(cls_logits, cls_targets_oh)

        key_name = f'cls_loss_{id}' if id is not None else 'cls_loss'
        loss_dict[key_name] = cls_loss

        # Get classification accuracy if needed
        if analysis_dict is not None:
            cls_preds = cls_logits.argmax(dim=1)
            cls_acc = (cls_preds == cls_targets).sum(dim=0) / len(cls_preds)

            key_name = f'cls_acc_{id}' if id is not None else 'cls_acc'
            analysis_dict[key_name] = 100 * cls_acc

        return loss_dict, analysis_dict

    def forward(self, mode, **kwargs):
        """
        Forward method of the BaseClsHead module.

        Args:
            mode (str): String containing the forward mode chosen from ['pred', 'loss'].
            kwargs (Dict): Dictionary of keyword arguments passed to the underlying forward method.

        Raises:
            ValueError: Error when an invalid forward mode is provided.
        """

        # Choose underlying forward method
        if mode == 'pred':
            self.forward_pred(**kwargs)

        elif mode == 'loss':
            self.forward_loss(**kwargs)

        else:
            error_msg = f"Invalid forward mode (got '{mode}')."
            raise ValueError(error_msg)
