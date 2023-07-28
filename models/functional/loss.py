"""
Collection of loss-related functions.
"""

import torch


def reduce_losses(losses, loss_reduction='mean', qry_ids=None, tgt_ids=None):
    """
    Method reducing the element-wise losses.

    Args:
        losses (FloatTensor): Element-wise losses of shape [num_loss_elems].
        loss_reduction (str): String containing the loss reduction mechanism (default='mean').
        qry_ids (LongTensor): Query indices corresponding to losses of shape [num_loss_elems, *] (default=None).
        tgt_ids (LongTensor): Target indices corresponding to losses of shape [num_loss_elems, *] (default=None).

    Returns:
        loss (FloatTensor): Reduced loss tensor of shape [].

    Raises:
        ValueError: Error when an invalid loss reduction mechanism is provided.
    """

    # Reduce loss
    if loss_reduction == 'mean':
        loss = losses.mean()

    elif loss_reduction == 'qry_sum':
        if (qry_ids.dim() > 1) and (losses.dim() == 1):
            qry_ids = qry_ids.flatten(1)
            assert (qry_ids.diff(dim=1) == 0).all().item()
            qry_ids = qry_ids[:, 0]

        inv_ids, qry_counts = qry_ids.unique(return_inverse=True, return_counts=True)[1:]
        loss_weights = torch.ones_like(qry_ids, dtype=torch.float) / qry_counts[inv_ids]
        loss = (loss_weights * losses).sum()

    elif loss_reduction == 'sum':
        loss = losses.sum()

    elif loss_reduction == 'tgt_sum':
        if (tgt_ids.dim() > 1) and (losses.dim() == 1):
            tgt_ids = tgt_ids.flatten(1)
            assert (tgt_ids.diff(dim=1) == 0).all().item()
            tgt_ids = tgt_ids[:, 0]

        inv_ids, tgt_counts = tgt_ids.unique(return_inverse=True, return_counts=True)[1:]
        loss_weights = torch.ones_like(tgt_ids, dtype=torch.float) / tgt_counts[inv_ids]
        loss = (loss_weights * losses).sum()

    else:
        error_msg = f"Invalid loss reduction mechanism (got '{loss_reduction}')."
        raise ValueError(error_msg)

    return loss


def sigmoid_dice_loss(pred_logits, tgt_labels, reduction='mean'):
    """
    Computes the DICE loss between the prediction logits and the corresponding binary target labels.

    The sigmoid function is used to compute the prediction probabilities from the prediction logits.

    Args:
        pred_logits (FloatTensor): Prediction logits of shape [num_groups, group_size].
        tgt_labels (FloatTensor): Binary target labels of shape [num_groups, group_size].
        reduction (str): String specifying the reduction operation applied on group-wise losses (default='mean').

    Returns:
        * If reduction is 'none':
            losses (FloatTensor): Tensor with group-wise DICE losses of shape [num_groups].

        * If reduction is 'mean':
            loss (FloatTensor): Mean of tensor with group-wise DICE losses of shape [].

        * If reduction is 'sum':
            loss (FloatTensor): Sum of tensor with group-wise DICE losses of shape [].

    Raises:
        ValueError: Error when an invalid reduction string is provided.
    """

    # Get prediction probabilities
    pred_probs = pred_logits.sigmoid()

    # Get group-wise DICE losses
    numerator = 2 * (pred_probs * tgt_labels).sum(dim=1)
    denominator = pred_probs.sum(dim=1) + tgt_labels.sum(dim=1)
    losses = 1 - (numerator + 1) / (denominator + 1)

    # Apply reduction operation on group-wise losses and return
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        loss = losses.mean()
        return loss
    elif reduction == 'sum':
        loss = losses.sum()
        return loss
    else:
        error_msg = f"Invalid reduction string (got '{reduction}')."
        raise ValueError(error_msg)


def update_loss_module(loss_module, loss_reduction=None):
    """
    Function updating the loss module by replacing the reduction attributes with 'none'.

    Args:
        loss_module (nn.Module): Loss module with one or multiple reduction attributes to be replaced with 'none'.
        loss_reduction (str): Loss reduction mechanism of the original loss module (default=None).

    Returns:
        loss_module (nn.Module): Updated loss module with the reduction attributes replaced by 'none'.
        loss_reduction (str): Loss reduction mechanism of the original loss module (or None).

    Raises:
        ValueError: Error when the loss module contains inconsistent loss reduction mechanisms.
    """

    # Update loss configuration dictionary
    if 'reduction' in vars(loss_module):
        found_loss_reduction = loss_module.reduction
        loss_module.reduction = 'none'

        if loss_reduction is None:
            loss_reduction = found_loss_reduction

        elif loss_reduction != found_loss_reduction:
            error_msg = f"Inconsistent loss reductions (got '{loss_reduction}' and '{found_loss_reduction}')."
            raise ValueError(error_msg)

    for module in loss_module.children():
        module, loss_reduction = update_loss_module(module, loss_reduction)

    return loss_module, loss_reduction
