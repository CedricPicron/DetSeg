"""
Collection of loss-related functions.
"""

import torch


def reduce_losses(losses, loss_balance=None, loss_reduction='mean', preds=None, targets=None, qry_ids=None,
                  tgt_ids=None):
    """
    Method reducing the element-wise losses.

    Args:
        losses (FloatTensor): Element-wise losses of shape [num_loss_elems].
        loss_balance (str): String containing the loss balancing mechanism (default=None).
        loss_reduction (str): String containing the loss reduction mechanism (default='mean').
        preds (FloatTensor): Tensor containing the predictions of shape [num_loss_elems] (default=None).
        targets (FloatTensor): Tensor containing the targets of shape [num_loss_elemes] (default=None).
        qry_ids (LongTensor): Query indices corresponding to losses of shape [num_loss_elems] (default=None).
        tgt_ids (LongTensor): Target indices corresponding to losses of shape [num_loss_elems] (default=None).

    Returns:
        loss (FloatTensor): Reduced loss tensor of shape [].

    Raises:
        ValueError: Error when an invalid loss balancing mechanism is provided.
        ValueError: Error when an invalid loss reduction mechanism is provided.
    """

    # Balance loss
    if loss_balance is None:
        pass

    elif loss_balance in ('hard', 'random'):
        pos_tgt_mask = targets > 0.5
        neg_tgt_mask = ~pos_tgt_mask
        balance_mask = torch.ones_like(pos_tgt_mask)

        num_pos_tgts = pos_tgt_mask.sum().item()
        num_neg_tgts = pos_tgt_mask.numel() - num_pos_tgts

        if num_pos_tgts > num_neg_tgts:
            num_removals = num_pos_tgts - num_neg_tgts
            pos_tgt_ids = pos_tgt_mask.nonzero()[:, 0]

            if loss_balance == 'hard':
                pos_tgt_losses = losses[pos_tgt_mask]
                remove_ids = pos_tgt_losses.topk(num_removals, largest=False)[1]

            elif loss_balance == 'random':
                remove_ids = torch.randperm(num_pos_tgts, device=targets.device)
                remove_ids = remove_ids[:num_removals]

            remove_ids = pos_tgt_ids[remove_ids]
            balance_mask[remove_ids] = False

        elif num_neg_tgts > num_pos_tgts:
            num_removals = num_neg_tgts - num_pos_tgts
            neg_tgt_ids = neg_tgt_mask.nonzero()[:, 0]

            if loss_balance == 'hard':
                neg_tgt_losses = losses[neg_tgt_mask]
                remove_ids = neg_tgt_losses.topk(num_removals, largest=False)[1]

            elif loss_balance == 'random':
                remove_ids = torch.randperm(num_neg_tgts, device=targets.device)
                remove_ids = remove_ids[:num_removals]

            remove_ids = neg_tgt_ids[remove_ids]
            balance_mask[remove_ids] = False

        losses = losses[balance_mask]

    else:
        error_msg = f"Invalid loss balancing mechanism (got '{loss_balance}')."
        raise ValueError(error_msg)

    # Reduce loss
    if loss_reduction == 'mean':
        loss = losses.mean()

    elif loss_reduction == 'qry_sum':
        if loss_balance is not None:
            qry_ids = qry_ids[balance_mask]

        inv_ids, qry_counts = qry_ids.unique(return_inverse=True, return_counts=True)[1:]
        loss_weights = torch.ones_like(qry_ids, dtype=torch.float) / qry_counts[inv_ids]
        loss = (loss_weights * losses).sum()

    elif loss_reduction == 'sum':
        loss = losses.sum()

    elif loss_reduction == 'tgt_sum':
        if loss_balance is not None:
            tgt_ids = tgt_ids[balance_mask]

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


def update_loss_cfg(loss_cfg, loss_reduction=None):
    """
    Function updating the loss configuration dictionary by replacing the loss reduction mechanism with 'none'.

    Args:
        loss_cfg (Dict): Configuration dictionary specifying a loss module with arbitrary loss reduction.
        loss_reduction (str): Loss reduction mechanism of the original loss configuration dictionary (default=None).

    Returns:
        loss_cfg (Dict): Configuration dictionary specifying the updated loss module with loss reduction 'none'.
        loss_reduction (str): Loss reduction mechanism of the original loss configuration dictionary (or None).

    Raises:
        ValueError: Error when the loss configuration dictionary contains inconsistent loss reduction mechanisms.
    """

    # Update loss configuration dictionary
    for key, value in loss_cfg.items():
        found_loss_reduction = False

        if key == 'reduction':
            loss_cfg[key] = 'none'
            new_loss_reduction = value
            found_loss_reduction = True

        elif isinstance(value, dict):
            new_loss_reduction = update_loss_cfg(value)[1]
            found_loss_reduction = True

        if found_loss_reduction:
            if loss_reduction is None:
                loss_reduction = new_loss_reduction

            elif loss_reduction != new_loss_reduction:
                error_msg = f"Inconsistent loss reductions (got '{loss_reduction}' and '{new_loss_reduction}')."
                raise ValueError(error_msg)

    return loss_cfg, loss_reduction
