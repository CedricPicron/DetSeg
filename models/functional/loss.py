"""
Collection of loss functions.
"""


def dice_loss(pred_maps, tgt_maps, reduction='sum'):
    """
    Computes the DICE loss between the prediction and corresponding target maps.

    Args:
        pred_maps (FloatTensor): Prediction maps with logits of shape [num_maps, mH, mW].
        tgt_maps (FloatTensor): Target maps with binary labels of shape [num_maps, mH, mW].
        reduction (str): String specifying the reduction operation applied on losses tensor (default='sum').

    Returns:
        * If reduction is 'none':
            losses (FloatTensor): Tensor with per-map DICE losses of shape [num_maps].

        * If reduction is 'mean':
            loss (FloatTensor): Mean of tensor with per-map DICE losses of shape [1].

        * If reduction is 'sum':
            loss (FloatTensor): Sum of tensor with per-map DICE losses of shape [1].

    Raises:
        ValueError: Error when invalid reduction string is provided.
    """

    # Flatten prediction and target maps
    pred_maps = pred_maps.flatten(start_dim=1)
    tgt_maps = tgt_maps.flatten(start_dim=1)

    # Get maps with prediction probabilities
    pred_maps = pred_maps.sigmoid()

    # Compute per-map DICE losses
    numerator = 2 * (pred_maps * tgt_maps).sum(dim=1)
    denominator = pred_maps.sum(dim=1) + tgt_maps.sum(dim=1)
    losses = 1 - (numerator + 1) / (denominator + 1)

    # Apply reduction operation to tensor with per-map DICE losses and return
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        loss = losses.mean(dim=0)
        return loss
    elif reduction == 'sum':
        loss = losses.sum(dim=0)
        return loss
    else:
        error_msg = f"Invalid reduction string '{reduction}'."
        raise ValueError(error_msg)
