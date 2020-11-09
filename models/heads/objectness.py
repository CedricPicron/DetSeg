"""
Objectness head modules and build function.
"""

import torch
from torch import nn
import torch.nn.functional as F


class ObjectnessHead(nn.Module):
    """
    Class implementing the ObjectnessHead module.

    Attributes:
        projs (ModuleList): List of size [num_maps] with linear projection modules.
        loss_weight (float): Weight factor used to scale the objectness loss.
        with_disputed_loss (bool): Bool indicating if loss should be applied at disputed ground-truth positions.
        feat_map_size_correction (bool): Bool indicating whether to scale losses with relative feature map sizes.
        beta (float): Threshold at which smooth L1 loss changes from L1 to L2 loss during disputed loss computation.
    """

    def __init__(self, feat_sizes, loss_weight, with_disputed_loss, feat_map_size_correction, beta=None):
        """
        Initializes the ObjectnessHead module.

        Args:
            feat_sizes (List): List of size [num_maps] containing the feature size in each map.
            loss_weight (float): Weight factor used to scale the objectness loss.
            with_disputed_loss (bool): Bool indicating if loss should be applied at disputed ground-truth positions.
            feat_map_size_correction (bool): Bool indicating whether to scale losses with relative feature map sizes.
            beta (float): Threshold at which smooth L1 loss changes from L1 to L2 loss during disputed loss computation.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Initialization of linear projection modules
        self.projs = nn.ModuleList([nn.Linear(f, 1) for f in feat_sizes])

        # Set remaining attributes as specified by input arguments
        self.loss_weight = loss_weight
        self.with_disputed_loss = with_disputed_loss
        self.feat_map_size_correction = feat_map_size_correction
        self.beta = beta

    def forward(self, feat_maps, tgt_maps=None):
        """
        Forward method of the ObjectnessHead module.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, H, W, feat_size].
            tgt_maps (List): List of size [num_maps] with ground-truth objectness maps of shape [batch_size, H, W].

        Returns:
            - Training only:
                loss (FloatTensor): Tensor of shape [1] containing the weighted objectness loss.

            - Inference only:
                pred_maps (List): List of size [num_maps] with predicted objectness maps of shape [batch_size, H, W].
        """

        # Compute logit maps
        logit_maps = [proj(feat_map).squeeze(-1) for feat_map, proj in zip(feat_maps, self.projs)]

        # Compute and return predicted objectness probability maps (inference only)
        if not self.train:
            pred_maps = [F.sigmoid(logit_map) for logit_map in logit_maps]
            return pred_maps

        # Flatten logit and target maps and initialize losses tensor (training only)
        logits = torch.cat([logit_map.flatten() for logit_map in logit_maps])
        targets = torch.cat([tgt_map.flatten() for tgt_map in tgt_maps])
        losses = torch.zeros_like(logits)

        # Compute losses at no-object ground-truth positions (training only)
        no_obj_mask = targets == 0
        losses[no_obj_mask] = torch.log(1 + torch.exp(logits[no_obj_mask]))

        # Compute losses at object ground-truth positions (trainin only)
        obj_mask = targets == 1
        losses[obj_mask] = torch.log(1 + torch.exp(-logits[obj_mask]))

        # Compute losses at disputed ground-truth positions if desired (training only)
        if self.with_disputed_loss:
            disputed = torch.bitwise_and(targets > 0 and targets < 1)
            losses[disputed] = F.smooth_l1_loss(logits[disputed]/4, targets[disputed]-0.5, beta=self.beta)

        # Apply feature map size corrections to losses if desired (training only)
        if self.feat_map_size_correction:
            map_sizes = [0, *[logit_map.shape[1] * logit_map.shape[2] for logit_map in logit_maps]]
            indices = torch.cumsum(torch.tensor(map_sizes, device=logits.device), dim=0)
            scales = [4**i for i in range(-len(logit_maps)+1, 1)]
            losses = torch.cat([scale*losses[i0:i1] for i0, i1, scale in zip(indices[:-1], indices[1:], scales)])

        # Average and scale loss (training only)
        batch_size = logit_maps[0].shape[0]
        loss = torch.sum(losses) / batch_size
        loss = self.loss_weight * loss

        return loss


def build_obj_head(args):
    """
    Build objectness head module from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        obj_head (nn.Module): The specified objectness head module.

    Raises:
        ValueError: Error when unknown objectness head type was provided.
    """

    # Check command-line arguments
    check = args.max_resolution_id > args.min_resolution_id
    msg = "'--max_resolution_id' should be larger than '--min_resolution_id'"
    assert check, msg

    # Get feature sizes and number of heads list
    map_ids = range(args.min_resolution_id, args.max_resolution_id+1)
    feat_sizes = [min((args.base_feat_size * 2**i, args.max_feat_size)) for i in map_ids]

    # Build desired objectness head module
    if args.obj_head_type == 'default':
        loss_weight = args.obj_head_weight
        disputed_bool = args.disputed_loss
        correction_bool = not args.no_map_size_correction
        beta = args.obj_head_beta
        obj_head = ObjectnessHead(feat_sizes, loss_weight, disputed_bool, correction_bool, beta)
    else:
        raise ValueError(f"Unknown objectness head type '{args.obj_head_type}' was provided.")

    return obj_head
