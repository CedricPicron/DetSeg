"""
Segmentation head modules and build function.
"""

import torch
from torch import nn
import torch.nn.functional as F


class BinarySegHead(nn.Module):
    """
    Class implementing the BinarySegHead module, segmenting objects from background.

    Attributes:
        projs (ModuleList): List of size [num_maps] with linear projection modules.
        disputed_loss (bool): Bool indicating if loss should be applied at disputed ground-truth positions.
        disputed_beta (float): Threshold value at which disputed smooth L1 loss changes from L1 to L2 loss.
        map_size_correction (bool): Bool indicating whether to scale losses relative to their map sizes.
        loss_weight (float): Weight factor used to scale the binary segmentation loss.
    """

    def __init__(self, feat_sizes, disputed_loss, disputed_beta, map_size_correction, loss_weight):
        """
        Initializes the BinarySegHead module.

        Args:
            feat_sizes (List): List of size [num_maps] containing the feature size of each map.
            disputed_loss (bool): Bool indicating if loss should be applied at disputed ground-truth positions.
            disputed_beta (float): Threshold value at which disputed smooth L1 loss changes from L1 to L2 loss.
            map_size_correction (bool): Bool indicating whether to scale losses relative to their map sizes.
            loss_weight (float): Weight factor used to scale the binary segmentation loss.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Initialization of linear projection modules
        self.projs = nn.ModuleList([nn.Linear(f, 1) for f in feat_sizes])

        # Set remaining attributes as specified by the input arguments
        self.disputed_loss = disputed_loss
        self.disputed_beta = disputed_beta
        self.map_size_correction = map_size_correction
        self.loss_weight = loss_weight

    @staticmethod
    def required_mask_types():
        """
        Method returning the required mask types of the BinarySegHead module.

        Returns:
            List of strings containing the names of the module's required mask types.
        """

        return ['binary_masks']

    def forward(self, feat_maps, tgt_dict=None):
        """
        Forward method of the BinarySegHead module.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, fH, fW, feat_size].
            tgt_dict (Dict): Optional target dictionary during training and validation with at least following key:
                - binary_masks (List): binary (object + background) segmentation masks of shape [batch_size, fH, fW].

        Returns:
            * If tgt_dict is not None (i.e. during training and validation):
                loss_dict (Dictionary): Loss dictionary containing following key:
                    - binary_seg_loss (FloatTensor): weighted binary segmentation loss of shape [1].

                analysis_dict (Dictionary): Analysis dictionary containing following key:
                    - binary_seg_accuracy (FloatTensor): accuracy of the binary segmentation of shape [1].

            * If tgt_dict is None (i.e. during testing and possibly during validation):
                pred_dict (Dict): Prediction dictionary containing following key:
                    - binary_maps (List): predicted binary segmentation maps of shape [batch_size, fH, fW].
        """

        # Compute logit maps
        logit_maps = [proj(feat_map).squeeze(-1) for feat_map, proj in zip(feat_maps, self.projs)]

        # Compute and return dictionary with predicted binary segmentation maps (validation/testing only)
        if tgt_dict is None:
            pred_dict = {'binary_maps': [F.sigmoid(logit_map) for logit_map in logit_maps]}
            return pred_dict

        # Flatten logit and target maps and initialize losses tensor (trainval only)
        logits = torch.cat([logit_map.flatten() for logit_map in logit_maps])
        targets = torch.cat([tgt_mask.flatten() for tgt_mask in tgt_dict['binary_masks']])
        losses = torch.zeros_like(logits)

        # Compute losses at background ground-truth positions (trainval only)
        background_mask = targets == 0
        losses[background_mask] = torch.log(1 + torch.exp(logits[background_mask]))

        # Compute losses at object ground-truth positions (trainval only)
        object_mask = targets == 1
        losses[object_mask] = torch.log(1 + torch.exp(-logits[object_mask]))

        # Compute losses at disputed ground-truth positions if desired (trainval only)
        if self.disputed_loss:
            disputed = torch.bitwise_and(targets > 0 and targets < 1)
            losses[disputed] = F.smooth_l1_loss(logits[disputed]/4, targets[disputed]-0.5, beta=self.disputed_beta)

        # Apply map size corrections to losses if desired (trainval only)
        if self.map_size_correction:
            map_sizes = [logit_map.shape[1] * logit_map.shape[2] for logit_map in logit_maps]
            scales = [map_sizes[-1]/map_size for map_size in map_sizes]
            indices = torch.cumsum(torch.tensor([0, *map_sizes], device=logits.device), dim=0)
            losses = torch.cat([scale*losses[i0:i1] for scale, i0, i1 in zip(scales, indices[:-1], indices[1:])])

        # Get loss dictionary with weighted binary segmentation loss (trainval only)
        batch_size = logit_maps[0].shape[0]
        avg_loss = torch.sum(losses) / batch_size
        loss_dict = {'binary_seg_loss': self.loss_weight * avg_loss}

        # Perform analyses and place them in analysis dictionary (trainval only)
        with torch.no_grad():
            analysis_dict = {}

        return loss_dict, analysis_dict


def build_seg_heads(args):
    """
    Build segmentation head modules from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        seg_heads (List): List of specified segmentation head modules.

    Raises:
        ValueError: Error when unknown segmentation head type was provided.
    """

    # Get feature sizes and number of heads list
    map_ids = range(args.min_resolution_id, args.max_resolution_id+1)
    feat_sizes = [min((args.base_feat_size * 2**i, args.max_feat_size)) for i in map_ids]

    # Initialize empty list of segmentation head modules
    seg_heads = []

    # Build desired segmentation head modules
    for seg_head_type in args.seg_heads:
        if seg_head_type == 'binary_seg':
            head_args = [args.disputed_loss, args.disputed_beta, not args.no_map_size_correction, args.bin_seg_weight]
            binary_seg_head = BinarySegHead(feat_sizes, *head_args)
            seg_heads.append(binary_seg_head)

        else:
            raise ValueError(f"Unknown segmentation head type '{seg_head_type}' was provided.")

    return seg_heads
