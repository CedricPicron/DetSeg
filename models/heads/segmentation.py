"""
Segmentation head modules and build function.
"""

from detectron2.utils.visualizer import Visualizer
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
    def required_map_types():
        """
        Method returning the required map types of the BinarySegHead module.

        Returns:
            List of strings containing the names of the module's required map types.
        """

        return ['binary_maps']

    @staticmethod
    def get_accuracy(pred_correctness):
        """
        Method returning the accuracy corresponding to the given BoolTensor 'pred_correctness'.

        Args:
            pred_correctness (BoolTensor): Tensor of shape [num_predictions] containing the correctness of predictions.

        Returns:
            accuracy (FloatTensor): Tensor of shape [] containing the accuracy (between 0 and 1) of the given tensor.
        """

        # Compute accuracy
        if len(pred_correctness) > 0:
            accuracy = pred_correctness.sum() / float(len(pred_correctness))
        else:
            accuracy = torch.tensor(1.0, dtype=pred_correctness.dtype, device=pred_correctness.device)

        return accuracy

    @staticmethod
    def perform_accuracy_analyses(pred_correctness, num_bg_targets):
        """
        Method performing accuracy-related analyses.

        Args:
            pred_correctness (BoolTensor): Tensor of shape [num_predictions] containing the correctness of predictions.
            num_bg_targets (int): Integer containing the number of targets labeled as background.

        Returns:
            analysis_dict (Dict): Dictionary of accuracy-related analyses containing following keys:
                - binary_seg_acc (FloatTensor): accuracy of the binary segmentation of shape [1];
                - binary_seg_acc_bg (FloatTensor): background accuracy of the binary segmentation of shape [1];
                - binary_seg_acc_obj (FloatTensor): object accuracy of the binary segmentation of shape [1].
        """

        # Compute (global) accuracy and place it into analysis dictionary
        accuracy = BinarySegHead.get_accuracy(pred_correctness)
        analysis_dict = {'binary_seg_acc': 100*accuracy}

        # Compute background and object accuracy and place them into analysis dictionary
        bg_accuracy = BinarySegHead.get_accuracy(pred_correctness[:num_bg_targets])
        obj_accuracy = BinarySegHead.get_accuracy(pred_correctness[num_bg_targets:])
        analysis_dict.update({'binary_seg_acc_bg': 100*bg_accuracy, 'binary_seg_acc_obj': 100*obj_accuracy})

        return analysis_dict

    def forward(self, feat_maps, tgt_dict=None, **kwargs):
        """
        Forward method of the BinarySegHead module.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, fH, fW, feat_size].
            tgt_dict (Dict): Optional target dictionary during training and validation with at least following key:
                - binary_maps (List): binary (object + background) segmentation maps of shape [batch_size, fH, fW].

            kwargs(Dict): Dictionary of keyword arguments, potentially containing following keys:
                - extended_analysis (bool): boolean indicating whether to perform extended analyses or not.

        Returns:
            * If tgt_dict is not None (i.e. during training and validation):
                loss_dict (Dictionary): Loss dictionary containing following key:
                    - binary_seg_loss (FloatTensor): weighted binary segmentation loss of shape [1].

                analysis_dict (Dictionary): Analysis dictionary containing at least following keys:
                    - binary_seg_accuracy (FloatTensor): accuracy of the binary segmentation of shape [1];
                    - binary_seg_error (FloatTensor): average error of the binary segmentation of shape [1].

            * If tgt_dict is None (i.e. during testing and possibly during validation):
                pred_dict (Dict): Prediction dictionary containing following key:
                    - binary_maps (List): predicted binary segmentation maps of shape [batch_size, fH, fW].
        """

        # Compute logit maps
        logit_maps = [proj(feat_map).squeeze(-1) for feat_map, proj in zip(feat_maps, self.projs)]

        # Compute and return dictionary with predicted binary segmentation maps (validation/testing only)
        if tgt_dict is None:
            pred_dict = {'binary_maps': [torch.clamp(logit_map/4 + 0.5, 0.0, 1.0) for logit_map in logit_maps]}
            return pred_dict

        # Flatten logit and target maps and initialize losses tensor (trainval only)
        logits = torch.cat([logit_map.flatten() for logit_map in logit_maps])
        targets = torch.cat([tgt_map.flatten() for tgt_map in tgt_dict['binary_maps']])
        losses = torch.zeros_like(logits)

        # Compute losses at background ground-truth positions (trainval only)
        bg_mask = targets == 0
        losses[bg_mask] = torch.log(1 + torch.exp(logits[bg_mask]))

        # Compute losses at object ground-truth positions (trainval only)
        obj_mask = targets == 1
        losses[obj_mask] = torch.log(1 + torch.exp(-logits[obj_mask]))

        # Compute losses at disputed ground-truth positions if desired (trainval only)
        if self.disputed_loss:
            disputed = torch.bitwise_and(targets > 0, targets < 1)
            smooth_l1_kwargs = {'reduction': 'none', 'beta': self.disputed_beta}
            losses[disputed] = F.smooth_l1_loss(logits[disputed]/4 + 0.5, targets[disputed], **smooth_l1_kwargs)

        # Apply map size corrections to losses if desired (trainval only)
        if self.map_size_correction:
            map_sizes = [logit_map.numel() for logit_map in logit_maps]
            scales = [map_sizes[-1]/map_size for map_size in map_sizes]
            indices = torch.cumsum(torch.tensor([0, *map_sizes], device=logits.device), dim=0)
            losses = torch.cat([scale*losses[i0:i1] for scale, i0, i1 in zip(scales, indices[:-1], indices[1:])])

        # Get loss dictionary with weighted binary segmentation loss (trainval only)
        batch_size = logit_maps[0].shape[0]
        avg_loss = torch.sum(losses) / batch_size
        loss_dict = {'binary_seg_loss': self.loss_weight * avg_loss}

        # Perform accuracy and error analyses and place them in analysis dictionary (trainval only)
        with torch.no_grad():
            preds = torch.clamp(logits/4 + 0.5, 0.0, 1.0)

            # Perform accuracy-related analyses and place them in analysis dictionary (trainval only)
            pred_correctness = torch.cat([preds[bg_mask] < 0.5, preds[obj_mask] > 0.5], dim=0)
            analysis_dict = BinarySegHead.perform_accuracy_analyses(pred_correctness, bg_mask.sum())

            # Perform error analysis and place it in analysis dictionary (trainval only)
            avg_error = torch.abs(preds - targets).mean()
            analysis_dict['binary_seg_error'] = avg_error

            # If desired, perform extended analyses (trainval only)
            if kwargs.setdefault('extended_analysis', False):

                # Perform map-specific accuracy analyses and them to analysis dictionary (trainval only)
                map_sizes = [logit_map.numel() for logit_map in logit_maps]
                indices = torch.cumsum(torch.tensor([0, *map_sizes], device=logits.device), dim=0)

                for i, i0, i1 in zip(range(len(logit_maps)), indices[:-1], indices[1:]):
                    map_preds = preds[i0:i1]
                    pred_correctness = [map_preds[bg_mask[i0:i1]] < 0.5, map_preds[obj_mask[i0:i1]] > 0.5]
                    pred_correctness = torch.cat(pred_correctness, dim=0)

                    num_bg_targets = bg_mask[i0:i1].sum()
                    map_analysis_dict = BinarySegHead.perform_accuracy_analyses(pred_correctness, num_bg_targets)
                    analysis_dict.update({f'{k}_f{i}': v for k, v in map_analysis_dict.items()})

        return loss_dict, analysis_dict

    def visualize(self, images, pred_dict, tgt_dict):
        """
        Draws predicted and target binary segmentations on given full-resolution images.

        Args:
            images (NestedTensor): NestedTensor consisting of:
                - images.tensor (FloatTensor): padded images of shape [batch_size, 3, max_iH, max_iW];
                - images.mask (BoolTensor): masks encoding inactive pixels of shape [batch_size, max_iH, max_iW].

            pred_dict (Dict): Prediction dictionary containing at least following key:
                - binary_maps (List): predicted binary segmentation maps of shape [batch_size, fH, fW].

            tgt_dict (Dict): Target dictionary containing at least following key:
                - binary_maps (List): target binary segmentation maps of shape [batch_size, fH, fW].

        Returns:
            images_dict (Dict): Dictionary of images with drawn predicted and target binary segmentations.
        """

        # Get padded images and corresponding image masks
        images, img_masks = images.decompose()

        # Get desired keys and merge prediction and target binary maps into single list
        keys = [f'pred_f{i}' for i in range(len(pred_dict['binary_maps']))]
        keys.extend([f'tgt_f{i}' for i in range(len(tgt_dict['binary_maps']))])
        binary_maps_list = [*pred_dict['binary_maps'], *tgt_dict['binary_maps']]

        # Get possible map sizes
        map_size = tuple(images.shape[-2:])
        map_sizes = [map_size]

        while map_size != (1, 1):
            map_size = ((map_size[0]+1)//2, (map_size[1]+1)//2)
            map_sizes.append(map_size)

        # Convert images to desired visualization format
        images = images.permute(0, 2, 3, 1)
        images = (images * 255).to(torch.uint8)
        images = images.cpu().numpy()

        # Get image sizes without padding in (height, width) format
        img_sizes = [(sum(~img_mask[:, 0]).item(), sum(~img_mask[0, :]).item()) for img_mask in img_masks]

        # Get interpolation kwargs and initialize dictionary of annotated images
        interpolation_kwargs = {'mode': 'bilinear', 'align_corners': True}
        images_dict = {}

        # Get annotated images for each batch of binary maps
        for key, binary_maps in zip(keys, binary_maps_list):

            # Get number of times the binary maps were downsampled
            map_size = tuple(binary_maps.shape[-2:])
            times_downsampled = map_sizes.index(map_size)

            # Upsample binary maps to image resolution
            for map_id in range(times_downsampled-1, -1, -1):
                H, W = map_sizes[map_id]
                pH, pW = (int(H % 2 == 0), int(W % 2 == 0))

                binary_maps = F.pad(binary_maps.unsqueeze(1), (0, pW, 0, pH), mode='replicate')
                binary_maps = F.interpolate(binary_maps, size=(H+pH, W+pW), **interpolation_kwargs)
                binary_maps = binary_maps[:, :, :H, :W].squeeze(1)

            # Get binary masks and convert them to NumPy ndarray
            binary_masks = (binary_maps >= 0.5).cpu().numpy()

            # Draw binary masks on corresponding images
            for i, image, img_size, binary_mask in zip(range(len(images)), images, img_sizes, binary_masks):
                visualizer = Visualizer(image)
                visualizer.draw_binary_mask(binary_mask)

                annotated_image = visualizer.output.get_image()
                images_dict[f'bin_seg_{key}_{i}'] = annotated_image[:img_size[0], :img_size[1], :]

        return images_dict


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
        if seg_head_type == 'binary':
            head_args = [args.disputed_loss, args.disputed_beta, not args.no_map_size_correction, args.bin_seg_weight]
            binary_seg_head = BinarySegHead(feat_sizes, *head_args)
            seg_heads.append(binary_seg_head)

        else:
            raise ValueError(f"Unknown segmentation head type '{seg_head_type}' was provided.")

    return seg_heads
