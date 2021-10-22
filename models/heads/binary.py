"""
Binary segmentation head.
"""

from detectron2.utils.visualizer import Visualizer
import torch
from torch import nn
import torch.nn.functional as F

from models.functional.downsample import downsample_masks
from models.modules.projector import Projector


class BinarySegHead(nn.Module):
    """
    Class implementing the BinarySegHead module, segmenting objects from background.

    Attributes:
        proj (Projector): Module projecting feature maps to logit maps from which predictions are made.
        disputed_loss (bool): Bool indicating if loss should be applied at disputed ground-truth positions.
        disputed_beta (float): Threshold value at which disputed smooth L1 loss changes from L1 to L2 loss.
        loss_weight (float): Weight factor used to scale the binary segmentation loss.
    """

    def __init__(self, feat_sizes, disputed_loss, disputed_beta, loss_weight):
        """
        Initializes the BinarySegHead module.

        Args:
            feat_sizes (List): List of size [num_maps] containing the feature size of each map.
            disputed_loss (bool): Bool indicating if loss should be applied at disputed ground-truth positions.
            disputed_beta (float): Threshold value at which disputed smooth L1 loss changes from L1 to L2 loss.
            loss_weight (float): Weight factor used to scale the binary segmentation loss.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Initialization of projector module
        fixed_settings = {'out_feat_size': 1, 'proj_type': 'conv1', 'conv_stride': 1}
        proj_dicts = [{'in_map_id': i, **fixed_settings} for i in range(len(feat_sizes))]
        self.proj = Projector(feat_sizes, proj_dicts)

        # Set remaining attributes as specified by the input arguments
        self.disputed_loss = disputed_loss
        self.disputed_beta = disputed_beta
        self.loss_weight = loss_weight

    @staticmethod
    def get_accuracy(preds, targets):
        """
        Method returning the accuracy of the given predictions compared to the given targets.

        Args:
            preds (BoolTensor): Tensor of shape [num_targets] containing the object/background predictions.
            targets (BoolTensor): Tensor of shape [num_targets] containing the object/background targets.

        Returns:
            accuracy (FloatTensor): Tensor of shape [] containing the prediction accuracy (between 0 and 1).
        """

        # Get boolean tensor indicating correct and incorrect predictions
        pred_correctness = torch.eq(preds, targets)

        # Compute accuracy
        if len(pred_correctness) > 0:
            accuracy = pred_correctness.sum() / float(len(pred_correctness))
        else:
            accuracy = torch.tensor(1.0).to(pred_correctness.device)

        return accuracy

    @staticmethod
    def perform_accuracy_analyses(preds, targets):
        """
        Method performing accuracy-related analyses.

        Args:
            preds (BoolTensor): Tensor of shape [num_targets] containing the object/background predictions.
            targets (BoolTensor): Tensor of shape [num_targets] containing the object/background targets.

        Returns:
            analysis_dict (Dict): Dictionary of accuracy-related analyses containing following keys:
                - bin_seg_acc (FloatTensor): accuracy of the binary segmentation of shape [];
                - bin_seg_acc_bg (FloatTensor): background accuracy of the binary segmentation of shape [];
                - bin_seg_acc_obj (FloatTensor): object accuracy of the binary segmentation of shape [].
        """

        # Compute general accuracy and place it into analysis dictionary
        accuracy = BinarySegHead.get_accuracy(preds, targets)
        analysis_dict = {'bin_seg_acc': 100*accuracy}

        # Compute background accuracy and place it into analysis dictionary
        bg_mask = targets == 0
        bg_accuracy = BinarySegHead.get_accuracy(preds[bg_mask], targets[bg_mask])
        analysis_dict['bin_seg_acc_bg'] = 100*bg_accuracy

        # Compute object accuracy and place it into analysis dictionary
        obj_mask = targets == 1
        obj_accuracy = BinarySegHead.get_accuracy(preds[obj_mask], targets[obj_mask])
        analysis_dict['bin_seg_acc_obj'] = 100*obj_accuracy

        return analysis_dict

    @torch.no_grad()
    def forward_init(self, images, feat_maps, tgt_dict=None):
        """
        Forward initialization method of the BinarySegHead module.

        The forward initialization consists of 2 steps:
            1) Get the desired full-resolution binary masks.
            2) Downsample the full-resolution masks to maps with the same resolutions as found in 'feat_maps'.

        Args:
            images (Images): Images structure containing the batched images.
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

            tgt_dict (Dict): Optional target dictionary used during trainval containing at least following keys:
                - sizes (LongTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries;
                - masks (ByteTensor): padded segmentation masks of shape [num_targets_total, max_iH, max_iW].

        Returns:
            * If tgt_dict is not None (i.e. during training and validation):
                tgt_dict (Dict): Updated target dictionary containing following additional key:
                    - binary_maps (List): binary (object + background) segmentation maps of shape [batch_size, fH, fW].

                attr_dict (Dict): Empty dictionary.
                buffer_dict (Dict): Empty dictionary.

            * If tgt_dict is None (i.e. during testing):
                tgt_dict (None): Contains the None value.
                attr_dict (Dict): Empty dictionary.
                buffer_dict (Dict): Empty dictionary.
        """

        # Return when no target dictionary is provided (testing only)
        if tgt_dict is None:
            return None, {}, {}

        # Get full-resolution binary target masks (trainval only)
        tgt_sizes = tgt_dict['sizes']
        tgt_masks = tgt_dict['masks']
        binary_masks = torch.stack([tgt_masks[i0:i1].any(dim=0) for i0, i1 in zip(tgt_sizes[:-1], tgt_sizes[1:])])

        # Compute and place downsampled binary maps into target dictionary (trainval only)
        map_sizes = [tuple(feat_map.shape[2:]) for feat_map in feat_maps]
        tgt_dict['binary_maps'] = downsample_masks(binary_masks, map_sizes)

        return tgt_dict, {}, {}

    def forward(self, feat_maps, feat_masks=None, tgt_dict=None, **kwargs):
        """
        Forward method of the BinarySegHead module.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].
            feat_masks (List): Optional list [num_maps] with masks of active features of shape [batch_size, fH, fW].

            tgt_dict (Dict): Optional target dictionary during training and validation with at least following key:
                - binary_maps (List): binary (object + background) segmentation maps of shape [batch_size, fH, fW].

            kwargs (Dict): Dictionary of keyword arguments, potentially containing following key:
                - extended_analysis (bool): boolean indicating whether to perform extended analyses or not.

        Returns:
            * If tgt_dict is not None (i.e. during training and validation):
                loss_dict (Dictionary): Loss dictionary containing following key:
                    - binary_seg_loss (FloatTensor): weighted binary segmentation loss of shape [].

                analysis_dict (Dictionary): Analysis dictionary containing at least following keys:
                    - bin_seg_acc (FloatTensor): accuracy of the binary segmentation of shape [];
                    - bin_seg_acc_bg (FloatTensor): background accuracy of the binary segmentation of shape [];
                    - bin_seg_acc_obj (FloatTensor): object accuracy of the binary segmentation of shape [];
                    - bin_seg_error (FloatTensor): average error of the binary segmentation of shape [].

            * If tgt_dict is None (i.e. during testing and possibly during validation):
                pred_dicts (List): List of prediction dictionaries with each dictionary containing following keys:
                    - binary_maps (List): predicted binary segmentation maps of shape [batch_size, fH, fW].
        """

        # Compute logit maps
        logit_maps = self.proj(feat_maps)
        logit_maps = [logit_map.squeeze(1) for logit_map in logit_maps]

        # Compute and return dictionary with predicted binary segmentation maps (validation/testing only)
        if tgt_dict is None:
            pred_dict = {'binary_maps': [torch.clamp(logit_map/4 + 0.5, 0.0, 1.0) for logit_map in logit_maps]}
            pred_dicts = [pred_dict]

            return pred_dicts

        # Flatten and concatenate logit and target maps (trainval only)
        logits = torch.cat([logit_map.flatten() for logit_map in logit_maps])
        targets = torch.cat([tgt_map.flatten() for tgt_map in tgt_dict['binary_maps']])

        # Initialize losses tensor (trainval only)
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

        # Get average losses corresponding to each map (trainval only)
        map_sizes = [logit_map.numel() for logit_map in logit_maps]
        indices = torch.cumsum(torch.tensor([0, *map_sizes], device=logits.device), dim=0)
        avg_losses = [torch.mean(losses[i0:i1]) for i0, i1 in zip(indices[:-1], indices[1:])]

        # Get loss dictionary with weighted binary segmentation loss (trainval only)
        loss_dict = {'bin_seg_loss': self.loss_weight * sum(avg_losses)}

        # Perform accuracy and error analyses and place them in analysis dictionary (trainval only)
        with torch.no_grad():

            # Get predictions (trainval only)
            preds = torch.clamp(logits/4 + 0.5, 0.0, 1.0)

            # Assume no padded regions when feature masks are missing (trainval only)
            if feat_masks is None:
                tensor_kwargs = {'dtype': torch.bool, 'device': feat_maps[0].device}
                feat_masks = [torch.ones(*feat_map[:, 0].shape, **tensor_kwargs) for feat_map in feat_maps]

            # Flatten and concatenate masks of active features (trainval only)
            active_mask = torch.cat([feat_mask.flatten() for feat_mask in feat_masks])

            # Get mask of entries that will be used during accuracy-related analyses (trainval only)
            acc_mask = torch.bitwise_or(bg_mask, obj_mask)
            acc_mask = torch.bitwise_and(acc_mask, active_mask)

            # Perform accuracy-related analyses and place them in analysis dictionary (trainval only)
            analysis_dict = BinarySegHead.perform_accuracy_analyses(preds[acc_mask] > 0.5, targets[acc_mask] > 0.5)

            # Perform error analysis and place it in analysis dictionary (trainval only)
            analysis_dict['bin_seg_error'] = torch.abs(preds[active_mask] - targets[active_mask]).mean()

            # If requested, perform extended analyses (trainval only)
            if kwargs.setdefault('extended_analysis', False):

                # Perform map-specific accuracy analyses and place them into analysis dictionary (trainval only)
                map_sizes = [logit_map.numel() for logit_map in logit_maps]
                indices = torch.cumsum(torch.tensor([0, *map_sizes], device=targets.device), dim=0)

                for i, i0, i1 in zip(range(len(logit_maps)), indices[:-1], indices[1:]):
                    map_acc_mask = acc_mask[i0:i1]
                    map_preds = preds[i0:i1][map_acc_mask]
                    map_targets = targets[i0:i1][map_acc_mask]

                    map_analysis_dict = BinarySegHead.perform_accuracy_analyses(map_preds > 0.5, map_targets > 0.5)
                    analysis_dict.update({f'{k}_f{i}': v for k, v in map_analysis_dict.items()})

        return loss_dict, analysis_dict

    def visualize(self, images, pred_dicts, tgt_dict):
        """
        Draws predicted and target binary segmentations on given full-resolution images.

        Args:
            images (Images): Images structure containing the batched images.

            pred_dicts (List): List of prediction dictionaries with each dictionary containing following keys:
                - binary_maps (List): predicted binary segmentation maps of shape [batch_size, fH, fW].

            tgt_dict (Dict): Target dictionary containing at least following key:
                - binary_maps (List): target binary segmentation maps of shape [batch_size, fH, fW].

        Returns:
            images_dict (Dict): Dictionary of images with drawn predicted and target binary segmentations.
        """

        # Combine prediction and target binary maps and get corresponding map names
        binary_maps = [map for p in pred_dicts for map in p['binary_maps']] + tgt_dict['binary_maps']
        map_names = [f'pred_{i}_f{j}' for i, p in enumerate(pred_dicts, 1) for j in range(len(p['binary_maps']))]
        map_names.extend([f'tgt_f{j}' for j in range(len(tgt_dict['binary_maps']))])

        # Get possible map sizes in (height, width) format
        map_width, map_height = images.size(mode='with_padding')
        map_sizes = [(map_height, map_width)]

        while (map_height, map_width) != (1, 1):
            map_height = (map_height+1)//2
            map_width = (map_width+1)//2
            map_sizes.append((map_height, map_width))

        # Get image sizes without padding in (width, height) format
        img_sizes = images.size(mode='without_padding')

        # Get and convert tensor with images
        images = images.images.clone().permute(0, 2, 3, 1)
        images = (images * 255).to(torch.uint8).cpu().numpy()

        # Get interpolation kwargs and initialize dictionary of annotated images
        interpolation_kwargs = {'mode': 'bilinear', 'align_corners': True}
        images_dict = {}

        # Get annotated images for each binary map
        for map_name, binary_map in zip(map_names, binary_maps):

            # Get number of times the binary map was downsampled
            map_size = tuple(binary_map.shape[-2:])
            times_downsampled = map_sizes.index(map_size)

            # Upsample binary map to image resolution
            for map_id in range(times_downsampled-1, -1, -1):
                H, W = map_sizes[map_id]
                pH, pW = (int(H % 2 == 0), int(W % 2 == 0))

                binary_map = F.pad(binary_map.unsqueeze(1), (0, pW, 0, pH), mode='replicate')
                binary_map = F.interpolate(binary_map, size=(H+pH, W+pW), **interpolation_kwargs)
                binary_map = binary_map[:, :, :H, :W].squeeze(1)

            # Get binary mask and convert it to NumPy ndarray
            binary_mask = (binary_map >= 0.5).cpu().numpy()

            # Draw image binary masks on corresponding images
            for i, image, img_size, img_binary_mask in zip(range(len(images)), images, img_sizes, binary_mask):
                visualizer = Visualizer(image)
                visualizer.draw_binary_mask(img_binary_mask)

                annotated_image = visualizer.output.get_image()
                images_dict[f'bin_seg_{map_name}_{i}'] = annotated_image[:img_size[1], :img_size[0], :]

        return images_dict
