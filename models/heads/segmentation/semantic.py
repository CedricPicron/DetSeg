"""
Semantic segmentation head.
"""

from detectron2.utils.visualizer import Visualizer
import torch
from torch import nn
import torch.nn.functional as F

from models.projector import Projector
from models.utils import downsample_index_maps


class SemanticSegHead(nn.Module):
    """
    Class implementing the SemanticSegHead module, performing semantic segmentation.

    Attributes:
        proj (Projector): Module projecting feature maps to logit maps from which predictions are made.
        num_classes (int): Integer containing the number of object classes (without background).
        class_weights (FloatTensor): Tensor of shape [num_classes+1] containing class-specific loss weights.
        loss_weight (float): Weight factor used to scale the semantic segmentation loss as a whole.
        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
    """

    def __init__(self, feat_sizes, num_classes, bg_weight, loss_weight, metadata):
        """
        Initializes the SemanticSegHead module.

        Args:
            feat_sizes (List): List of size [num_maps] containing the feature size of each map.
            num_classes (int): Integer containing the number of object classes (without background).
            bg_weight (float): Cross entropy weight scaling the losses in target background positions.
            loss_weight (float): Weight factor used to scale the semantic segmentation loss.
            metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Initialization of projector module
        fixed_settings = {'out_feat_size': num_classes+1, 'proj_type': 'conv1', 'conv_stride': 1}
        proj_dicts = [{'in_map_id': i, **fixed_settings} for i in range(len(feat_sizes))]
        self.proj = Projector(feat_sizes, proj_dicts)

        # Register buffer of cross-entropy class weights
        self.register_buffer('class_weights', torch.ones(num_classes+1))
        self.class_weights[num_classes] = bg_weight

        # Set number of classes and loss weight attributes
        self.num_classes = num_classes
        self.loss_weight = loss_weight

        # Set dataset metadata attribute used during visualization
        metadata.stuff_classes = metadata.thing_classes
        metadata.stuff_colors = metadata.thing_colors
        self.metadata = metadata

    @staticmethod
    def get_accuracy(preds, targets):
        """
        Method returning the accuracy of the given predictions compared to the given targets.

        Args:
            preds (LongTensor): Tensor of shape [num_targets] containing the predicted class indices.
            targets (LongTensor): Tensor of shape [num_targets] containing the target class indices.

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

    def perform_accuracy_analyses(self, preds, targets):
        """
        Method performing accuracy-related analyses.

        Args:
            preds (LongTensor): Tensor of shape [num_targets] containing the predicted class indices.
            targets (LongTensor): Tensor of shape [num_targets] containing the target class indices.

        Returns:
            analysis_dict (Dict): Dictionary of accuracy-related analyses containing following keys:
                - sem_seg_acc (FloatTensor): accuracy of the semantic segmentation of shape [];
                - sem_seg_acc_bg (FloatTensor): background accuracy of the semantic segmentation of shape [];
                - sem_seg_acc_obj (FloatTensor): object accuracy of the semantic segmentation of shape [].
        """

        # Compute general accuracy and place it into analysis dictionary
        accuracy = SemanticSegHead.get_accuracy(preds, targets)
        analysis_dict = {'sem_seg_acc': 100*accuracy}

        # Compute background accuracy and place it into analysis dictionary
        bg_mask = targets == self.num_classes
        bg_accuracy = SemanticSegHead.get_accuracy(preds[bg_mask], targets[bg_mask])
        analysis_dict['sem_seg_acc_bg'] = 100*bg_accuracy

        # Compute object accuracy and place it into analysis dictionary
        obj_mask = targets < self.num_classes
        obj_accuracy = SemanticSegHead.get_accuracy(preds[obj_mask], targets[obj_mask])
        analysis_dict['sem_seg_acc_obj'] = 100*obj_accuracy

        return analysis_dict

    @torch.no_grad()
    def forward_init(self, images, feat_maps, tgt_dict=None):
        """
        Forward initialization method of the SemanticSegHead module.

        The forward initialization consists of 2 steps:
            1) Get the desired full-resolution semantic segmentation masks.
            2) Downsample the full-resolution masks to maps with the same resolutions as found in 'feat_maps'.

        Args:
            images (Images): Images structure containing the batched images.
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

            tgt_dict (Dict): Optional target dictionary used during trainval containing at least following keys:
                - labels (LongTensor): tensor of shape [num_targets_total] containing the class indices;
                - sizes (LongTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries;
                - masks (ByteTensor): padded segmentation masks of shape [num_targets_total, max_iH, max_iW].

        Returns:
            * If tgt_dict is not None (i.e. during training and validation):
                tgt_dict (Dict): Updated target dictionary containing following additional key:
                    - semantic_maps (List): semantic segmentation maps with class indices of shape [batch_size, fH, fW].

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

        # Some renaming for clarity
        tgt_labels = tgt_dict['labels']
        tgt_sizes = tgt_dict['sizes']
        tgt_masks = tgt_dict['masks']

        # Initialize full-resolution semantic maps
        batch_size = len(tgt_sizes) - 1
        tensor_kwargs = {'dtype': torch.long, 'device': tgt_masks.device}
        semantic_maps = torch.full((batch_size, *tgt_masks.shape[-2:]), self.num_classes, **tensor_kwargs)

        # Compute full-resolution semantic maps for each batch entry
        for i, i0, i1 in zip(range(batch_size), tgt_sizes[:-1], tgt_sizes[1:]):
            for tgt_label, tgt_mask in zip(tgt_labels[i0:i1], tgt_masks[i0:i1]):
                semantic_maps[i].masked_fill_(tgt_mask, tgt_label)

        # Compute and place downsampled semantic masks into target dictionary
        map_sizes = [tuple(feat_map.shape[2:]) for feat_map in feat_maps]
        tgt_dict['semantic_maps'] = downsample_index_maps(semantic_maps, map_sizes)

        return tgt_dict, {}, {}

    def forward(self, feat_maps, feat_masks=None, tgt_dict=None, **kwargs):
        """
        Forward method of the SemanticSegHead module.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].
            feat_masks (List): Optional list [num_maps] with masks of active features of shape [batch_size, fH, fW].

            tgt_dict (Dict): Optional target dictionary during training and validation with at least following key:
                - semantic_maps (List): semantic segmentation maps with class indices of shape [batch_size, fH, fW].

            kwargs (Dict): Dictionary of keyword arguments, potentially containing following key:
                - extended_analysis (bool): boolean indicating whether to perform extended analyses or not.

        Returns:
            * If tgt_dict is not None (i.e. during training and validation):
                loss_dict (Dictionary): Loss dictionary containing following key:
                    - sem_seg_loss (FloatTensor): weighted semantic segmentation loss of shape [].

                analysis_dict (Dictionary): Analysis dictionary containing at least following keys:
                    - sem_seg_acc (FloatTensor): accuracy of the semantic segmentation of shape [];
                    - sem_seg_acc_bg (FloatTensor): background accuracy of the semantic segmentation of shape [];
                    - sem_seg_acc_obj (FloatTensor): object accuracy of the semantic segmentation of shape [].

            * If tgt_dict is None (i.e. during testing and possibly during validation):
                pred_dict (Dict): Prediction dictionary containing following key:
                    - semantic_maps (List): predicted semantic segmentation maps of shape [batch_size, fH, fW].
        """

        # Compute logit maps
        logit_maps = self.proj(feat_maps)

        # Compute and return dictionary with predicted semantic segmentation maps (validation/testing only)
        if tgt_dict is None:
            pred_dict = {'semantic_maps': [torch.argmax(logit_map, dim=1) for logit_map in logit_maps]}
            return pred_dict

        # Compute average cross entropy losses corresponding to each map (trainval only)
        tgt_maps = tgt_dict['semantic_maps']
        avg_losses = [F.cross_entropy(map0, map1, self.class_weights) for map0, map1 in zip(logit_maps, tgt_maps)]

        # Get loss dictionary with weighted semantic segmentation loss (trainval only)
        loss_dict = {'sem_seg_loss': self.loss_weight * sum(avg_losses)}

        # Perform accuracy and error analyses and place them in analysis dictionary (trainval only)
        with torch.no_grad():

            # Get list of prediction maps
            pred_maps = [torch.argmax(logit_map, dim=1) for logit_map in logit_maps]

            # Flatten and concatenate prediction and target maps (trainval only)
            preds = torch.cat([pred_map.flatten() for pred_map in pred_maps])
            targets = torch.cat([tgt_map.flatten() for tgt_map in tgt_maps])

            # Assume no padded regions when feature masks are missing (trainval only)
            if feat_masks is None:
                tensor_kwargs = {'dtype': torch.bool, 'device': feat_maps[0].device}
                feat_masks = [torch.ones(*feat_map[:, 0].shape, **tensor_kwargs) for feat_map in feat_maps]

            # Flatten and concatenate masks of active features (trainval only)
            active_mask = torch.cat([feat_mask.flatten() for feat_mask in feat_masks])

            # Perform accuracy-related analyses and place them in analysis dictionary (trainval only)
            analysis_dict = self.perform_accuracy_analyses(preds[active_mask], targets[active_mask])

            # If requested, perform extended analyses (trainval only)
            if kwargs.setdefault('extended_analysis', False):

                # Perform map-specific accuracy analyses and place them into analysis dictionary (trainval only)
                map_sizes = [tgt_map.numel() for tgt_map in tgt_maps]
                indices = torch.cumsum(torch.tensor([0, *map_sizes], device=targets.device), dim=0)

                for i, i0, i1 in zip(range(len(tgt_maps)), indices[:-1], indices[1:]):
                    map_acc_mask = active_mask[i0:i1]
                    map_preds = preds[i0:i1][map_acc_mask]
                    map_targets = targets[i0:i1][map_acc_mask]

                    map_analysis_dict = self.perform_accuracy_analyses(map_preds, map_targets)
                    analysis_dict.update({f'{k}_f{i}': v for k, v in map_analysis_dict.items()})

        return loss_dict, analysis_dict

    def visualize(self, images, pred_dict, tgt_dict):
        """
        Draws predicted and target semantic segmentations on given full-resolution images.

        Args:
            images (Images): Images structure containing the batched images.

            pred_dict (Dict): Prediction dictionary containing at least following key:
                - semantic_maps (List): predicted semantic segmentation maps of shape [batch_size, fH, fW].

            tgt_dict (Dict): Target dictionary containing at least following key:
                - semantic_maps (List): semantic segmentation maps with class indices of shape [batch_size, fH, fW].

        Returns:
            images_dict (Dict): Dictionary of images with drawn predicted and target semantic segmentations.
        """

        # Get desired keys and merge prediction and target semantic maps into single list
        keys = [f'pred_f{i}' for i in range(len(pred_dict['semantic_maps']))]
        keys.extend([f'tgt_f{i}' for i in range(len(tgt_dict['semantic_maps']))])
        semantic_maps_list = [*pred_dict['semantic_maps'], *tgt_dict['semantic_maps']]

        # Get possible map sizes in (height, width) format
        map_width, map_height = images.size()
        map_sizes = [(map_height, map_width)]

        while (map_height, map_width) != (1, 1):
            map_height = (map_height+1)//2
            map_width = (map_width+1)//2
            map_sizes.append((map_height, map_width))

        # Get image sizes without padding in (width, height) format
        img_sizes = images.size(with_padding=False)

        # Get and convert tensor with images
        images = images.images.clone().permute(0, 2, 3, 1)
        images = (images * 255).to(torch.uint8).cpu().numpy()

        # Get interpolation kwargs and initialize dictionary of annotated images
        interpolation_kwargs = {'mode': 'bilinear', 'align_corners': True}
        images_dict = {}

        # Get annotated images for each batch of semantic maps
        for key, semantic_maps in zip(keys, semantic_maps_list):

            # Get number of times the semantic maps were downsampled
            map_size = tuple(semantic_maps.shape[-2:])
            times_downsampled = map_sizes.index(map_size)

            # Get class-specific semantic maps
            class_semantic_maps = torch.stack([semantic_maps == i for i in range(self.num_classes+1)], dim=1).float()

            # Upsample class-specific semantic maps to image resolution
            for map_id in range(times_downsampled-1, -1, -1):
                H, W = map_sizes[map_id]
                pH, pW = (int(H % 2 == 0), int(W % 2 == 0))

                class_semantic_maps = F.pad(class_semantic_maps, (0, pW, 0, pH), mode='replicate')
                class_semantic_maps = F.interpolate(class_semantic_maps, size=(H+pH, W+pW), **interpolation_kwargs)
                class_semantic_maps = class_semantic_maps[:, :, :H, :W]

            # Get general (non-class specific) semantic maps anc convert to NumPy ndarray
            semantic_maps = torch.argmax(class_semantic_maps, dim=1).cpu().numpy()

            # Draw semantic masks on corresponding images
            for i, image, img_size, semantic_map in zip(range(len(images)), images, img_sizes, semantic_maps):
                visualizer = Visualizer(image, metadata=self.metadata)
                visualizer.draw_sem_seg(semantic_map)

                annotated_image = visualizer.output.get_image()
                images_dict[f'sem_seg_{key}_{i}'] = annotated_image[:img_size[1], :img_size[0], :]

        return images_dict
