"""
Base segmentation head.
"""
from abc import ABCMeta, abstractmethod

from detectron2.structures.instances import Instances
from detectron2.utils.visualizer import ColorMode, Visualizer
import torch
from torch import nn
import torchvision.transforms.functional as T


class BaseSegHead(nn.Module, metaclass=ABCMeta):
    """
    Abstract class implementing the BaseSegHead module.
    """

    def __init__(self):
        """
        Initializes the BaseSegHead module.
        """

        # Initialization of default nn.Module
        super().__init__()

    @abstractmethod
    def compute_segs(self, storage_dict, pred_dicts, **kwargs):
        """
        Method computing the segmentation predictions.

        Args:
            storage_dict (Dict): Dictionary storing various items of interest.
            pred_dicts (List): List of size [num_pred_dicts] collecting various prediction dictionaries.
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            pred_dicts (List): List with prediction dictionaries containing following additional entry:
                pred_dict (Dict): Prediction dictionary containing following keys:
                    - labels (LongTensor): predicted class indices of shape [num_preds];
                    - masks (BoolTensor): predicted segmentation masks of shape [num_preds, iH, iW];
                    - scores (FloatTensor): normalized prediction scores of shape [num_preds];
                    - batch_ids (LongTensor): batch indices of predictions of shape [num_preds].
        """
        pass

    @torch.no_grad()
    def draw_segs(self, storage_dict, images_dict, pred_dicts, tgt_dict=None, vis_score_thr=0.4, id=None, **kwargs):
        """
        Draws predicted and target segmentations on the corresponding images.

        Segmentations must have a score of at least the score threshold to be drawn, and target segmentations get a
        default score of 100%.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following key:
                - images (Images): Images structure containing the batched images of size [batch_size].

            images_dict (Dict): Dictionary with annotated images of predictions/targets.

            pred_dicts (List): List with prediction dictionaries containing as last entry:
                pred_dict (Dict): Prediction dictionary containing following keys:
                    - labels (LongTensor): predicted class indices of shape [num_preds];
                    - masks (BoolTensor): predicted segmentation masks of shape [num_preds, iH, iW];
                    - scores (FloatTensor): normalized prediction scores of shape [num_preds];
                    - batch_ids (LongTensor): batch indices of predictions of shape [num_preds].

            tgt_dict (Dict): Optional target dictionary containing at least following keys when given:
                - labels (LongTensor): target class indices of shape [num_targets];
                - masks (BoolTensor): target segmentation masks of shape [num_targets, iH, iW];
                - sizes (LongTensor): cumulative number of targets per batch entry of size [batch_size+1].

            vis_score_thr (float): Threshold indicating the minimum score for an instance to be drawn (default=0.4).
            id (int or str): Integer or string containing the head id (default=None).
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            images_dict (Dict): Dictionary containing additional images annotated with segmentations.

        Raises:
            ValueError: Error when an invalid segmentation type is provided.
        """

        # Retrieve images from storage dictionary and get batch size
        images = storage_dict['images']
        batch_size = len(images)

        # Initialize list of draw dictionaries and list of dictionary names
        draw_dicts = []
        dict_names = []

        # Get prediction draw dictionary and dictionary name
        pred_dict = pred_dicts[-1]
        draw_dict = {}

        if self.seg_type == 'instance':
            pred_scores = pred_dict['scores']
            sufficient_score = pred_scores >= vis_score_thr

            draw_dict['labels'] = pred_dict['labels'][sufficient_score]
            draw_dict['masks'] = pred_dict['masks'][sufficient_score]
            draw_dict['scores'] = pred_scores[sufficient_score]

            pred_batch_ids = pred_dict['batch_ids'][sufficient_score]
            pred_sizes = [0] + [(pred_batch_ids == i).sum() for i in range(batch_size)]
            pred_sizes = torch.tensor(pred_sizes, device=pred_scores.device).cumsum(dim=0)
            draw_dict['sizes'] = pred_sizes

        elif self.seg_type == 'panoptic':
            thing_ids = tuple(self.metadata.thing_dataset_id_to_contiguous_id.values())

            pred_labels = pred_dict['labels']
            pred_masks = pred_dict['masks']
            pred_batch_ids = pred_dict['batch_ids']

            pred_sizes = [0] + [(pred_batch_ids == i).sum() for i in range(batch_size)]
            pred_sizes = torch.tensor(pred_sizes, device=pred_labels.device).cumsum(dim=0)
            draw_dict['sizes'] = pred_sizes

            mask_list = []
            segments = []

            for i0, i1 in zip(pred_sizes[:-1], pred_sizes[1:]):
                max_vals, mask_i = pred_masks[i0:i1].int().max(dim=0)
                mask_i[max_vals == 0] = -1
                mask_list.append(mask_i)

                for j in range(i1-i0):
                    cat_id = pred_labels[i0+j]
                    is_thing = cat_id in thing_ids

                    segment_ij = {'id': j, 'category_id': cat_id, 'isthing': is_thing}
                    segments.append(segment_ij)

            draw_dict['masks'] = torch.stack(mask_list, dim=0)
            draw_dict['segments'] = segments

        else:
            error_msg = f"Invalid segmentation type in BaseSegHead (got '{self.seg_type}')."
            raise ValueError(error_msg)

        draw_dicts.append(draw_dict)
        dict_name = f"seg_pred_{id}" if id is not None else "seg_pred"
        dict_names.append(dict_name)

        # Get target draw dictionary and dictionary name if needed
        if tgt_dict is not None and not any('seg_tgt' in key for key in images_dict.keys()):
            draw_dict = {}

            if self.seg_type == 'instance':
                draw_dict['labels'] = tgt_dict['labels']
                draw_dict['masks'] = tgt_dict['masks']
                draw_dict['scores'] = torch.ones_like(tgt_dict['labels'], dtype=torch.float)
                draw_dict['sizes'] = tgt_dict['sizes']

            elif self.seg_type == 'panoptic':
                tgt_labels = tgt_dict['labels']
                tgt_masks = tgt_dict['masks']
                tgt_sizes = tgt_dict['sizes']

                mask_list = []
                segments = []

                for i0, i1 in zip(tgt_sizes[:-1], tgt_sizes[1:]):
                    mask_i = tgt_masks[i0:i1].int().argmax(dim=0)
                    mask_list.append(mask_i)

                    for j in range(i1-i0):
                        cat_id = tgt_labels[i0+j]
                        is_thing = cat_id in thing_ids

                        segment_ij = {'id': j, 'category_id': cat_id, 'isthing': is_thing}
                        segments.append(segment_ij)

                draw_dict['masks'] = torch.stack(mask_list, dim=0)
                draw_dict['segments'] = segments
                draw_dict['sizes'] = tgt_sizes

            else:
                error_msg = f"Invalid segmentation type in BaseSegHead (got '{self.seg_type}')."
                raise ValueError(error_msg)

            draw_dicts.append(draw_dict)
            dict_names.append('seg_tgt')

        # Get image sizes without padding in (width, height) format
        img_sizes = images.size(mode='without_padding')

        # Get image sizes with padding in (height, width) format
        img_sizes_pad = images.size(mode='with_padding')
        img_sizes_pad = (img_sizes_pad[1], img_sizes_pad[0])

        # Draw 2D object detections on images and add them to images dictionary
        for dict_name, draw_dict in zip(dict_names, draw_dicts):
            sizes = draw_dict['sizes']

            for i, i0, i1 in zip(range(batch_size), sizes[:-1], sizes[1:]):
                img_size = img_sizes[i]
                img_size = (img_size[1], img_size[0])

                image = images.images[i].clone()
                image = T.crop(image, 0, 0, *img_size)
                image = image.permute(1, 2, 0) * 255
                image = image.to(torch.uint8).cpu().numpy()

                if self.seg_type == 'instance':
                    visualizer = Visualizer(image, metadata=self.metadata, instance_mode=ColorMode.IMAGE)

                    if i1 > i0:
                        labels_i = draw_dict['labels'][i0:i1].cpu().numpy()
                        scores_i = draw_dict['scores'][i0:i1].cpu().numpy()

                        masks_i = draw_dict['masks'][i0:i1]
                        masks_i = T.resize(masks_i, img_sizes_pad)
                        masks_i = T.crop(masks_i, 0, 0, *img_size)
                        masks_i = masks_i.cpu().numpy()

                        instances = Instances(img_size, pred_classes=labels_i, pred_masks=masks_i, scores=scores_i)
                        visualizer.draw_instance_predictions(instances)

                elif self.seg_type == 'panoptic':
                    mask_i = draw_dict['masks'][i]
                    mask_i = T.crop(mask_i, 0, 0, *img_size).cpu()
                    segments_i = draw_dict['segments'][i0:i1]

                    visualizer = Visualizer(image, metadata=self.metadata, instance_mode=ColorMode.SEGMENTATION)
                    visualizer.draw_panoptic_seg(mask_i, segments_i)

                else:
                    error_msg = f"Invalid segmentation type in BaseSegHead (got '{self.seg_type}')."
                    raise ValueError(error_msg)

                annotated_image = visualizer.output.get_image()
                images_dict[f'{dict_name}_{i}'] = annotated_image

        return images_dict

    @abstractmethod
    def forward_pred(self, storage_dict, images_dict=None, **kwargs):
        """
        Forward prediction method of the BaseSegHead module.

        Args:
            storage_dict (Dict): Dictionary storing various items of interest.
            images_dict (Dict): Dictionary with annotated images of predictions/targets (default=None).
            kwargs (Dict): Dictionary of keyword arguments passed to some underlying methods.

        Returns:
            storage_dict (Dict): Dictionary with (possibly) additional stored items of interest.
            images_dict (Dict): Dictionary (possibly) containing additional images annotated with segmentations.
        """
        pass

    @abstractmethod
    def forward_loss(self, storage_dict, tgt_dict, loss_dict, analysis_dict=None, id=None, **kwargs):
        """
        Forward loss method of the BaseSegHead module.

        Args:
            storage_dict (Dict): Dictionary storing various items of interest.
            tgt_dict (Dict): Target dictionary with ground-truth information used during trainval.
            loss_dict (Dict): Dictionary containing different weighted loss terms.
            analysis_dict (Dict): Dictionary containing different analyses (default=None).
            id (int or str): Integer or string containing the head id (default=None).
            kwargs (Dict): Dictionary of keyword arguments passed to some underlying modules.

        Returns:
            loss_dict (Dict): Loss dictionary (possibly) containing additional weighted loss terms.
            analysis_dict (Dict): Analysis dictionary (possibly) containing additional analyses (or None).
        """
        pass

    def forward(self, mode, **kwargs):
        """
        Forward method of the BaseSegHead module.

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
