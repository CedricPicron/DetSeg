"""
Collection of RoI (Region of Interest) heads.
"""

from detectron2.structures.instances import Instances
from detectron2.utils.visualizer import Visualizer
from mmdet.models.roi_heads import StandardRoIHead as MMDetStandardRoIHead
import torch
import torchvision.transforms.functional as T

from models.build import MODELS


@MODELS.register_module()
class StandardRoIHead(MMDetStandardRoIHead):
    """
    Class implementing the StandardRoIHead module.

    The module is based on the StandardRoIHead module from MMDetection.

    Attributes:
        get_segs (bool): Boolean indicating whether to get segmentation predictions.
        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
    """

    def __init__(self, metadata, get_segs=True, **kwargs):
        """
        Initializes the StandardRoIHead.

        Args:
            metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
            get_segs (bool): Boolean indicating whether to get segmentation predictions (default=True).
            kwargs (Dict): Dictionary of keyword arguments passed to the parent __init__ method.
        """

        # Initialize module using parent __init__ method
        super().__init__(**kwargs)

        # Set additional attributes
        self.get_segs = get_segs
        self.metadata = metadata

    @torch.no_grad()
    def compute_segs(self, storage_dict, pred_dicts, cum_feats_batch=None, **kwargs):
        """
        Method computing the segmentation predictions.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - cls_logits (FloatTensor): classification logits of shape [num_feats, num_labels];
                - pred_boxes (Boxes): predicted 2D bounding boxes of size [num_feats];
                - mask_logits (FloatTensor): map with mask logits of shape [num_feats, mC, mH, mW].

            pred_dicts (List): List of size [num_pred_dicts] collecting various prediction dictionaries.
            cum_feats_batch (LongTensor): Cumulative number of features per batch entry [batch_size+1] (default=None).

        Returns:
            pred_dicts (List): List with prediction dictionaries containing following additional entry:
                pred_dict (Dict): Prediction dictionary containing following keys:
                    - labels (LongTensor): predicted class indices of shape [num_preds];
                    - masks (BoolTensor): predicted segmentation masks of shape [num_preds, fH, fW];
                    - scores (FloatTensor): normalized prediction scores of shape [num_preds];
                    - batch_ids (LongTensor): batch indices of predictions of shape [num_preds].

        Raises:
            ValueError: Error when an invalid type of duplicate removal mechanism is provided.
        """

        # Retrieve desired items from storage dictionary
        cls_logits = storage_dict['cls_logits']
        pred_boxes = storage_dict['pred_boxes']
        mask_logits = storage_dict['mask_logits']

        # Add prediction dictionary to list of prediction dictionaries
        pred_dicts.append(pred_dict)

        return pred_dicts

    @torch.no_grad()
    def draw_segs(self, storage_dict, images_dict, pred_dicts, tgt_dict=None, vis_score_thr=0.4, id=None, **kwargs):
        """
        Draws predicted and target segmentations on the corresponding images.

        Segmentations must have a score of at least the score threshold to be drawn. Target segmentations get a default
        100% score.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following key:
                - images (Images): images structure containing the batched images of size [batch_size].

            images_dict (Dict): Dictionary with annotated images of predictions/targets.

            pred_dicts (List): List with prediction dictionaries containing as last entry:
                pred_dict (Dict): Prediction dictionary containing following keys:
                    - labels (LongTensor): predicted class indices of shape [num_preds];
                    - masks (BoolTensor): predicted segmentation masks of shape [num_preds, fH, fW];
                    - scores (FloatTensor): normalized prediction scores of shape [num_preds];
                    - batch_ids (LongTensor): batch indices of predictions of shape [num_preds].

            tgt_dict (Dict): Optional target dictionary containing at least following keys when given:
                - labels (LongTensor): target class indices of shape [num_targets];
                - masks (BoolTensor): target segmentation masks of shape [num_targets, iH, iW];
                - sizes (LongTensor): cumulative number of targets per batch entry of size [batch_size+1].

            vis_score_thr (float): Threshold indicating the minimum score for a segmentation to be drawn (default=0.4).
            id (int): Integer containing the head id (default=None).
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            images_dict (Dict): Dictionary containing additional images annotated with segmentations.
        """

        # Retrieve images from storage dictionary
        images = storage_dict['images']

        # Initialize list of draw dictionaries and list of dictionary names
        draw_dicts = []
        dict_names = []

        # Get prediction draw dictionary and dictionary name
        pred_dict = pred_dicts[-1]

        pred_scores = pred_dict['scores']
        sufficient_score = pred_scores >= vis_score_thr

        pred_labels = pred_dict['labels'][sufficient_score]
        pred_masks = pred_dict['masks'][sufficient_score]
        pred_scores = pred_scores[sufficient_score]
        pred_batch_ids = pred_dict['batch_ids'][sufficient_score]

        pred_sizes = [0] + [(pred_batch_ids == i).sum() for i in range(len(images))]
        pred_sizes = torch.tensor(pred_sizes, device=pred_scores.device).cumsum(dim=0)

        draw_dict_keys = ['labels', 'masks', 'scores', 'sizes']
        draw_dict_values = [pred_labels, pred_masks, pred_scores, pred_sizes]

        draw_dict = {k: v for k, v in zip(draw_dict_keys, draw_dict_values)}
        draw_dicts.append(draw_dict)

        dict_name = f"seg_pred_{id}" if id is not None else "seg_pred"
        dict_names.append(dict_name)

        # Get target draw dictionary and dictionary name if needed
        if tgt_dict is not None and not any('seg_tgt' in key for key in images_dict.keys()):
            tgt_labels = tgt_dict['labels']
            tgt_masks = tgt_dict['masks']
            tgt_scores = torch.ones_like(tgt_labels, dtype=torch.float)
            tgt_sizes = tgt_dict['sizes']

            draw_dict_values = [tgt_labels, tgt_masks, tgt_scores, tgt_sizes]
            draw_dict = {k: v for k, v in zip(draw_dict_keys, draw_dict_values)}

            draw_dicts.append(draw_dict)
            dict_names.append('seg_tgt')

        # Get number of images and image sizes without padding in (width, height) format
        num_images = len(images)
        img_sizes = images.size(mode='without_padding')

        # Get image sizes with padding in (height, width) format
        img_sizes_pad = images.size(mode='with_padding')
        img_sizes_pad = (img_sizes_pad[1], img_sizes_pad[0])

        # Draw 2D object detections on images and add them to images dictionary
        for dict_name, draw_dict in zip(dict_names, draw_dicts):
            sizes = draw_dict['sizes']

            for image_id, i0, i1 in zip(range(num_images), sizes[:-1], sizes[1:]):
                img_size = img_sizes[image_id]
                img_size = (img_size[1], img_size[0])

                image = images.images[image_id].clone()
                image = T.crop(image, 0, 0, *img_size)
                image = image.permute(1, 2, 0) * 255
                image = image.to(torch.uint8).cpu().numpy()

                metadata = self.metadata
                visualizer = Visualizer(image, metadata=metadata)

                if i1 > i0:
                    img_labels = draw_dict['labels'][i0:i1].cpu().numpy()
                    img_scores = draw_dict['scores'][i0:i1].cpu().numpy()

                    img_masks = draw_dict['masks'][i0:i1]
                    img_masks = T.resize(img_masks, img_sizes_pad)
                    img_masks = T.crop(img_masks, 0, 0, *img_size)
                    img_masks = img_masks.cpu().numpy()

                    instances = Instances(img_size, pred_classes=img_labels, pred_masks=img_masks, scores=img_scores)
                    visualizer.draw_instance_predictions(instances)

                annotated_image = visualizer.output.get_image()
                images_dict[f'{dict_name}_{image_id}'] = annotated_image

        return images_dict

    def forward_pred(self, in_feats, storage_dict, cum_feats_batch=None, images_dict=None, **kwargs):
        """
        Forward prediction method of the StandardRoIHead.

        Args:
            in_feats (FloatTensor): Input features of shape [num_feats, in_feat_size].

            storage_dict (Dict): Storage dictionary (possibly) requiring following keys:
                - feat_maps (List): list of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW];
                - images (Images): images structure containing the batched images of size [batch_size];
                - pred_boxes (Boxes): predicted 2D bounding boxes of size [num_feats].

            cum_feats_batch (LongTensor): Cumulative number of features per batch entry [batch_size+1] (default=None).
            images_dict (Dict): Dictionary with annotated images of predictions/targets (default=None).
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            storage_dict (Dict): Storage dictionary (possibly) containing following additional key:
                - mask_logits (FloatTensor): map with mask logits of shape [num_feats, mC, mH, mW].

            images_dict (Dict): Dictionary with (possibly) additional images annotated with 2D boxes/segmentations.
        """

        # Retrieve desired items from storage dictionary
        feat_maps = storage_dict['feat_maps']
        images = storage_dict['images']

        # Get number of features and device
        num_feats = len(in_feats)
        device = in_feats.device

        # Get cumulative number of features per batch entry if missing
        if cum_feats_batch is None:
            cum_feats_batch = torch.tensor([0, num_feats], device=device)

        # Get batch indices
        batch_size = len(cum_feats_batch) - 1
        batch_ids = torch.arange(batch_size, device=device)
        batch_ids = batch_ids.repeat_interleave(cum_feats_batch.diff())

        # Get box-related predictions
        if self.with_bbox:

            # Get predicted 2D bounding boxes
            pred_boxes = None
            storage_dict['pred_boxes'] = pred_boxes

        # Get mask-related predictions
        if self.with_mask:

            # Get predicted 2D bounding boxes
            pred_boxes = storage_dict['pred_boxes']

            # Get RoI-boxes
            pred_boxes = pred_boxes.clone().normalize(images).to_format('xyxy').boxes
            roi_boxes = torch.cat([batch_ids[:, None], pred_boxes], dim=1)

            # Get mask features
            mask_feat_maps = feat_maps[:self.mask_roi_extractor.num_inputs]
            mask_feats = self.mask_roi_extractor(mask_feat_maps, roi_boxes)

            # Get mask logits
            mask_logits = self.mask_head(mask_feats)
            storage_dict['mask_logits'] = mask_logits

            # Get segmentation predictions if needed
            if self.get_segs and not self.training:
                self.compute_segs(storage_dict=storage_dict, cum_feats_batch=cum_feats_batch, **kwargs)

            # Draw predicted and target segmentations if needed
            if images_dict is not None:
                self.draw_segs(storage_dict=storage_dict, images_dict=images_dict, **kwargs)

        return storage_dict, images_dict

    def forward_loss(self, **kwargs):
        """
        Forward loss method of the StandardRoIHead.
        """

    def forward(self, mode, **kwargs):
        """
        Forward method of the StandardRoIHead.

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
