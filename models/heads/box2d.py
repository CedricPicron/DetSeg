"""
Collection of 2D bounding box heads.
"""

from detectron2.layers import batched_nms
from detectron2.structures.instances import Instances
from detectron2.utils.visualizer import Visualizer
import torch
from torch import nn
import torchvision.transforms.functional as T

from models.build import build_model, MODELS
from structures.boxes import apply_box_deltas, Boxes, box_iou, get_box_deltas


@MODELS.register_module()
class BaseBox2dHead(nn.Module):
    """
    Class implementing the BaseBox2dHead module.

    Attributes:
        logits (nn.Module): Module computing the 2D bounding box logits.
        box_encoding (str): String containing the type of box encoding scheme.
        get_dets (bool): Boolean indicating whether to get 2D object detection predictions.

        dup_attrs (Dict): Optional dictionary specifying the duplicate removal mechanism, possibly containing:
            - type (str): string containing the type of duplicate removal mechanism (mandatory);
            - nms_candidates (int): integer containing the maximum number of candidate detections retained before NMS;
            - nms_thr (float): value of IoU threshold used during NMS to remove duplicate detections.

        max_dets (int): Optional integer with the maximum number of returned 2D object detection predictions.
        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
        matcher (nn.Module): Optional module determining the 2D target boxes.
        report_match_stats (bool): Boolean indicating whether to report matching statistics.
        loss (nn.Module): Module computing the 2D bounding box loss.
    """

    def __init__(self, logits_cfg, box_encoding, get_dets, metadata, loss_cfg, dup_attrs=None, max_dets=None,
                 matcher_cfg=None, report_match_stats=True, **kwargs):
        """
        Initializes the BaseBox2dHead module.

        Args:
            logits_cfg (Dict): Configuration dictionary specifying the logits module.
            box_encoding (str): String containing the type of box encoding scheme.
            get_dets (bool): Boolean indicating whether to get 2D object detection predictions.
            metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
            loss_cfg (Dict): Configuration dictionary specifying the loss module.
            dup_attrs (Dict): Attribute dictionary specifying the duplicate removal mechanism (default=None).
            max_dets (int): Integer with the maximum number of returned 2D object detection predictions (default=None).
            matcher_cfg (Dict): Configuration dictionary specifying the matcher module (default=None).
            report_match_stats (bool): Boolean indicating whether to report matching statistics (default=True).
            kwargs (Dict): Dictionary of unused keyword arguments.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build logits module
        self.logits = build_model(logits_cfg)

        # Build matcher module if needed
        self.matcher = build_model(matcher_cfg) if matcher_cfg is not None else None

        # Build loss module
        self.loss = build_model(loss_cfg)

        # Set remaining attributes
        self.box_encoding = box_encoding
        self.get_dets = get_dets
        self.dup_attrs = dup_attrs
        self.max_dets = max_dets
        self.metadata = metadata
        self.report_match_stats = report_match_stats

    @torch.no_grad()
    def compute_dets(self, storage_dict, pred_dicts, cum_feats_batch=None, **kwargs):
        """
        Method computing the 2D object detection predictions.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - cls_logits (FloatTensor): classification logits of shape [num_feats, num_labels];
                - pred_boxes (Boxes): predicted 2D bounding boxes of size [num_feats].

            pred_dicts (List): List of size [num_pred_dicts] collecting various prediction dictionaries.
            cum_feats_batch (LongTensor): Cumulative number of features per batch entry [batch_size+1] (default=None).
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            pred_dicts (List): List with prediction dictionaries containing following additional entry:
                pred_dict (Dict): Prediction dictionary containing following keys:
                    - labels (LongTensor): predicted class indices of shape [num_preds];
                    - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_preds];
                    - scores (FloatTensor): normalized prediction scores of shape [num_preds];
                    - batch_ids (LongTensor): batch indices of predictions of shape [num_preds].

        Raises:
            ValueError: Error when an invalid type of duplicate removal mechanism is provided.
        """

        # Retrieve classification logits and predicted 2D bounding boxes
        cls_logits = storage_dict['cls_logits']
        pred_boxes = storage_dict['pred_boxes']

        # Get number of features, number of labels and device
        num_feats, num_labels = cls_logits.size()
        device = cls_logits.device

        # Get cumulative number of features per batch entry if missing
        if cum_feats_batch is None:
            cum_feats_batch = torch.tensor([0, num_feats], device=device)

        # Get batch indices
        batch_size = len(cum_feats_batch) - 1
        batch_ids = torch.arange(batch_size, device=device)
        batch_ids = batch_ids.repeat_interleave(cum_feats_batch.diff())

        # Only keep entries with well-defined boxes
        well_defined = pred_boxes.well_defined()
        cls_logits = cls_logits[well_defined]
        pred_boxes = pred_boxes[well_defined]
        batch_ids = batch_ids[well_defined]

        # Get 2D detections for each combination of box and class label
        num_feats = len(cls_logits)
        num_classes = num_labels - 1

        pred_labels = torch.arange(num_classes, device=device)[None, :].expand(num_feats, -1).reshape(-1)
        pred_boxes.boxes = pred_boxes.boxes[:, None, :].expand(-1, num_classes, -1).reshape(-1, 4)
        pred_scores = cls_logits[:, :-1].sigmoid().view(-1)
        batch_ids = batch_ids[:, None].expand(-1, num_classes).reshape(-1)

        # Initialize prediction dictionary
        pred_keys = ('labels', 'boxes', 'scores', 'batch_ids')
        pred_dict = {pred_key: [] for pred_key in pred_keys}

        # Iterate over every batch entry
        for i in range(batch_size):

            # Get predictions corresponding to batch entry
            batch_mask = batch_ids == i

            pred_labels_i = pred_labels[batch_mask]
            pred_boxes_i = pred_boxes[batch_mask]
            pred_scores_i = pred_scores[batch_mask]

            # Remove duplicate predictions if needed
            if self.dup_attrs is not None:
                dup_removal_type = self.dup_attrs['type']

                if dup_removal_type == 'nms':
                    num_candidates = self.dup_attrs['nms_candidates']
                    candidate_ids = pred_scores_i.topk(num_candidates)[1]

                    pred_labels_i = pred_labels_i[candidate_ids]
                    pred_boxes_i = pred_boxes_i[candidate_ids].to_format('xyxy')
                    pred_scores_i = pred_scores_i[candidate_ids]

                    iou_thr = self.dup_attrs['nms_thr']
                    non_dup_ids = batched_nms(pred_boxes_i.boxes, pred_scores_i, pred_labels_i, iou_thr)

                    pred_labels_i = pred_labels_i[non_dup_ids]
                    pred_boxes_i = pred_boxes_i[non_dup_ids]
                    pred_scores_i = pred_scores_i[non_dup_ids]

                else:
                    error_msg = f"Invalid type of duplicate removal mechanism (got '{dup_removal_type}')."
                    raise ValueError(error_msg)

            # Only keep top predictions if needed
            if self.max_dets is not None:
                if len(pred_scores_i) > self.max_dets:
                    top_pred_ids = pred_scores_i.topk(self.max_dets)[1]

                    pred_labels_i = pred_labels_i[top_pred_ids]
                    pred_boxes_i = pred_boxes_i[top_pred_ids]
                    pred_scores_i = pred_scores_i[top_pred_ids]

            # Add predictions to prediction dictionary
            pred_dict['labels'].append(pred_labels_i)
            pred_dict['boxes'].append(pred_boxes_i)
            pred_dict['scores'].append(pred_scores_i)
            pred_dict['batch_ids'].append(torch.full_like(pred_labels_i, i))

        # Concatenate predictions of different batch entries
        pred_dict.update({k: torch.cat(v, dim=0) for k, v in pred_dict.items() if k not in ['boxes']})
        pred_dict['boxes'] = Boxes.cat(pred_dict['boxes'])

        # Add prediction dictionary to list of prediction dictionaries
        pred_dicts.append(pred_dict)

        return pred_dicts

    @torch.no_grad()
    def draw_dets(self, storage_dict, images_dict, pred_dicts, tgt_dict=None, vis_score_thr=0.4, id=None, **kwargs):
        """
        Draws predicted and target 2D object detections on the corresponding images.

        Boxes must have a score of at least the score threshold to be drawn. Target boxes get a default 100% score.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following key:
                - images (Images): images structure containing the batched images of size [batch_size].

            images_dict (Dict): Dictionary with annotated images of predictions/targets.

            pred_dicts (List): List with prediction dictionaries containing as last entry:
                pred_dict (Dict): Prediction dictionary containing following keys:
                    - labels (LongTensor): predicted class indices of shape [num_preds];
                    - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_preds];
                    - scores (FloatTensor): normalized prediction scores of shape [num_preds];
                    - batch_ids (LongTensor): batch indices of predictions of shape [num_preds].

            tgt_dict (Dict): Optional target dictionary containing at least following keys when given:
                - labels (LongTensor): target class indices of shape [num_targets];
                - boxes (Boxes): target 2D bounding boxes of size [num_targets];
                - sizes (LongTensor): cumulative number of targets per batch entry of size [batch_size+1].

            vis_score_thr (float): Threshold indicating the minimum score for a box to be drawn (default=0.4).
            id (int): Integer containing the head id (default=None).
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            images_dict (Dict): Dictionary containing additional images annotated with 2D object detections.
        """

        # Retrieve images from storage dictionary
        images = storage_dict['images']

        # Initialize list of draw dictionaries and list of dictionary names
        draw_dicts = []
        dict_names = []

        # Get prediction draw dictionary and dictionary name
        pred_dict = pred_dicts[-1]

        pred_boxes = pred_dict['boxes'].to_img_scale(images).to_format('xyxy')
        well_defined = pred_boxes.well_defined()

        pred_scores = pred_dict['scores'][well_defined]
        sufficient_score = pred_scores >= vis_score_thr

        pred_labels = pred_dict['labels'][well_defined][sufficient_score]
        pred_boxes = pred_boxes.boxes[well_defined][sufficient_score]
        pred_scores = pred_scores[sufficient_score]
        pred_batch_ids = pred_dict['batch_ids'][well_defined][sufficient_score]

        pred_sizes = [0] + [(pred_batch_ids == i).sum() for i in range(len(images))]
        pred_sizes = torch.tensor(pred_sizes, device=pred_scores.device).cumsum(dim=0)

        draw_dict_keys = ['labels', 'boxes', 'scores', 'sizes']
        draw_dict_values = [pred_labels, pred_boxes, pred_scores, pred_sizes]

        draw_dict = {k: v for k, v in zip(draw_dict_keys, draw_dict_values)}
        draw_dicts.append(draw_dict)

        dict_name = f"box2d_pred_{id}" if id is not None else "box2d_pred"
        dict_names.append(dict_name)

        # Get target draw dictionary and dictionary name if needed
        if tgt_dict is not None and not any('box2d_tgt' in key for key in images_dict.keys()):
            tgt_labels = tgt_dict['labels']
            tgt_boxes = tgt_dict['boxes'].to_img_scale(images).to_format('xyxy').boxes
            tgt_scores = torch.ones_like(tgt_labels, dtype=torch.float)
            tgt_sizes = tgt_dict['sizes']

            draw_dict_values = [tgt_labels, tgt_boxes, tgt_scores, tgt_sizes]
            draw_dict = {k: v for k, v in zip(draw_dict_keys, draw_dict_values)}

            draw_dicts.append(draw_dict)
            dict_names.append('box2d_tgt')

        # Get number of images and image sizes without padding in (width, height) format
        num_images = len(images)
        img_sizes = images.size(mode='without_padding')

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
                    img_boxes = draw_dict['boxes'][i0:i1].cpu().numpy()
                    img_scores = draw_dict['scores'][i0:i1].cpu().numpy()

                    instances = Instances(img_size, pred_classes=img_labels, pred_boxes=img_boxes, scores=img_scores)
                    visualizer.draw_instance_predictions(instances)

                annotated_image = visualizer.output.get_image()
                images_dict[f'{dict_name}_{image_id}'] = annotated_image

        return images_dict

    def forward_pred(self, in_feats, storage_dict, images_dict=None, **kwargs):
        """
        Forward prediction method of the BaseBox2dHead module.

        Args:
            in_feats (FloatTensor): Input features of shape [num_feats, in_feat_size].

            storage_dict (Dict): Storage dictionary possibly containing following key:
                - prior_boxes (Boxes): prior 2D bounding boxes of size [num_feats].

            images_dict (Dict): Dictionary with annotated images of predictions/targets (default=None).
            kwargs (Dict): Dictionary of keyword arguments passed to some underlying methods.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional keys:
                - box_logits (FloatTensor): 2D bounding box logits of shape [num_feats, 4];
                - pred_boxes (Boxes): predicted 2D bounding boxes of size [num_feats].

            images_dict (Dict): Dictionary containing additional images annotated with 2D object detections (if given).

        Raises:
            ValueError: Error when an invalid type of box encoding scheme is provided.
        """

        # Get 2D bounding box logits
        box_logits = self.logits(in_feats)
        storage_dict['box_logits'] = box_logits

        # Get predicted 2D bounding boxes
        if self.box_encoding == 'prior_boxes':
            prior_boxes = storage_dict['prior_boxes']
            pred_boxes = apply_box_deltas(box_logits, prior_boxes)

        else:
            error_msg = f"Invalid type of box encoding scheme (got '{self.box_encoding}')."
            raise ValueError(error_msg)

        storage_dict['pred_boxes'] = pred_boxes

        # Get 2D object detection predictions if needed
        if self.get_dets and not self.training:
            self.compute_dets(storage_dict=storage_dict, **kwargs)

        # Draw predicted and target 2D object detections if needed
        if images_dict is not None:
            self.draw_dets(storage_dict=storage_dict, images_dict=images_dict, **kwargs)

        return storage_dict, images_dict

    def forward_loss(self, storage_dict, tgt_dict, loss_dict, analysis_dict=None, id=None, **kwargs):
        """
        Forward loss method of the BaseBox2dHead module.

        Args:
            storage_dict (Dict): Storage dictionary (possibly) containing following keys (after matching):
                - box_logits (FloatTensor): 2D bounding box logits of shape [num_feats, 4];
                - prior_boxes (Boxes): prior 2D bounding boxes of size [num_feats];
                - matched_qry_ids (LongTensor): indices of matched queries of shape [num_pos_queries];
                - matched_tgt_ids (LongTensor): indices of corresponding matched targets of shape [num_pos_queries].

            tgt_dict (Dict): Target dictionary containing at least following key:
                - boxes (Boxes): target 2D bounding boxes of size [num_targets].

            loss_dict (Dict): Dictionary containing different weighted loss terms.
            analysis_dict (Dict): Dictionary containing different analyses (default=None).
            id (int): Integer containing the head id (default=None).
            kwargs (Dict): Dictionary of keyword arguments passed to some underlying modules.

        Returns:
            loss_dict (Dict): Loss dictionary containing following additional key:
                - box_loss (FloatTensor): 2D bounding box loss of shape [].

            analysis_dict (Dict): Analysis dictionary (possibly) containing following additional keys (if not None):
                - box_multi_tgt_qry (FloatTensor): percentage of queries matched to multiple targets of shape [];
                - box_matched_qry (FloatTensor): percentage of matched queries of shape [];
                - box_matched_tgt (FloatTensor): percentage of matched targets of shape [];
                - box_acc (FloatTensor): 2D bounding box accuracy of shape [].

        Raises:
            ValueError: Error when an invalid type of box encoding scheme is provided.
        """

        # Perform matching if matcher is available
        if self.matcher is not None:
            self.matcher(storage_dict=storage_dict, tgt_dict=tgt_dict, analysis_dict=analysis_dict, **kwargs)

        # Retrieve 2D bounding box logits and matching results
        box_logits = storage_dict['box_logits']
        matched_qry_ids = storage_dict['matched_qry_ids']
        matched_tgt_ids = storage_dict['matched_tgt_ids']

        # Get device
        device = box_logits.device

        # Report matching statistics if needed
        if self.report_match_stats and analysis_dict is not None:

            # Get percentage of queries matched to multiple targets
            num_qrys = len(box_logits)
            qry_counts = matched_qry_ids.unique(return_counts=True)[1]

            box_multi_tgt_qry = (qry_counts > 1).sum().item() / num_qrys if num_qrys > 0 else 0.0
            box_multi_tgt_qry = torch.tensor(box_multi_tgt_qry, device=device)

            key_name = f'box_multi_tgt_qry_{id}' if id is not None else 'box_multi_tgt_qry'
            analysis_dict[key_name] = 100 * box_multi_tgt_qry

            # Get percentage of matched queries
            box_matched_qry = len(qry_counts) / num_qrys if num_qrys > 0 else 1.0
            box_matched_qry = torch.tensor(box_matched_qry, device=device)

            key_name = f'box_matched_qry_{id}' if id is not None else 'box_matched_qry'
            analysis_dict[key_name] = 100 * box_matched_qry

            # Get percentage of matched targets
            num_tgts = len(tgt_dict['boxes'])

            box_matched_tgt = len(matched_tgt_ids.unique()) / num_tgts if num_tgts > 0 else 1.0
            box_matched_tgt = torch.tensor(box_matched_tgt, device=device)

            key_name = f'box_matched_tgt_{id}' if id is not None else 'box_matched_tgt'
            analysis_dict[key_name] = 100 * box_matched_tgt

        # Handle case where there are no positive matches
        if len(matched_qry_ids) == 0:

            # Get 2D bounding box loss
            box_loss = 0.0 * box_logits.sum()
            key_name = f'box_loss_{id}' if id is not None else 'box_loss'
            loss_dict[key_name] = box_loss

            # Get 2D bounding box accuracy if needed
            if analysis_dict is not None:
                box_acc = 1.0 if len(tgt_dict['boxes']) == 0 else 0.0
                box_acc = torch.tensor(box_acc, dtype=box_loss.dtype, device=device)

                key_name = f'box_acc_{id}' if id is not None else 'box_acc'
                analysis_dict[key_name] = 100 * box_acc

            return loss_dict, analysis_dict

        # Get 2D bounding box logits with corresponding target boxes
        box_logits = box_logits[matched_qry_ids]
        tgt_boxes = tgt_dict['boxes'][matched_tgt_ids]

        # Get 2D bounding box targets
        if self.box_encoding == 'prior_boxes':
            prior_boxes = storage_dict['prior_boxes'][matched_qry_ids]
            box_targets = get_box_deltas(prior_boxes, tgt_boxes)

        else:
            error_msg = f"Invalid type of box encoding scheme (got '{self.box_encoding}')."
            raise ValueError(error_msg)

        # Get 2D bounding box loss
        box_loss = self.loss(box_logits, box_targets)
        key_name = f'box_loss_{id}' if id is not None else 'box_loss'
        loss_dict[key_name] = box_loss

        # Get 2D bounding box accuracy if needed
        if analysis_dict is not None:
            pred_boxes = storage_dict['pred_boxes'].detach()
            pred_boxes = pred_boxes[matched_qry_ids]
            box_acc = box_iou(pred_boxes, tgt_boxes).diag().mean()

            key_name = f'box_acc_{id}' if id is not None else 'box_acc'
            analysis_dict[key_name] = 100 * box_acc

        return loss_dict, analysis_dict

    def forward(self, mode, **kwargs):
        """
        Forward method of the BaseBox2dHead module.

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
