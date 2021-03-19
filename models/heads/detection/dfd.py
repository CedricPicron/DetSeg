"""
Duplicate-Free Detector (DFD) head.
"""
import math

from detectron2.layers import batched_nms
from detectron2.structures.instances import Instances
from detectron2.utils.visualizer import Visualizer
from fvcore.nn import sigmoid_focal_loss, smooth_l1_loss
import torch
from torch import nn
import torch.nn.functional as F

from models.functional.position import sine_pos_encodings
from models.modules.mlp import MLP
from structures.boxes import Boxes, box_iou


class DFD(nn.Module):
    """
    Class implementing the Duplicate-Free Detector (DFD) head.

    Attributes:
        dd_cls_head (MLP): Module computing the classification logits of the dense detector.
        dd_box_head (MLP): Module computing the bounding box logits of the dense detector.

        dd_delta_range_xy (float): Value determining the range of object location deltas of the dense detector.
        dd_delta_range_wh (float): Value determining the range of object size deltas of the dense detector.

        dd_focal_alpha (float): Alpha value of the sigmoid focal loss used during dense detector classification.
        dd_focal_gamma (float): Gamma value of the sigmoid focal loss used during dense detector classification.
        dd_cls_weight (float): Factor weighting the classification loss used during dense detector classification.

        dd_smooth_l1_beta (float): Beta value of the smooth L1 loss used during dense detector box prediction.
        dd_box_weight (float): Factor weighting the bounding box loss used during dense detector box prediction.

        dd_nms_candidates (int): Number of candidates retained for NMS during dense detector inference.
        dd_nms_threshold (float): Value determining the IoU threshold of NMS during dense detector inference.
        dd_max_detections (int): Maximum number of detections retained during dense detector inference.

        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
    """

    def __init__(self, feat_size, num_classes, dd_dict, metadata):
        """
        Initializes the DFD module.

        Args:
            feat_size (int): Integer containing the feature size.
            num_classes (int): Integer containing the number of object classes (without background).

            dd_dict (Dict): Dense detector dictionary containing following keys:
                - hidden_size (int): integer containing the hidden feature size used during the MLP operation;
                - num_hidden_layers (int): number of hidden layers of the MLP classification and bounding box heads;
                - prior_cls_prob (float): prior class probability;

                - delta_range_xy (float): value determining the range of object location deltas;
                - delta_range_wh (float): value determining the range of object size deltas;

                - focal_alpha (float): alpha value of the sigmoid focal loss used during classification;
                - focal_gamma (float): gamma value of the sigmoid focal loss used during classification;
                - cls_weight (float): factor weighting the classification loss;

                - smooth_l1_beta (float): beta value of the smooth L1 loss used during box prediction;
                - box_weight (float): factor weighting the bounding box loss;

                - nms_candidates (int): number of candidates retained for NMS during inference;
                - nms_threshold (float): value determining the IoU threshold of NMS during inference;
                - max_detections (int): maximum number of detections retained during inference.

            metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Initialization of dense detector
        self.dd_cls_head = MLP(feat_size, out_size=num_classes, **dd_dict)
        self.dd_box_head = MLP(feat_size, out_size=4, **dd_dict)

        prior_cls_prob = dd_dict['prior_cls_prob']
        bias_value = -(math.log((1 - prior_cls_prob) / prior_cls_prob))
        torch.nn.init.constant_(self.dd_cls_head.head[-1][-1].bias, bias_value)
        torch.nn.init.zeros_(self.dd_box_head.head[-1][-1].bias)

        self.dd_delta_range_xy = dd_dict['delta_range_xy']
        self.dd_delta_range_wh = dd_dict['delta_range_wh']

        self.dd_focal_alpha = dd_dict['focal_alpha']
        self.dd_focal_gamma = dd_dict['focal_gamma']
        self.dd_cls_weight = dd_dict['cls_weight']

        self.dd_smooth_l1_beta = dd_dict['smooth_l1_beta']
        self.dd_box_weight = dd_dict['box_weight']

        self.dd_nms_candidates = dd_dict['nms_candidates']
        self.dd_nms_threshold = dd_dict['nms_threshold']
        self.dd_max_detections = dd_dict['max_detections']

        # Set metadata attribute
        self.metadata = metadata

    @torch.no_grad()
    def forward_init(self, images, feat_maps, tgt_dict=None):
        """
        Forward initialization method of the DFD module.

        Args:
            images (Images): Images structure containing the batched images.
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

            tgt_dict (Dict): Optional target dictionary used during trainval containing at least following keys:
                - labels (LongTensor): tensor of shape [num_targets_total] containing the class indices;
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_targets_total];
                - sizes (LongTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries.

        Returns:
            * If tgt_dict is not None (i.e. during training and validation):
                tgt_dict (Dict): Updated target dictionary with following updated keys:
                    - labels (List): list of size [batch_size] with class indices of shape [num_targets];
                    - boxes (List): list of size [batch_size] with normalized Boxes structure of size [num_targets].

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

        # Get normalized bounding boxes
        norm_boxes = tgt_dict['boxes'].normalize(images)

        # Update target dictionary
        sizes = tgt_dict['sizes']
        tgt_dict['labels'] = [tgt_dict['labels'][i0:i1] for i0, i1 in zip(sizes[:-1], sizes[1:])]
        tgt_dict['boxes'] = [norm_boxes[i0:i1] for i0, i1 in zip(sizes[:-1], sizes[1:])]

        return tgt_dict, {}, {}

    def forward(self, feat_maps, feat_masks=None, tgt_dict=None, **kwargs):
        """
        Forward method of the DFD module.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].
            feat_masks (List): Optional list [num_maps] with masks of active features of shape [batch_size, fH, fW].

            tgt_dict (Dict): Optional target dictionary containing at least following keys:
                - labels (List): list of size [batch_size] with class indices of shape [num_targets];
                - boxes (List): list of size [batch_size] with normalized Boxes structure of size [num_targets].

        Returns:
            * If tgt_dict is not None (i.e. during training and validation):
                loss_dict (Dict): Loss dictionary containing following keys:
                    - dfd_dd_cls_loss (FloatTensor): weighted dense detector classification loss of shape [1];
                    - dfd_dd_box_loss (FloatTensor): weighted dense detector bounding box loss of shape [1].

                analysis_dict (Dict): Analysis dictionary containing following keys:
                    - dfd_dd_cls_acc (FloatTensor): dense detector classification accuracy of shape [1];
                    - dfd_dd_box_acc (FloatTensor): dense detector bounding box accuracy of shape [1].

            * If tgt_dict is None (i.e. during testing and possibly during validation):
                pred_dict (Dict): Prediction dictionary containing following keys:
                    - labels (LongTensor): predicted class indices of shape [num_preds_total];
                    - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_preds_total];
                    - scores (FloatTensor): normalized prediction scores of shape [num_preds_total];
                    - batch_ids (LongTensor): batch indices of predictions of shape [num_preds_total].
        """

        # Assume no padded regions when feature masks are missing
        if feat_masks is None:
            tensor_kwargs = {'dtype': torch.bool, 'device': feat_maps[0].device}
            feat_masks = [torch.ones(*feat_map[:, 0].shape, **tensor_kwargs) for feat_map in feat_maps]

        # Get position feature maps and position ids
        pos_feat_maps, pos_id_maps = sine_pos_encodings((feat_maps, feat_masks), input_type='pyramid', normalize=True)

        # Get set of object and position features
        obj_feats = torch.cat([feat_map.flatten(2).permute(0, 2, 1) for feat_map in feat_maps], dim=1)
        _ = torch.cat([pos_feat_map.flatten(2).permute(0, 2, 1) for pos_feat_map in pos_feat_maps], dim=1)

        # Get prior object locations
        prior_cxcy = torch.cat([pos_id_map.flatten(2).permute(0, 2, 1) for pos_id_map in pos_id_maps], dim=1)

        # Get prior object sizes
        num_maps = len(feat_maps)
        prior_w = [(1/feat_mask[:, 0, :].sum(dim=1)).repeat(*feat_mask.shape[1:], 1) for feat_mask in feat_masks]
        prior_h = [(1/feat_mask[:, :, 0].sum(dim=1)).repeat(*feat_mask.shape[1:], 1) for feat_mask in feat_masks]
        prior_wh = [torch.stack([prior_w[i], prior_h[i]], dim=3) for i in range(num_maps)]
        prior_wh = torch.cat([prior_wh[i].flatten(0, 1).permute(1, 0, 2) for i in range(num_maps)], dim=1)

        # Get dense detector classification and bounding box logits
        cls_logits = self.dd_cls_head([obj_feats])[0]
        box_logits = self.dd_box_head([obj_feats])[0]

        # Get dense detector predicted boxes
        box_deltas = box_logits.tanh()
        pred_cxcy = prior_cxcy + self.dd_delta_range_xy * box_deltas[:, :, :2]
        pred_wh = prior_wh * self.dd_delta_range_wh ** box_deltas[:, :, 2:]
        pred_boxes = torch.cat([pred_cxcy, pred_wh], dim=2)

        # Get dense detector loss and analysis dictionaries (trainval only)
        if tgt_dict is not None:

            # Some preparation
            batch_size = len(cls_logits)
            num_classes = cls_logits.shape[-1]

            tensor_kwargs = {'dtype': torch.float, 'device': cls_logits.device}
            loss_dict = {f'dfd_{k}_loss': torch.zeros(1, **tensor_kwargs) for k in ['dd_cls', 'dd_box']}
            analysis_dict = {f'dfd_{k}_acc': torch.zeros(1, **tensor_kwargs) for k in ['dd_cls', 'dd_box']}

            # Get loss and analysis for each batch entry
            for i in range(batch_size):

                # Get target labels
                tgt_labels = tgt_dict['labels'][i]

                # Skip batch entry if there are no targets
                if len(tgt_labels) == 0:
                    continue

                # Prepare predicted and target boxes
                pred_boxes_i = Boxes(pred_boxes[i], format='cxcywh', normalized=True)
                tgt_boxes_i = tgt_dict['boxes'][i].to_format('cxcywh')

                # Get prediction weights and indices of best target per prediction
                with torch.no_grad():
                    iou_matrix, _ = box_iou(pred_boxes_i, tgt_boxes_i)
                    pred_weights, tgt_ids = torch.max(iou_matrix, dim=1)

                # Get classification loss
                cls_logits_i = cls_logits[i]
                cls_targets = F.one_hot(tgt_labels[tgt_ids], num_classes).to(cls_logits_i.dtype)

                cls_kwargs = {'alpha': self.dd_focal_alpha, 'gamma': self.dd_focal_gamma, 'reduction': 'none'}
                cls_losses = sigmoid_focal_loss(cls_logits_i, cls_targets, **cls_kwargs)

                cls_losses = pred_weights * cls_losses.sum(dim=1)
                cls_loss = self.dd_cls_weight * cls_losses.sum(dim=0, keepdim=True)
                loss_dict['dfd_dd_cls_loss'] += cls_loss / batch_size

                # Get bounding box loss
                box_kwargs = {'beta': self.dd_smooth_l1_beta, 'reduction': 'none'}
                box_losses = smooth_l1_loss(pred_boxes_i.boxes, tgt_boxes_i.boxes[tgt_ids], **box_kwargs)

                box_losses = pred_weights * box_losses.sum(dim=1)
                box_loss = self.dd_box_weight * box_losses.sum(dim=0, keepdim=True)
                loss_dict['dfd_dd_box_loss'] += box_loss / batch_size

                # Get classification and box accuracy
                with torch.no_grad():

                    # Get box accuracy and indices of best prediction per target
                    box_accuracies, pred_ids = torch.max(iou_matrix, dim=0)
                    box_accuracy = box_accuracies.mean(dim=0, keepdim=True)
                    analysis_dict['dfd_dd_box_acc'] += 100 * box_accuracy / batch_size

                    # Get classification accuracy
                    pred_labels = torch.argmax(cls_logits_i[pred_ids], dim=1)
                    cls_accuracy = torch.eq(pred_labels, tgt_labels).sum(dim=0, keepdim=True) / len(tgt_labels)
                    analysis_dict['dfd_dd_cls_acc'] += 100 * cls_accuracy / batch_size

            return loss_dict, analysis_dict

        # Get dense detector predictions (validation/testing)
        if tgt_dict is None:

            # Some preparation
            batch_size = len(cls_logits)
            pred_dict = {k: [] for k in ['labels', 'boxes', 'scores', 'batch_ids']}

            # Get predictions for every batch entry
            for i in range(batch_size):

                # Get predicted labels and corresponding scores
                scores, labels = cls_logits[i].max(dim=1)
                scores = scores.sigmoid_()

                # Only keep best candidates for NMS
                scores, sort_ids = scores.sort(descending=True)
                candidate_ids = sort_ids[:self.dd_nms_candidates]

                labels = labels[candidate_ids]
                scores = scores[candidate_ids]

                # Perform NMS
                boxes = Boxes(pred_boxes[i][candidate_ids], format='cxcywh', normalized=True).to_format('xyxy')
                keep_ids = batched_nms(boxes.boxes, scores, labels, iou_threshold=self.dd_nms_threshold)
                detection_ids = keep_ids[:self.dd_max_detections]

                # Add final predictions to their corresponding lists
                pred_dict['labels'].append(labels[detection_ids])
                pred_dict['boxes'].append(boxes[detection_ids])
                pred_dict['scores'].append(scores[detection_ids])
                pred_dict['batch_ids'].append(torch.full_like(detection_ids, i, dtype=torch.int64))

            # Concatenate different batch entry predictions
            pred_dict.update({k: torch.cat(v, dim=0) for k, v in pred_dict.items() if k != 'boxes'})
            pred_dict['boxes'] = Boxes.cat(pred_dict['boxes'])

            return pred_dict

    def visualize(self, images, pred_dict, tgt_dict, score_treshold=0.4):
        """
        Draws predicted and target bounding boxes on given full-resolution images.

        Boxes must have a score of at least the score threshold to be drawn. Target boxes get a default 100% score.

        Args:
            images (Images): Images structure containing the batched images.

            pred_dict (Dict): Prediction dictionary containing at least following keys:
                - labels (LongTensor): predicted class indices of shape [num_preds_total];
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_preds_total];
                - scores (FloatTensor): normalized prediction scores of shape [num_preds_total];
                - batch_ids (LongTensor): batch indices of predictions of shape [num_preds_total].

            tgt_dict (Dict): Target dictionary containing at least following keys:
                - labels (List): list of size [batch_size] with class indices of shape [num_targets];
                - boxes (List): list of size [batch_size] with normalized Boxes structure of size [num_targets];
                - sizes (LongTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries.

            score_threshold (float): Threshold indicating the minimum score for a box to be drawn (default=0.4).

        Returns:
            images_dict (Dict): Dictionary of images with drawn predicted and target bounding boxes.
        """

        # Prepare predictions
        pred_boxes = pred_dict['boxes'].to_img_scale(images).to_format('xyxy')
        well_defined = pred_boxes.well_defined()
        pred_scores = pred_dict['scores'][well_defined]

        pred_labels = pred_dict['labels'][well_defined][pred_scores >= score_treshold]
        pred_boxes = pred_boxes.boxes[well_defined][pred_scores >= score_treshold]
        pred_batch_ids = pred_dict['batch_ids'][well_defined][pred_scores >= score_treshold]
        pred_scores = pred_scores[pred_scores >= score_treshold]

        # Prepare targets
        tgt_labels = torch.cat(tgt_dict['labels'])
        tgt_boxes = Boxes.cat(tgt_dict['boxes']).to_img_scale(images).to_format('xyxy').boxes

        # Concatenate predictions and targets
        labels = torch.cat([pred_labels, tgt_labels])
        boxes = torch.cat([pred_boxes, tgt_boxes])
        scores = torch.cat([pred_scores, torch.ones_like(tgt_labels, dtype=torch.float)])

        pred_sizes = [0] + [(pred_batch_ids == i).sum() for i in range(len(images))]
        pred_sizes = torch.tensor(pred_sizes).cumsum(dim=0).to(tgt_dict['sizes'])
        sizes = torch.cat([pred_sizes, pred_sizes[-1] + tgt_dict['sizes'][1:]])

        # Get image sizes without padding in (width, height) format
        img_sizes = images.size(with_padding=False)

        # Get and convert tensor with images
        images = images.images.clone().permute(0, 2, 3, 1)
        images = (images * 255).to(torch.uint8).cpu().numpy()

        # Get number of images and initialize images dictionary
        num_images = len(images)
        images_dict = {}

        # Draw bounding boxes on images and add them to images dictionary
        for i, i0, i1 in zip(range(2*num_images), sizes[:-1], sizes[1:]):
            image_id = i % num_images
            visualizer = Visualizer(images[image_id], metadata=self.metadata)

            img_size = img_sizes[image_id]
            img_size = (img_size[1], img_size[0])

            img_labels = labels[i0:i1].cpu().numpy()
            img_boxes = boxes[i0:i1].cpu().numpy()
            img_scores = scores[i0:i1].cpu().numpy()

            instances = Instances(img_size, pred_classes=img_labels, pred_boxes=img_boxes, scores=img_scores)
            visualizer.draw_instance_predictions(instances)

            annotated_image = visualizer.output.get_image()
            key = f'pred_{image_id}' if (i // num_images) == 0 else f'tgt_{image_id}'
            images_dict[f'ret_det_{key}'] = annotated_image[:img_size[0], :img_size[1], :]

        return images_dict
