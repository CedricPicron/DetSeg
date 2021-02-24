"""
Base Reinforced Detector (BRD) head.
"""
from copy import deepcopy

from detectron2.structures.instances import Instances
from detectron2.utils.visualizer import Visualizer
from fvcore.nn import sigmoid_focal_loss
import torch
from torch import nn
import torch.nn.functional as F

from models.functional.position import sine_pos_encodings
from models.modules.attention import SelfAttn1d
from models.modules.mlp import FFN, MLP
from models.modules.policy import PolicyNet
from structures.boxes import Boxes, box_giou
from utils.distributed import get_world_size, is_dist_avail_and_initialized


class BRD(nn.Module):
    """
    Class implementing the Base Reinforced Detector (BRD) module.

    Attributes:
        policy (PolicyNet): Policy network computing action masks and initial action losses.
        decoder (nn.Sequential): Sequence of decoder layers, with each layer having a self-attention and FFN operation.
        cls_head (MLP): Module computing the classification logits from object features.
        box_head (MLP): Module computing the bounding box logits from object features.

        focal_alpha (float): Alpha value of the sigmoid focal loss used during classification.
        focal_gamma (float): Gamma value of the sigmoid focal loss used during classification.
        reward_weight (float): Factor weighting the action rewards.
        punish_weight (float): Factor weighting the action punishments.
        cls_weight (float): Factor weighting the classification loss.
        l1_weight (float): Factor weighting the L1 bounding box loss.
        giou_weight (float): Factor weighting the GIoU bounding box loss.

        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
    """

    def __init__(self, feat_size, policy_dict, decoder_dict, head_dict, loss_dict, metadata):
        """
        Initializes the BRD module.

        Args:
            feat_size (int): Integer containing the feature size.

            policy_dict (Dict): Policy dictionary, potentially containing following keys:
                - num_groups (int): number of groups used for group normalization;
                - prior_prob (float): prior object probability;
                - inference_samples (int): maximum number of samples during inference;
                - num_hidden_layers (int): number of hidden layers of the policy head.

            decoder_dict (Dict): Decoder dictionary containing following keys:
                - num_heads (int): number of attention heads used during the self-attention operation;
                - hidden_size (int): integer containing the hidden feature size used during the FFN operation;
                - num_layers (int): number of consecutive decoder layers.

            head_dict (Dict): Head dictionary containing following keys:
                - num_classes (int): integer containing the number of object classes (without background);
                - hidden_size (int): integer containing the hidden feature size used during the MLP operation;
                - num_hidden_layers (int): number of hidden layers of the MLP head.

            loss_dict (Dict): Loss dictionary containing following keys:
                - focal_alpha (float): alpha value of the sigmoid focal loss used during classification;
                - focal_gamma (float): gamma value of the sigmoid focal loss used during classification;
                - reward_weight (float): factor weighting the action rewards;
                - punish_weight (float): factor weighting the action punishments;
                - cls_weight (float): factor weighting the classification loss;
                - l1_weight (float): factor weighting the L1 bounding box loss;
                - giou_weight (float): factor weighting the GIoU bounding box loss.

            metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Initialize policy network
        self.policy = PolicyNet(feat_size, input_type='pyramid', **policy_dict)

        # Initialize decoder
        self_attn = SelfAttn1d(feat_size, decoder_dict['num_heads'])
        ffn = FFN(feat_size, decoder_dict['hidden_size'])
        decoder_layer = nn.Sequential(self_attn, ffn)

        num_decoder_layers = decoder_dict['num_layers']
        self.decoder = nn.Sequential(*[deepcopy(decoder_layer) for _ in range(num_decoder_layers)])

        # Initialize classification and bounding box heads
        num_classes = head_dict.pop('num_classes')
        self.cls_head = MLP(feat_size, out_size=num_classes, **head_dict)
        self.box_head = MLP(feat_size, out_size=4, **head_dict)

        # Set loss-related attributes
        self.focal_alpha = loss_dict['focal_alpha']
        self.focal_gamma = loss_dict['focal_gamma']
        self.reward_weight = loss_dict['reward_weight']
        self.punish_weight = loss_dict['punish_weight']
        self.cls_weight = loss_dict['cls_weight']
        self.l1_weight = loss_dict['l1_weight']
        self.giou_weight = loss_dict['giou_weight']

        # Set metadata attribute
        self.metadata = metadata

    def forward_init(self, images, feat_maps, tgt_dict=None):
        """
        Forward initialization method of the BRD module.

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

    def get_loss(self, action_losses, cls_logits, box_logits, tgt_labels, tgt_boxes):
        """
        Get BRD loss and its corresponding analysis.

        Args:
            action_losses (List): List of size [batch_size] with initial action losses of shape [train_samples].
            cls_logits (List): List of size [batch_size] with classification logits of shape [num_preds, num_classes].
            box_logits (List): List of size [batch_size] with bounding box logits of shape [num_preds, 4].
            tgt_labels (List): List of size [batch_size] with class indices of shape [num_targets].
            tgt_boxes (List): List of size [batch_size] with normalized Boxes structure of size [num_targets].

        Returns:
            loss_dict (Dict): Loss dictionary containing following keys:
                - brd_action_loss (FloatTensor): weighted action loss of shape [1];
                - brd_cls_loss (FloatTensor): weighted classification loss of shape [1];
                - brd_l1_loss (FloatTensor): weigthed L1 bounding box loss of shape [1];
                - brd_giou_loss (FloatTensor): weighted GIoU bounding box loss of shape [1].

            analysis_dict (Dict): Analysis dictionary containing following keys:
                - brd_cls_acc (FloatTensor): classification accuracy of shape [1];
                - brd_num_preds (FloatTensor): number of predictions of shape [1].

        Raises:
            ValueError: Raised when target boxes are not normalized.
        """

        # Initialize loss and analysis dictionaries
        tensor_kwargs = {'dtype': torch.float, 'device': tgt_labels[0].device}
        loss_dict = {f'brd_{k}_loss': torch.zeros(1, **tensor_kwargs) for k in ['action', 'cls', 'l1', 'giou']}
        analysis_dict = {f'brd_{k}': torch.zeros(1, **tensor_kwargs) for k in ['cls_acc', 'num_preds']}

        # Get total number of target boxes across batch_entries
        batch_size = len(cls_logits)
        num_tgt_boxes = sum(len(tgt_boxes[i]) for i in range(batch_size))
        num_tgt_boxes = torch.tensor([num_tgt_boxes], **tensor_kwargs)

        # Average number of target boxes across nodes
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_tgt_boxes)
            num_tgt_boxes = torch.clamp(num_tgt_boxes/get_world_size(), min=1.0)

        # Get loss and analysis for every batch entry
        for i in range(batch_size):

            # Get number of predictions and targets
            num_preds = len(cls_logits[i])
            num_tgts = len(tgt_labels[i])

            # Add number of predictions to analysis dictionary
            analysis_dict['brd_num_preds'] += num_preds / batch_size

            # Skip batch entry if needed
            if num_preds <= 1 or num_tgts == 0:
                loss_dict['brd_action_loss'] += 0.0 * action_losses[i].sum()
                loss_dict['brd_cls_loss'] += 0.0 * cls_logits[i].sum()
                loss_dict['brd_l1_loss'] += 0.0 * box_logits[i].sum()
                loss_dict['brd_giou_loss'] += 0.0 * box_logits[i].sum()
                continue

            # Prepare boxes
            pred_boxes_i = Boxes(box_logits[i].sigmoid(), format='cxcywh', normalized=True)
            tgt_boxes_i = tgt_boxes[i].to_format('cxcywh')

            # Check whether target boxes are normalized
            if not tgt_boxes_i.normalized:
                raise ValueError("Target boxes should be normalized when using BRD loss.")

            # Get action weights
            with torch.no_grad():

                # Get loss matrix
                cls_loss = -self.cls_weight * torch.sigmoid(cls_logits[i])[:, tgt_labels[i]]
                l1_loss = self.l1_weight * torch.cdist(pred_boxes_i.boxes, tgt_boxes_i.boxes, p=1)
                giou_loss = -self.giou_weight * box_giou(pred_boxes_i, tgt_boxes_i)
                loss_matrix = cls_loss + l1_loss + giou_loss

                # Get sorted loss matrix and corresponding prediction rankings
                sorted_loss_matrix, pred_rankings = torch.sort(loss_matrix, dim=0)

                # Get initial ranking matrix
                ranking_matrix = torch.empty_like(pred_rankings)
                index_matrix = torch.arange(num_preds).repeat(num_tgts, 1).t().to(pred_rankings)
                ranking_matrix.scatter_(dim=0, index=pred_rankings, src=index_matrix)

                # Update ranking matrix to avoid duplicates by taking loss values into account
                norm_loss_matrix = (loss_matrix-loss_matrix.min())/(loss_matrix.max()-loss_matrix.min())
                ranking_matrix = ranking_matrix + norm_loss_matrix

                # Get best losses per prediction
                pred_ids = torch.arange(num_preds)
                best_rankings, best_tgt_ids = torch.min(ranking_matrix, dim=1)
                best_losses = loss_matrix[pred_ids, best_tgt_ids]

                # Get baseline losses per prediction and action weights
                reward_action = self.reward_weight * 1/num_tgts * (best_losses - sorted_loss_matrix[1, best_tgt_ids])
                punish_action = self.punish_weight * 1/num_preds * (best_losses - sorted_loss_matrix[0, best_tgt_ids])
                action_weights = torch.where(best_rankings < 1, reward_action, punish_action)

            # Get weighted action loss
            action_loss = torch.sum(action_weights * action_losses[i], dim=0, keepdim=True)
            loss_dict['brd_action_loss'] += action_loss

            # Get best prediction indices per target
            best_pred_ids = pred_rankings[0, :]

            # Get classification loss
            cls_logits_i = cls_logits[i][best_pred_ids, :]
            cls_targets = F.one_hot(tgt_labels[i], cls_logits_i.shape[1]).to(cls_logits_i.dtype)

            cls_kwargs = {'alpha': self.focal_alpha, 'gamma': self.focal_gamma, 'reduction': 'sum'}
            cls_loss = sigmoid_focal_loss(cls_logits_i, cls_targets, **cls_kwargs)
            loss_dict['brd_cls_loss'] += self.cls_weight * cls_loss / num_tgt_boxes

            # Get L1 bounding box loss
            l1_loss = F.l1_loss(pred_boxes_i.boxes[best_pred_ids, :], tgt_boxes_i.boxes, reduction='sum')
            loss_dict['brd_l1_loss'] += self.l1_weight * l1_loss / num_tgt_boxes

            # Get GIoU bounding box loss
            giou_loss = (1 - torch.diag(box_giou(pred_boxes_i[best_pred_ids], tgt_boxes_i))).sum()
            loss_dict['brd_giou_loss'] += self.giou_weight * giou_loss / num_tgt_boxes

            # Perform classification accurcy analysis
            with torch.no_grad():

                # Get number of correct predictions
                pred_labels = torch.argmax(cls_logits_i, dim=1)
                num_correct_preds = torch.eq(pred_labels, tgt_labels[i]).sum(dim=0, keepdim=True)

                # Average number of correct predictions accross nodes
                if is_dist_avail_and_initialized():
                    torch.distributed.all_reduce(num_correct_preds)
                    num_correct_preds = num_correct_preds/get_world_size()

                # Get classification accuracy
                analysis_dict['brd_cls_acc'] += 100 * num_correct_preds / num_tgt_boxes

        return loss_dict, analysis_dict

    @staticmethod
    def make_predictions(cls_logits, box_logits):
        """
        Make classified bounding box predictions based on given classification and bounding box predictions.

        Args:
            cls_logits (List): List of size [batch_size] with classification logits of shape [num_preds, num_classes].
            box_logits (List): List of size [batch_size] with bounding box logits of shape [num_preds, 4].

        Returns:
            pred_dict (Dict): Prediction dictionary containing following keys:
                - labels (LongTensor): predicted class indices of shape [num_preds_total];
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_preds_total];
                - scores (FloatTensor): normalized prediction scores of shape [num_preds_total];
                - batch_ids (LongTensor): batch indices of predictions of shape [num_preds_total].
        """

        # Initialize prediction dictionary
        pred_dict = {}
        pred_dict['labels'] = []
        pred_dict['boxes'] = []
        pred_dict['scores'] = []
        pred_dict['batch_ids'] = []

        # Get batch size and general box information
        batch_size = len(cls_logits)
        box_kwargs = {'format': 'cxcywh', 'normalized': True}

        # Add predictions to prediction dictionary
        for i in range(batch_size):
            scores, labels = cls_logits[i].sigmoid().max(dim=1)
            boxes = Boxes(box_logits[i].sigmoid(), **box_kwargs)
            batch_ids = torch.full_like(labels, i)

            pred_dict['labels'].append(labels)
            pred_dict['boxes'].append(boxes)
            pred_dict['scores'].append(scores)
            pred_dict['batch_ids'].append(batch_ids)

        # Concatenate predictions across batch entries
        pred_dict.update({k: torch.cat(v, dim=0) for k, v in pred_dict.items() if k != 'boxes'})
        pred_dict['boxes'] = Boxes.cat(pred_dict['boxes'])

        return pred_dict

    def forward(self, feat_maps, feat_masks=None, tgt_dict=None, **kwargs):
        """
        Forward method of the BRD module.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].
            feat_masks (List): Optional list [num_maps] with masks of active features of shape [batch_size, fH, fW].

            tgt_dict (Dict): Optional target dictionary containing at least following keys:
                - labels (List): list of size [batch_size] with class indices of shape [num_targets];
                - boxes (List): list of size [batch_size] with normalized Boxes structure of size [num_targets].

        Returns:
            * If tgt_dict is not None (i.e. during training and validation):
                loss_dict (Dict): Loss dictionary containing following keys:
                    - brd_action_loss (FloatTensor): weighted action loss of shape [];
                    - brd_cls_loss (FloatTensor): weighted classification loss of shape [];
                    - brd_l1_loss (FloatTensor): weigthed L1 bounding box loss of shape [];
                    - brd_giou_loss (FloatTensor): weighted GIoU bounding box loss of shape [].

                analysis_dict (Dict): Analysis dictionary containing following key:
                    - brd_cls_acc (FloatTensor): classification accuracy of shape [].

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

        # Get position-augmented features
        pos_maps = sine_pos_encodings((feat_maps, feat_masks), input_type='pyramid')
        aug_maps = [feat_map+pos_map for feat_map, pos_map in zip(feat_maps, pos_maps)]
        aug_feats = torch.cat([aug_map.flatten(2).permute(0, 2, 1) for aug_map in aug_maps], dim=1)

        # Apply policy network to obtain object features
        if tgt_dict is not None:
            sample_masks, action_losses = self.policy(feat_maps, mode='training')
            obj_feats = [aug_feats[i][sample_masks[i]] for i in range(len(aug_feats))]

        else:
            sample_ids = self.policy(feat_maps, mode='inference')
            obj_feats = [aug_feats[i][sample_ids[i]] for i in range(len(aug_feats))]

        # Process object features with decoder
        obj_feats = self.decoder(obj_feats)

        # Get classification and bounding box logits
        cls_logits = self.cls_head(obj_feats)
        box_logits = self.box_head(obj_feats)

        # Get loss and analysis dictionaries during trainval
        if tgt_dict is not None:
            tgt_labels = tgt_dict['labels']
            tgt_boxes = tgt_dict['boxes']
            loss_dict, analysis_dict = self.get_loss(action_losses, cls_logits, box_logits, tgt_labels, tgt_boxes)

            return loss_dict, analysis_dict

        # Get prediction dictionary validation/testing
        else:
            pred_dict = BRD.make_predictions(cls_logits, box_logits)

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
