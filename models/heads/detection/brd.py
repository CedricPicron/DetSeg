"""
Base Reinforced Detector (BRD) head.
"""
from copy import deepcopy
import math

from detectron2.structures.instances import Instances
from detectron2.utils.visualizer import Visualizer
from fvcore.nn import sigmoid_focal_loss
from scipy.optimize import linear_sum_assignment as lsa
import torch
from torch import nn
import torch.nn.functional as F

from models.functional.position import sine_pos_encodings
from models.modules.attention import LegacySelfAttn1d
from models.modules.container import Sequential
from models.modules.ffn import FFN
from models.modules.mlp import MLP
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

        inter_loss (bool): Whether to apply losses on predictions from intermediate decoder layers.
        rel_preds (bool): Whether to predict boxes relative to predicted boxes from previous layer.
        use_all_preds (bool): Whether to use all predictions from a layer during actual loss computation.
        use_lsa (bool): Whether to use linear sum assignment during loss matching.

        delta_range_xy (float): Value determining the range of object location deltas.
        delta_range_wh (float): Value determining the range of object size deltas.

        focal_alpha (float): Alpha value of the sigmoid focal loss used during classification.
        focal_gamma (float): Gamma value of the sigmoid focal loss used during classification.

        reward_weight (float): Factor weighting the action rewards.
        punish_weight (float): Factor weighting the action punishments.

        cls_rank_weight (float): Factor weighting the ranking classification loss.
        l1_rank_weight (float): Factor weighting the ranking L1 bounding box loss.
        giou_rank_weight (float): Factor weighting the ranking GIoU bounding box loss.

        cls_loss_weight (float): Factor weighting the actual classification loss.
        l1_loss_weight (float): Factor weighting the actual L1 bounding box loss.
        giou_loss_weight (float): Factor weighting the actual GIoU bounding box loss.

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
                - layers (int): number of hidden layers of the MLP head;
                - prior_cls_prob (float): prior class probability.

            loss_dict (Dict): Loss dictionary containing following keys:
                - inter_loss (bool): whether to apply losses on predictions from intermediate decoder layers;
                - rel_preds (bool): whether to predict boxes relative to predicted boxes from previous layer;
                - use_all_preds (bool): whether to use all predictions from a layer during actual loss computation;
                - use_lsa (bool): whether to use linear sum assignment during loss matching;

                - delta_range_xy (float): value determining the range of object location deltas;
                - delta_range_wh (float): value determining the range of object size deltas;

                - focal_alpha (float): alpha value of the sigmoid focal loss used during classification;
                - focal_gamma (float): gamma value of the sigmoid focal loss used during classification;

                - reward_weight (float): factor weighting the action rewards;
                - punish_weight (float): factor weighting the action punishments;

                - cls_rank_weight (float): factor weighting the ranking classification loss;
                - l1_rank_weight (float): factor weighting the ranking L1 bounding box loss;
                - giou_rank_weight (float): factor weighting the ranking GIoU bounding box loss;

                - cls_loss_weight (float): factor weighting the actual classification loss;
                - l1_loss_weight (float): factor weighting the actual L1 bounding box loss;
                - giou_loss_weight (float): factor weighting the actual GIoU bounding box loss.

            metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Initialize policy network
        self.policy = PolicyNet(feat_size, input_type='pyramid', **policy_dict)

        # Initialize decoder
        self_attn = LegacySelfAttn1d(feat_size, decoder_dict['num_heads'])
        ffn = FFN(feat_size, decoder_dict['hidden_size'])
        decoder_layer = Sequential(self_attn, ffn)

        num_decoder_layers = decoder_dict['num_layers']
        self.decoder = Sequential(*[deepcopy(decoder_layer) for _ in range(num_decoder_layers)])

        # Initialize classification and bounding box heads
        num_classes = head_dict.pop('num_classes')
        prior_cls_prob = head_dict.pop('prior_cls_prob')
        self.cls_head = MLP(feat_size, out_size=num_classes, **head_dict)
        self.box_head = MLP(feat_size, out_size=4, **head_dict)

        # Set loss-related attributes
        self.inter_loss = loss_dict['inter_loss']
        self.rel_preds = loss_dict['rel_preds']
        self.use_all_preds = loss_dict['use_all_preds']
        self.use_lsa = loss_dict['use_lsa']

        self.delta_range_xy = loss_dict['delta_range_xy']
        self.delta_range_wh = loss_dict['delta_range_wh']

        self.focal_alpha = loss_dict['focal_alpha']
        self.focal_gamma = loss_dict['focal_gamma']

        self.reward_weight = loss_dict['reward_weight']
        self.punish_weight = loss_dict['punish_weight']

        self.cls_rank_weight = loss_dict['cls_rank_weight']
        self.l1_rank_weight = loss_dict['l1_rank_weight']
        self.giou_rank_weight = loss_dict['giou_rank_weight']

        self.cls_loss_weight = loss_dict['cls_loss_weight']
        self.l1_loss_weight = loss_dict['l1_loss_weight']
        self.giou_loss_weight = loss_dict['giou_loss_weight']

        # Set metadata attribute
        self.metadata = metadata

        # Set default initial values of module parameters
        self.reset_parameters(prior_cls_prob)

    def reset_parameters(self, prior_cls_prob):
        """
        Resets module parameters to default initial values.

        Args:
            prior_cls_prob (float): Prior class probability.
        """

        bias_value = -(math.log((1 - prior_cls_prob) / prior_cls_prob))
        torch.nn.init.constant_(self.cls_head.mlp[-1][-1].bias, bias_value)
        torch.nn.init.zeros_(self.box_head.mlp[-1][-1].bias)

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
        norm_boxes = tgt_dict['boxes'].normalize(images, with_padding=True)

        # Update target dictionary
        sizes = tgt_dict['sizes']
        tgt_dict['labels'] = [tgt_dict['labels'][i0:i1] for i0, i1 in zip(sizes[:-1], sizes[1:])]
        tgt_dict['boxes'] = [norm_boxes[i0:i1] for i0, i1 in zip(sizes[:-1], sizes[1:])]

        return tgt_dict, {}, {}

    def get_loss(self, action_losses, cls_logits, pred_boxes, tgt_labels, tgt_boxes):
        """
        Get BRD loss and its corresponding analysis.

        Args:
            action_losses (List): List of size [batch_size] with initial action losses of shape [train_samples].
            cls_logits (List): List of size [batch_size] with classification logits of shape [num_preds, num_classes].
            pred_boxes (List): List of size [batch_size] with normalized Boxes structure of size [num_preds].
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
        device = tgt_labels[0].device
        tensor_kwargs = {'dtype': torch.float, 'device': device}
        loss_dict = {f'brd_{k}_loss': torch.zeros(1, **tensor_kwargs) for k in ['action', 'cls', 'l1', 'giou']}
        analysis_dict = {f'brd_{k}': torch.zeros(1, **tensor_kwargs) for k in ['cls_acc', 'num_preds']}

        # Get total number of target boxes across batch entries
        batch_size = len(cls_logits)
        num_tgts_loc = sum(len(tgt_boxes[i]) for i in range(batch_size))
        num_tgts_glob_avg = torch.tensor([num_tgts_loc], **tensor_kwargs)

        # Average number of target boxes across nodes
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_tgts_glob_avg)
            num_tgts_glob_avg = torch.clamp(num_tgts_glob_avg/get_world_size(), min=1.0)

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
                loss_dict['brd_l1_loss'] += 0.0 * pred_boxes[i].boxes.sum()
                loss_dict['brd_giou_loss'] += 0.0 * pred_boxes[i].boxes.sum()
                continue

            # Get prediction and target boxes in (center_x, center_y, width, height) format
            pred_boxes_i = pred_boxes[i].to_format('cxcywh')
            tgt_boxes_i = tgt_boxes[i].to_format('cxcywh')

            # Get action weights
            with torch.no_grad():

                # Get loss matrix
                cls_loss = -self.cls_rank_weight * torch.sigmoid(cls_logits[i])[:, tgt_labels[i]]
                l1_loss = self.l1_rank_weight * torch.cdist(pred_boxes_i.boxes, tgt_boxes_i.boxes, p=1)
                giou_loss = -self.giou_rank_weight * box_giou(pred_boxes_i, tgt_boxes_i)
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

            # Get prediction and target ids for loss computation
            if self.use_lsa:
                pred_ids, tgt_ids = lsa(loss_matrix.cpu())
                pred_ids = torch.as_tensor(pred_ids, device=device)
                tgt_ids = torch.as_tensor(tgt_ids, device=device)

            else:
                pred_ids = pred_rankings[0, :]
                tgt_ids = torch.arange(num_tgts, device=device)

            if self.use_all_preds:
                pred_ids = torch.cat([pred_ids, torch.arange(num_preds, device=device)], dim=0)
                tgt_ids = torch.cat([tgt_ids, best_tgt_ids], dim=0)

            # Get classification loss
            cls_logits_i = cls_logits[i][pred_ids, :]
            num_classes = cls_logits_i.shape[1]
            cls_targets_i = F.one_hot(tgt_labels[i][tgt_ids], num_classes).to(cls_logits_i.dtype)

            cls_kwargs = {'alpha': self.focal_alpha, 'gamma': self.focal_gamma, 'reduction': 'mean'}
            cls_loss = sigmoid_focal_loss(cls_logits_i, cls_targets_i, **cls_kwargs)
            loss_dict['brd_cls_loss'] += self.cls_loss_weight * (num_tgts/num_tgts_glob_avg) * (num_classes*cls_loss)

            # Get L1 bounding box loss
            l1_loss = F.l1_loss(pred_boxes_i.boxes[pred_ids, :], tgt_boxes_i.boxes[tgt_ids, :], reduction='mean')
            loss_dict['brd_l1_loss'] += self.l1_loss_weight * (num_tgts/num_tgts_glob_avg) * (4*l1_loss)

            # Get GIoU bounding box loss
            giou_loss = (1 - torch.diag(box_giou(pred_boxes_i[pred_ids], tgt_boxes_i[tgt_ids]))).mean()
            loss_dict['brd_giou_loss'] += self.giou_loss_weight * (num_tgts/num_tgts_glob_avg) * giou_loss

            # Perform classification accurcy analysis
            with torch.no_grad():

                # Get number of correct predictions
                matched_tgts = min(num_preds, num_tgts) if self.use_lsa else num_tgts
                pred_labels = torch.argmax(cls_logits_i[:matched_tgts], dim=1)
                num_correct_preds = torch.eq(pred_labels, tgt_labels[i][:matched_tgts]).sum(dim=0, keepdim=True)

                # Average number of correct predictions accross nodes
                if is_dist_avail_and_initialized():
                    torch.distributed.all_reduce(num_correct_preds)
                    num_correct_preds = num_correct_preds/get_world_size()

                # Get classification accuracy
                analysis_dict['brd_cls_acc'] += 100 * num_correct_preds / num_tgts_glob_avg

        return loss_dict, analysis_dict

    @staticmethod
    def make_predictions(cls_logits, pred_boxes):
        """
        Make classified bounding box predictions based on given classification and bounding box predictions.

        Args:
            cls_logits (List): List of size [batch_size] with classification logits of shape [num_preds, num_classes].
            pred_boxes (List): List of size [batch_size] with normalized Boxes structure of size [num_preds].

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

        # Get batch size
        batch_size = len(cls_logits)

        # Add predictions to prediction dictionary
        for i in range(batch_size):
            scores, labels = cls_logits[i].sigmoid().max(dim=1)
            boxes = pred_boxes[i]
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
                pred_dicts (List): List of prediction dictionaries with each dictionary containing following keys:
                    - labels (LongTensor): predicted class indices of shape [num_preds_total];
                    - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_preds_total];
                    - scores (FloatTensor): normalized prediction scores of shape [num_preds_total];
                    - batch_ids (LongTensor): batch indices of predictions of shape [num_preds_total].
        """

        # Get batch size
        batch_size = len(feat_maps[0])

        # Assume no padded regions when feature masks are missing
        if feat_masks is None:
            tensor_kwargs = {'dtype': torch.bool, 'device': feat_maps[0].device}
            feat_masks = [torch.ones(*feat_map[:, 0].shape, **tensor_kwargs) for feat_map in feat_maps]

        # Get position feature maps and position ids
        pos_feat_maps, pos_id_maps = sine_pos_encodings(feat_maps, normalize=True)

        # Get object features, position features and prior object locations before sampling
        obj_feats = torch.cat([feat_map.flatten(2).permute(0, 2, 1) for feat_map in feat_maps], dim=1)
        pos_feats = torch.cat([pos_feat_map.flatten(1).t() for pos_feat_map in pos_feat_maps], dim=0)
        prior_cxcy = torch.cat([pos_id_map.flatten(1).t() for pos_id_map in pos_id_maps], dim=0)

        # Get prior object sizes before sampling
        map_numel = torch.tensor([feat_map.flatten(2).shape[-1] for feat_map in feat_maps]).to(prior_cxcy.device)
        prior_wh = torch.tensor([[1/s for s in feat_map.shape[:1:-1]] for feat_map in feat_maps]).to(prior_cxcy)
        prior_wh = torch.repeat_interleave(prior_wh, map_numel, dim=0)

        # Get prior boxes before sampling
        prior_boxes = torch.cat([prior_cxcy, prior_wh], dim=1)

        # Apply policy to get object features with corresponding information
        if tgt_dict is not None:
            sample_masks, action_losses = self.policy(feat_maps, mode='training')
            obj_feats = [obj_feats[i][sample_masks[i]] for i in range(batch_size)]
            pos_feats = [pos_feats[sample_masks[i]] for i in range(batch_size)]
            prior_boxes = [prior_boxes[sample_masks[i]] for i in range(batch_size)]

        else:
            sample_ids = self.policy(feat_maps, mode='inference')
            obj_feats = [obj_feats[i][sample_ids[i]] for i in range(batch_size)]
            pos_feats = [pos_feats[sample_ids[i]] for i in range(batch_size)]
            prior_boxes = [prior_boxes[sample_ids[i]] for i in range(batch_size)]

        # Process object features with decoder
        decoder_output = self.decoder(obj_feats, return_intermediate=self.inter_loss, pos_feat_list=pos_feats)
        obj_feats_list = [obj_feats, *decoder_output] if self.inter_loss else [decoder_output]
        num_pred_sets = len(obj_feats_list)

        # Some preparation during trainval
        if tgt_dict is not None:
            tgt_labels = tgt_dict['labels']
            tgt_boxes = tgt_dict['boxes']

            loss_dict = {}
            analysis_dict = {}

        # Get losses with corresponding analyses or get predictions
        for set_id, obj_feats in enumerate(obj_feats_list):

            # Continue if desired
            if tgt_dict is None and set_id < num_pred_sets-1 and not self.rel_preds:
                continue

            # Get classification and bounding box logits
            cls_logits = self.cls_head(obj_feats)
            box_logits = self.box_head(obj_feats)

            # Get prediction boxes
            pred_boxes = []

            for i in range(batch_size):
                box_deltas = box_logits[i].tanh()

                pred_cxcy = self.delta_range_xy * box_deltas[:, :2] + prior_boxes[i][:, :2]
                pred_wh = self.delta_range_wh ** box_deltas[:, 2:] * prior_boxes[i][:, 2:]
                pred_boxes_i = torch.cat([pred_cxcy, pred_wh], dim=1)

                if self.rel_preds:
                    prior_boxes[i] = pred_boxes_i.detach()

                pred_boxes_i = Boxes(pred_boxes_i, format='cxcywh', normalized='img_with_padding')
                pred_boxes.append(pred_boxes_i)

            # Continue if desired
            if tgt_dict is None and set_id < num_pred_sets-1:
                continue

            # Get loss and analysis dictionaries during trainval
            if tgt_dict is not None:
                inputs = [action_losses, cls_logits, pred_boxes, tgt_labels, tgt_boxes]
                local_loss_dict, local_analysis_dict = self.get_loss(*inputs)

                if self.inter_loss:
                    loss_dict.update({f'{k}_{set_id}': v for k, v in local_loss_dict.items()})
                    analysis_dict.update({f'{k}_{set_id}': v for k, v in local_analysis_dict.items()})

                else:
                    loss_dict = local_loss_dict
                    analysis_dict = local_analysis_dict

                if set_id == num_pred_sets-1:
                    return loss_dict, analysis_dict

            # Get prediction dictionary validation/testing
            elif set_id == num_pred_sets-1:
                pred_dict = BRD.make_predictions(cls_logits, pred_boxes)
                pred_dicts = [pred_dict]

                return pred_dicts

    def visualize(self, images, pred_dicts, tgt_dict, score_treshold=0.4):
        """
        Draws predicted and target bounding boxes on given full-resolution images.

        Boxes must have a score of at least the score threshold to be drawn. Target boxes get a default 100% score.

        Args:
            images (Images): Images structure containing the batched images.

            pred_dicts (List): List of prediction dictionaries with each dictionary containing following keys:
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

        # Get keys found in draw dictionaries
        draw_dict_keys = ['labels', 'boxes', 'scores', 'sizes']

        # Get draw dictionaries for predictions
        pred_draw_dicts = []

        for pred_dict in pred_dicts:
            pred_boxes = pred_dict['boxes'].to_img_scale(images).to_format('xyxy')
            well_defined = pred_boxes.well_defined()

            pred_scores = pred_dict['scores'][well_defined]
            sufficient_score = pred_scores >= score_treshold

            pred_labels = pred_dict['labels'][well_defined][sufficient_score]
            pred_boxes = pred_boxes.boxes[well_defined][sufficient_score]
            pred_scores = pred_scores[sufficient_score]
            pred_batch_ids = pred_dict['batch_ids'][well_defined][sufficient_score]

            pred_sizes = [0] + [(pred_batch_ids == i).sum() for i in range(len(images))]
            pred_sizes = torch.tensor(pred_sizes).cumsum(dim=0).to(tgt_dict['sizes'])

            draw_dict_values = [pred_labels, pred_boxes, pred_scores, pred_sizes]
            pred_draw_dict = {k: v for k, v in zip(draw_dict_keys, draw_dict_values)}
            pred_draw_dicts.append(pred_draw_dict)

        # Get draw dictionary for targets
        tgt_labels = torch.cat(tgt_dict['labels'])
        tgt_boxes = Boxes.cat(tgt_dict['boxes']).to_img_scale(images).to_format('xyxy').boxes
        tgt_scores = torch.ones_like(tgt_labels, dtype=torch.float)
        tgt_sizes = tgt_dict['sizes']

        draw_dict_values = [tgt_labels, tgt_boxes, tgt_scores, tgt_sizes]
        tgt_draw_dict = {k: v for k, v in zip(draw_dict_keys, draw_dict_values)}

        # Combine draw dicationaries and get corresponding dictionary names
        draw_dicts = [*pred_draw_dicts, tgt_draw_dict]
        dict_names = [f'pred_{i+1}'for i in range(len(pred_dicts))] + ['tgt']

        # Get image sizes without padding in (width, height) format
        img_sizes = images.size(with_padding=False)

        # Get and convert tensor with images
        images = images.images.clone().permute(0, 2, 3, 1)
        images = (images * 255).to(torch.uint8).cpu().numpy()

        # Get number of images and initialize images dictionary
        num_images = len(images)
        images_dict = {}

        # Draw bounding boxes on images and add them to images dictionary
        for dict_name, draw_dict in zip(dict_names, draw_dicts):
            sizes = draw_dict['sizes']

            for image_id, i0, i1 in zip(range(num_images), sizes[:-1], sizes[1:]):
                visualizer = Visualizer(images[image_id], metadata=self.metadata)

                img_size = img_sizes[image_id]
                img_size = (img_size[1], img_size[0])

                img_labels = draw_dict['labels'][i0:i1].cpu().numpy()
                img_boxes = draw_dict['boxes'][i0:i1].cpu().numpy()
                img_scores = draw_dict['scores'][i0:i1].cpu().numpy()

                instances = Instances(img_size, pred_classes=img_labels, pred_boxes=img_boxes, scores=img_scores)
                visualizer.draw_instance_predictions(instances)

                annotated_image = visualizer.output.get_image()
                images_dict[f'brd_{dict_name}_{image_id}'] = annotated_image[:img_size[0], :img_size[1], :]

        return images_dict
