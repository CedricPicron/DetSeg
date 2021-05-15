"""
Dense Object Discovery (DOD) head.
"""
from collections import OrderedDict
from copy import deepcopy
import math

from fvcore.nn import sigmoid_focal_loss
import torch
from torch import nn

from models.modules.convolution import BottleneckConv, ProjConv
from models.utils import get_feat_boxes
from structures.boxes import Boxes, box_iou


class DOD(nn.Module):
    """
    Class implementing the Dense Object Discovery (DOD) head.

    Attributes:
        net (nn.Sequential): DOD network computing the logits.

        ftm_metric (str): String containing feature-target metric used during feature-target matching.
        ftm_decision (str): String containing decision maker type used during feature-target matching.
        ftm_abs_threshold (float): Absolute threshold used by decision maker during feature-target matching.
        ftm_rel_threshold (int): Relative threshold used by decision maker during feature-target matching.

        loss_type (str): String containing the type of loss.
        focal_alpha (float): Alpha value of the sigmoid focal loss.
        focal_gamma (float): Gamma value of the sigmoid focal loss.
        pos_weight (float): Factor weighting the loss terms with positive targets.
        neg_weight (float): Factor weighting the loss terms with negative targets.

        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
    """

    def __init__(self, in_feat_size, net_dict, ftm_dict, loss_dict, metadata):
        """
        Initializes the DOD module.

        Arguments:
            in_feat_size (int): Integer containing the feature size of the input feature pyramid.

            net_dict (Dict): Network dictionary containing following keys:
                - feat_size (int): hidden feature size of the DOD network;
                - norm (str): string specifying the type of normalization used within the DOD network;
                - kernel_size (int): kernel size used by hidden layers of the DOD network;
                - bottle_size (int): bottleneck size used by bottleneck layers of the DOD network;
                - hidden_layers (int): number of hidden layers of the DOD network;
                - rel_preds (bool): whether or not DOD network should make relative predictions;
                - prior_prob (float): prior object probability determining the initial output layer bias value(s).

            ftm_dict (Dict): Feature-target matching dictionary containing following keys:
                - metric (str): string containing feature-target metric used during feature-target matching;
                - decision (str): string containing decision maker type used during feature-target matching;
                - abs_threshold (float): absolute threshold used by decision maker during feature-target matching;
                - rel_threshold (int): relative threshold used by decision maker during feature-target matching.

            loss_dict (Dict): Loss dictionary containing following keys:
                - type (str): string containing the type of loss;
                - focal_alpha (float): alpha value of the sigmoid focal loss;
                - focal_gamma (float): gamma value of the sigmoid focal loss;
                - pos_weight (float): factor weighting the loss terms with positive targets;
                - neg_weight (float): factor weighting the loss terms with negative targets.

            metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Initialization of the DOD network
        feat_size = net_dict['feat_size']
        norm = net_dict['norm']
        in_layer = ProjConv(in_feat_size, feat_size, norm=norm, skip=False)

        if net_dict['kernel_size'] == 1:
            hidden_layer = ProjConv(feat_size, norm=norm, skip=True)
        else:
            bottle_size = net_dict['bottle_size']
            bottleneck_kwargs = {'kernel_size': net_dict['kernel_size'], 'norm': norm, 'skip': True}
            hidden_layer = BottleneckConv(feat_size, bottle_size, **bottleneck_kwargs)

        num_hidden_layers = net_dict['hidden_layers']
        hidden_layers = nn.Sequential(*[deepcopy(hidden_layer) for _ in range(num_hidden_layers)])

        out_feat_size = 2 if net_dict['rel_preds'] else 1
        out_layer = ProjConv(feat_size, out_feat_size, norm=norm, skip=False)

        obj_prior_prob = net_dict['prior_prob']
        obj_bias_value = -(math.log((1 - obj_prior_prob) / obj_prior_prob))
        torch.nn.init.constant_(out_layer.conv.bias[0], obj_bias_value)

        if out_feat_size == 2:
            torch.nn.init.constant_(out_layer.conv.bias[1], -obj_bias_value)

        net_dict = OrderedDict([('in', in_layer), ('hidden', hidden_layers), ('out', out_layer)])
        self.net = nn.Sequential(net_dict)

        # Set feature-target matching attributes
        self.ftm_metric = ftm_dict['metric']
        self.ftm_decision = ftm_dict['decision']
        self.ftm_abs_threshold = ftm_dict['abs_threshold']
        self.ftm_rel_threshold = ftm_dict['rel_threshold']

        # Set loss-related attributes
        self.loss_type = loss_dict['type']
        self.focal_alpha = loss_dict['focal_alpha']
        self.focal_gamma = loss_dict['focal_gamma']
        self.pos_weight = loss_dict['pos_weight']
        self.neg_weight = loss_dict['neg_weight']

        # Set metadata attribute
        metadata.stuff_classes = metadata.thing_classes
        metadata.stuff_colors = metadata.thing_colors
        self.metadata = metadata

    @torch.no_grad()
    def forward_init(self, images, feat_maps, tgt_dict=None):
        """
        Forward initialization method of the DOD module.

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

    def ft_matching(self, feat_maps, tgt_boxes, pos_preds=None):
        """
        Perform feature-target matching.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].
            tgt_boxes (List): List of size [batch_size] with normalized Boxes structure of size [num_targets].
            pos_preds (BoolTensor): Mask with positive predictions of shape [batch_size, num_feats] (default=None).

        Returns:
            * If pos_preds is None:
                matched_feats (BoolTensor): Mask with matched features of shape [batch_size, num_feats].

            * If pos_preds is not None:
                pos_useful (BoolTensor): Mask with useful positives predictions of shape [batch_size, num_feats].
                tgt_found (List): List [batch_size] with masks of found targets of shape [num_targets].
                matched_feats (BoolTensor): Mask of missing targets matched features of shape [batch_size, num_feats].

        Raises:
            ValueError: Error when unknown feature-target metric is present in 'ftm_metric' attribute.
            ValueError: Error when unknown feature-target decision maker type is present in 'ftm_decision' attribute.
        """

        # Get batch size, feature boxes and concatenated targets boxes
        batch_size = len(tgt_boxes)
        feat_boxes = get_feat_boxes(feat_maps)
        tgt_boxes = Boxes.cat(tgt_boxes)

        # Return if there are no target boxes
        if len(tgt_boxes) == 0:
            num_feats = len(feat_boxes)
            tensor_kwargs = {'dtype': torch.bool, 'device': feat_maps[0].device}
            matched_feats = torch.zeros(batch_size, num_feats, **tensor_kwargs)

            if pos_preds is None:
                return matched_feats

            pos_useful = torch.zeros_like(matched_feats)
            tgt_found = [torch.zeros(0, **tensor_kwargs) for _ in range(batch_size)]

            return pos_useful, tgt_found, matched_feats

        # Get feature-target similarity matrices
        if self.ftm_metric == 'iou':
            sim_matrix = box_iou(feat_boxes, tgt_boxes)
            tgts_per_img = tgt_boxes.boxes_per_img.tolist()

        else:
            error_msg = f"Unknown feature-target metric '{self.ftm_metric}' during feature-target matching."
            raise ValueError(error_msg)

        # Get masks of matching feature-target pairs
        if self.ftm_decision == 'abs':
            ftm_mask = sim_matrix > self.ftm_abs_threshold

        elif self.ftm_decision == 'rel':
            ftm_ids = torch.argsort(sim_matrix, dim=0)[-self.ftm_rel_threshold:, :]
            ftm_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
            ftm_mask[ftm_ids, torch.arange(ftm_ids.shape[1])] = True

        else:
            error_msg = f"Unknown decision maker type '{self.ftm_decision}' during feature-target matching."
            raise ValueError(error_msg)

        ftm_masks = torch.split(ftm_mask, tgts_per_img, dim=1)
        ftm_masks = [torch.sum(ftm_mask, dim=1) > 0 for ftm_mask in ftm_masks]
        matched_feats = torch.stack(ftm_masks, dim=0)

        return matched_feats

    def forward(self, feat_maps, tgt_dict=None, mode='self', train_dict=None, **kwargs):
        """
        Forward method of the DOD module.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

            tgt_dict (Dict): Updated target dictionary with following updated keys:
                - labels (List): list of size [batch_size] with class indices of shape [num_targets];
                - boxes (List): list of size [batch_size] with normalized Boxes structure of size [num_targets].

            mode (str): Mode of forward DOD method chosen from {'self', 'pred', 'train'}.

            train_dict (Dict): Dictionary needed during 'train' mode requiring following keys:
                - logits (FloatTensor): tensor containing DOD logits of shape [batch_size, num_feats, {1, 2}];
                - pos_preds (BoolTensor): mask with positives predictions of shape [batch_size, num_feats];
                - pos_useful (BoolTensor): mask with useful positives predictions of shape [batch_size, num_feats];
                - tgt_found (List): list [batch_size] with masks of found targets of shape [num_targets].

            kwargs (Dict): Dictionary of keyword arguments, potentially containing following key:
                - visualize (bool): boolean indicating whether in visualization mode or not.

        Returns:
            * If mode is 'self' or 'train':
                loss_dict (Dict): Dictionary of different weighted loss terms used for backpropagation during training.
                analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

            * If mode is 'pred':
                analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.
                pos_preds (BoolTensor): Mask with positives predictions of shape [batch_size, num_feats].
                logits (FloatTensor): tensor containing DOD logits of shape [batch_size, num_feats, {1, 2}].

        Raises:
            ValueError: Error when unknown forward DOD mode is provided.
            RuntimeError: Error when logits have size different than 1 or 2.
            ValueError: Error when in 'train' mode, but no target dictionary was provided.
            ValueError: Error when in 'train' mode, but no training dictionary was provided.
            ValueError: Error when one of the required keys of the training dictionary is missing.
            ValueError: Error when unknown loss type is present in 'loss_type' attribute.
        """

        # Initialize empty analysis dictionary
        analysis_dict = {}

        # Check provided mode
        if mode not in ('self', 'pred', 'train'):
            error_msg = f"Mode should be chosen from {{'self', 'pred', 'train'}} , but got '{mode}'."
            raise ValueError(error_msg)

        # Get logits and mask with positive predictions if desired
        if mode in ('self', 'pred'):

            # Get logits
            logits = torch.cat([logit_map.flatten(2).permute(0, 2, 1) for logit_map in self.net(feat_maps)], dim=1)

            # Get mask with positive predictions
            if logits.shape[-1] == 1:
                pos_preds = logits[:, :, 0] > 0.5
            elif logits.shape[-1] == 2:
                pos_preds = logits[:, :, 0] > logits[:, :, 1]
            else:
                error_msg = f"Logits should have size 1 or 2, but got size {logits.shape[-1]}."
                raise RuntimeError(error_msg)

            analysis_dict['pos_preds_num'] = torch.sum(pos_preds)[None]
            analysis_dict['pos_preds_ratio'] = 100 * analysis_dict['pos_preds_num'] / torch.numel(pos_preds)

        # Return if in 'pred' mode
        if mode == 'pred':
            analysis_dict = {f'dod_{k}': v for k, v in analysis_dict.items()}

            return analysis_dict, pos_preds, logits

        # Handle case where no target dictionary is provided
        if tgt_dict is None:
            if mode == 'train':
                error_msg = "A target dictionary must be provided when in 'train' mode."
                raise ValueError(error_msg)
            else:
                return {}

        # Check whether correct training dictionary was given when in 'train' mode
        if mode == 'train':
            if train_dict is None:
                error_msg = "A training dictionary must be provided when in 'train' mode."
                raise ValueError(error_msg)

            for required_key in ('logits', 'pos_preds', 'pos_useful', 'tgt_found'):
                if required_key not in train_dict:
                    error_msg = f"The key '{required_key}' is missing from the training dictionary."
                    raise ValueError(error_msg)

        # Get useful positives, found targets and matched features of missing targets
        if mode == 'train':
            pos_preds = train_dict['pos_preds']
            pos_useful = train_dict['pos_useful']
            tgt_found = train_dict['tgt_found']

            missed_tgt_boxes = [boxes_i[~found_i] for boxes_i, found_i in zip(tgt_dict['boxes'], tgt_found)]
            matched_feats = self.ft_matching(feat_maps, missed_tgt_boxes)

        else:
            pos_useful, tgt_found, matched_feats = self.ft_matching(feat_maps, tgt_dict['boxes'], pos_preds=pos_preds)

        analysis_dict['pos_useful_num'] = torch.sum(pos_useful)[None]
        analysis_dict['pos_useful_ratio'] = 100 * analysis_dict['pos_useful_num'] / torch.numel(pos_useful)

        tgt_found = torch.cat(tgt_found, dim=0)
        analysis_dict['tgt_found_num'] = torch.sum(tgt_found)[None]
        analysis_dict['tgt_found_ratio'] = 100 * analysis_dict['tgt_found_num'] / torch.numel(tgt_found)

        analysis_dict['matched_feats_num'] = torch.sum(matched_feats)[None]
        analysis_dict['matched_feats_ratio'] = 100 * analysis_dict['matched_feats_num'] / torch.numel(matched_feats)

        # Get targets
        pos_tgts = pos_useful | matched_feats
        neg_tgts = pos_preds & ~pos_tgts

        targets = torch.full_like(pos_preds, fill_value=-1, dtype=torch.int64)
        targets[pos_tgts] = 1
        targets[neg_tgts] = 0

        analysis_dict['pos_tgts_num'] = torch.sum(pos_tgts)[None]
        analysis_dict['pos_tgts_ratio'] = 100 * analysis_dict['pos_tgts_num'] / torch.numel(pos_tgts)

        analysis_dict['neg_tgts_num'] = torch.sum(neg_tgts)[None]
        analysis_dict['neg_tgts_ratio'] = 100 * analysis_dict['neg_tgts_num'] / torch.numel(neg_tgts)

        # Recover logits from training dictionary when in 'train' mode
        if mode == 'train':
            logits = train_dict['logits']

        # Get weighted positive and negative losses
        loss_feats = targets != -1
        logits = logits[loss_feats, :]
        targets = targets[loss_feats]

        if self.loss_type == 'sigmoid_focal':
            if logits.shape[1] == 1:
                focal_targets = targets[:, None].to(logits.dtype)
            elif logits.shape[1] == 2:
                focal_targets = torch.stack([targets == 1, targets == 0], dim=1).to(logits.dtype)
            else:
                error_msg = f"Logits should have size 1 or 2, but got size {logits.shape[-1]}."
                raise RuntimeError(error_msg)

            focal_kwargs = {'alpha': self.focal_alpha, 'gamma': self.focal_gamma, 'reduction': 'none'}
            losses = sigmoid_focal_loss(logits, focal_targets, **focal_kwargs).sum(dim=1)

        else:
            error_msg = f"Unknown loss type '{self.loss_type}' during DOD loss computation."
            raise ValueError(error_msg)

        loss_dict = {}
        loss_dict['pos_loss'] = self.pos_weight * losses[targets == 1].sum(dim=0, keepdim=True)
        loss_dict['neg_loss'] = self.neg_weight * losses[targets == 0].sum(dim=0, keepdim=True)

        # Return loss and analysis dictionaries with 'dod' tags
        loss_dict = {f'dod_{k}': v for k, v in loss_dict.items()}
        analysis_dict = {f'dod_{k}': v for k, v in analysis_dict.items()}

        return loss_dict, analysis_dict
