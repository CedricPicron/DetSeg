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
from models.modules.loss import SigmoidHillLoss
from structures.boxes import Boxes, box_iou, get_feat_boxes


class DOD(nn.Module):
    """
    Class implementing the Dense Object Discovery (DOD) head.

    Attributes:
        net (nn.Sequential): DOD network computing the logits.
        hill_loss (SigmoidHillLoss): Module computing the sigmoid hill losses from logits.

        pos_pred (float): Threshold determining positive predictions.
        neg_pred (float): Threshold determining negative predictions.

        tgt_metric (str): String containing the feature-target matching metric.
        tgt_decision (str): String containing the target decision maker type.
        abs_pos_tgt (float): Absolute threshold used during positive target decision making.
        abs_neg_tgt (float): Absolute threshold used during negative target decision making.
        rel_pos_tgt (int): Relative threshold used during positive target decision making.
        rel_neg_tgt (int): Relative threshold used during negative target decision making.
        static_tgt (bool): Boolean indicating whether targets are static, i.e. independent of predictions.

        loss_type (str): String containing the type of loss.
        focal_alpha (float): Alpha value of the sigmoid focal loss.
        focal_gamma (float): Gamma value of the sigmoid focal loss.
        pos_weight (float): Factor weighting the loss terms with positive targets.
        neg_weight (float): Factor weighting the loss terms with negative targets.
        hill_weight (float): Factor weighting the sigmoid hill loss terms.

        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
    """

    def __init__(self, in_feat_size, net_dict, pred_dict, tgt_dict, loss_dict, metadata):
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

            pred_dict (Dict): Dictionary with items used during prediction containing following keys:
                - pos_pred (float): threshold determining positive predictions;
                - neg_pred (float): threshold determining negative predictions.

            tgt_dict (Dict): Dictionary with items used during target computation containing following keys:
                - metric (str): string containing the feature-target matching metric;
                - decision (str): string containing the target decision maker type;
                - abs_pos_tgt (float): absolute threshold used during positive target decision making;
                - abs_neg_tgt (float): absolute threshold used during negative target decision making;
                - rel_pos_tgt (int): relative threshold used during positive target decision making;
                - rel_neg_tgt (int): relative threshold used during negative target decision making;
                - static_tgt (bool): boolean indicating whether targets are static, i.e. independent of predictions.

            loss_dict (Dict): Loss dictionary containing following keys:
                - type (str): string containing the type of loss;
                - focal_alpha (float): alpha value of the sigmoid focal loss;
                - focal_gamma (float): gamma value of the sigmoid focal loss;
                - pos_weight (float): factor weighting the loss terms with positive targets;
                - neg_weight (float): factor weighting the loss terms with negative targets;
                - hill_weight (float): factor weighting the sigmoid hill loss terms.

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

        # Initialization of the sigmoid hill loss
        self.hill_loss = SigmoidHillLoss()

        # Set prediction-related attributes
        self.pos_pred = pred_dict['pos_pred']
        self.neg_pred = pred_dict['neg_pred']

        # Set target-related attributes
        self.tgt_metric = tgt_dict['metric']
        self.tgt_decision = tgt_dict['decision']
        self.abs_pos_tgt = tgt_dict['abs_pos_tgt']
        self.abs_neg_tgt = tgt_dict['abs_neg_tgt']
        self.rel_pos_tgt = tgt_dict['rel_pos_tgt']
        self.rel_neg_tgt = tgt_dict['rel_neg_tgt']
        self.static_tgt = tgt_dict['static_tgt']

        # Set loss-related attributes
        self.loss_type = loss_dict['type']
        self.focal_alpha = loss_dict['focal_alpha']
        self.focal_gamma = loss_dict['focal_gamma']
        self.pos_weight = loss_dict['pos_weight']
        self.neg_weight = loss_dict['neg_weight']
        self.hill_weight = loss_dict['hill_weight']

        # Set metadata attribute
        metadata.stuff_classes = metadata.thing_classes
        metadata.stuff_colors = metadata.thing_colors
        self.metadata = metadata

    @torch.no_grad()
    def get_target_masks(self, feat_maps, tgt_boxes, neg_preds, mode='self', pos_preds=None, tgt_found=None):
        """
        Get positive and negative target masks.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].
            tgt_boxes (List): List of size [batch_size] with normalized Boxes structure of size [num_targets].
            neg_preds (BoolTensor): Mask with negative predictions of shape [batch_size, num_feats].
            mode (str): Mode of DOD forward method, with modes {'self', 'train'} allowed (default='self').
            pos_preds (BoolTensor): mask with positive predictions of shape [batch_size, num_feats] (default=None).
            tgt_found (List): list [batch_size] with masks of found targets of shape [num_targets] (default=None).

        Returns:
            * If mode is 'self':
                pos_useful (BoolTensor): Mask with useful positives predictions of shape [batch_size, num_feats].
                tgt_found (List): List [batch_size] with masks of found targets of shape [num_targets].
                pos_tgts (BoolTensor): Mask with positive targets of shape [batch_size, num_feats].
                neg_tgts (BoolTensor): Mask with negative targets of shape [batch_size, num_feats].

            * If mode is 'train':
                pos_tgts (BoolTensor): Mask with positive targets of shape [batch_size, num_feats].
                neg_tgts (BoolTensor): Mask with negative targets of shape [batch_size, num_feats].

        Raises:
            ValueError: Error when positive predictions mask is not provided while in 'self' mode.
            ValueError: Error when found targets mask is not provided while in 'train' mode.
            ValueError: Error when invalid forward DOD mode is provided.
            ValueError: Error when unknown feature-target metric is present in 'tgt_metric' attribute.
            ValueError: Error when unknown target decision maker type is present in 'tgt_decision' attribute.
        """

        # Check inputs
        if mode == 'self':
            if pos_preds is None:
                error_msg = "The positive predictions mask should be provided when in 'self' mode, but got None."
                raise ValueError(error_msg)

        elif mode == 'train':
            if tgt_found is None:
                error_msg = "The found targets mask should be provided when in 'train' mode, but got None."
                raise ValueError(error_msg)

        else:
            error_msg = f"Mode should be chosen from {{'self', 'train'}} , but got '{mode}'."
            raise ValueError(error_msg)

        # Get batch size, feature boxes and concatenated targets boxes
        batch_size = len(tgt_boxes)
        feat_boxes = get_feat_boxes(feat_maps)
        tgt_boxes = Boxes.cat(tgt_boxes)

        # Return if there are no target boxes
        if len(tgt_boxes) == 0:
            pos_tgts = torch.zeros_like(neg_preds)
            neg_tgts = ~neg_preds

            if mode == 'train':
                return pos_tgts, neg_tgts

            pos_useful = torch.zeros_like(neg_preds)
            tgt_found = [torch.zeros(0).to(neg_preds) for _ in range(batch_size)]

            return pos_useful, tgt_found, pos_tgts, neg_tgts

        # Get feature-target similarity matrices
        if self.tgt_metric == 'iou':
            sim_matrix = box_iou(feat_boxes, tgt_boxes)
            tgts_per_img = tgt_boxes.boxes_per_img.tolist()

        else:
            error_msg = f"Unknown feature-target metric '{self.tgt_metric}'."
            raise ValueError(error_msg)

        # Get static positive and negative target masks
        if self.tgt_decision == 'abs':
            pos_mask = sim_matrix >= self.abs_pos_tgt
            non_neg_mask = sim_matrix >= self.abs_neg_tgt

        elif self.tgt_decision == 'rel':
            sorted_ids = torch.argsort(sim_matrix, dim=0)
            num_tgts = sorted_ids.shape[1]

            pos_ids = sorted_ids[-self.rel_pos_tgt:, :]
            pos_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
            pos_mask[pos_ids, torch.arange(num_tgts)] = True

            non_neg_ids = sorted_ids[-self.rel_neg_tgt:, :]
            non_neg_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
            non_neg_mask[non_neg_ids, torch.arange(num_tgts)] = True

        else:
            error_msg = f"Unknown target decision maker type '{self.tgt_decision}'."
            raise ValueError(error_msg)

        pos_masks = torch.split(pos_mask, tgts_per_img, dim=1)
        pos_tgts = torch.stack([torch.sum(pos_mask, dim=1) > 0 for pos_mask in pos_masks], dim=0)

        non_neg_masks = torch.split(non_neg_mask, tgts_per_img, dim=1)
        neg_tgts = torch.stack([torch.sum(non_neg_mask, dim=1) == 0 for non_neg_mask in non_neg_masks], dim=0)

        # Get useful positives and found targets if in 'self' mode
        if mode == 'self':
            pos_useful = pos_preds & pos_tgts
            tgt_found = [(pos_masks[i] & pos_preds[i, :, None]).sum(dim=0) > 0 for i in range(batch_size)]

        # Get dynamic positive and negative target masks if desired
        if not self.static_tgt:
            pos_masks = [pos_masks[i][:, ~tgt_found[i]] for i in range(batch_size)]
            pos_tgts = torch.stack([torch.sum(pos_mask, dim=1) > 0 for pos_mask in pos_masks], dim=0)
            neg_tgts = ~neg_preds & neg_tgts

        # Return desired items depending on mode
        if mode == 'self':
            return pos_useful, tgt_found, pos_tgts, neg_tgts
        else:
            return pos_tgts, neg_tgts

    def forward(self, feat_maps, tgt_dict=None, images=None, mode='self', train_dict=None, **kwargs):
        """
        Forward method of the DOD module.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

            tgt_dict (Dict): Optional target dictionary used during trainval containing at least following keys:
                - labels (LongTensor): tensor of shape [num_targets_total] containing the class indices;
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_targets_total];
                - sizes (LongTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries.

            images (Images): Images structure containing the batched images (default=None).
            mode (str): Mode of forward DOD method chosen from {'pred', 'self', 'train'}.

            train_dict (Dict): Dictionary needed during 'train' mode requiring following keys:
                - logits (FloatTensor): tensor containing DOD logits of shape [batch_size, num_feats, {1, 2}];
                - pos_preds (BoolTensor): mask with positive predictions of shape [batch_size, num_feats];
                - neg_preds (BoolTensor): mask with negative predictions of shape [batch_size, num_feats];
                - pos_useful (BoolTensor): mask with useful positives predictions of shape [batch_size, num_feats];
                - tgt_found (List): list [batch_size] with masks of found targets of shape [num_targets].

            kwargs (Dict): Dictionary of keyword arguments not used by this head module.

        Returns:
            * If mode is 'pred':
                logits (FloatTensor): tensor containing DOD logits of shape [batch_size, num_feats, {1, 2}].
                pos_preds (BoolTensor): Mask with positive predictions of shape [batch_size, num_feats].
                neg_preds (BoolTensor): Mask with negative predictions of shape [batch_size, num_feats].
                analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

            * If mode is 'self' and tgt_dict is None:
                pred_dicts (List): Empty list.
                analysis_dict (Dict):  Dictionary of different analyses used for logging purposes only.

            * If mode is 'self' or 'train' and tgt_dict is not None:
                * If mode is 'self' and module in evaluation mode:
                     pred_dicts (List): Empty list.

                loss_dict (Dict): Dictionary of different weighted loss terms used for backpropagation during training.
                analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

        Raises:
            ValueError: Error when unknown forward DOD mode is provided.
            RuntimeError: Error when logits have size different than 1 or 2.
            ValueError: Error when in 'train' mode, but no target dictionary was provided.
            ValueError: Error when a target dictionary is provided without a corresponding Images structure.
            ValueError: Error when in 'train' mode, but no training dictionary was provided.
            ValueError: Error when one of the required keys of the training dictionary is missing.
            ValueError: Error when unknown loss type is present in 'loss_type' attribute.
        """

        # Get batch size and initialize empty analysis dictionary
        batch_size = len(feat_maps[0])
        analysis_dict = {}

        # Check provided mode
        if mode not in ('pred', 'self', 'train'):
            error_msg = f"Mode should be chosen from {{'pred', 'self', 'train'}} , but got '{mode}'."
            raise ValueError(error_msg)

        # Make predictions
        if mode in ('pred', 'self'):

            # Get logits
            logits = torch.cat([logit_map.flatten(2).permute(0, 2, 1) for logit_map in self.net(feat_maps)], dim=1)

            # Get positive and negative prediction masks
            if logits.shape[-1] == 1:
                obj_probs = torch.sigmoid(logits)[:, :, 0]
            elif logits.shape[-1] == 2:
                obj_probs = torch.softmax(logits, dim=2)[:, :, 0]
            else:
                error_msg = f"Logits should have size 1 or 2, but got size {logits.shape[-1]}."
                raise RuntimeError(error_msg)

            pos_preds = obj_probs >= self.pos_pred
            neg_preds = obj_probs < self.neg_pred

            analysis_dict['pos_preds'] = torch.sum(pos_preds)[None] / batch_size
            analysis_dict['neg_preds'] = torch.sum(neg_preds)[None] / batch_size

        # Return if in 'pred' mode
        if mode == 'pred':
            analysis_dict = {f'dod_{k}': v for k, v in analysis_dict.items()}
            return logits, pos_preds, neg_preds, analysis_dict

        # Handle case where no target dictionary is provided
        if tgt_dict is None:
            if mode == 'train':
                error_msg = "A target dictionary must be provided when in 'train' mode."
                raise ValueError(error_msg)
            else:
                pred_dicts = []
                return pred_dicts, analysis_dict

        # Get target boxes in desired format
        if images is not None:
            sizes = tgt_dict['sizes']
            tgt_boxes = tgt_dict['boxes'].normalize(images, with_padding=True)
            tgt_boxes = [tgt_boxes[i0:i1] for i0, i1 in zip(sizes[:-1], sizes[1:])]
        else:
            error_msg = "A corresponding Images structure must be provided along with the target dictionary."
            raise ValueError(error_msg)

        # Check whether correct training dictionary was given when in 'train' mode
        if mode == 'train':
            if train_dict is None:
                error_msg = "A training dictionary must be provided when in 'train' mode."
                raise ValueError(error_msg)

            for required_key in ('logits', 'pos_preds', 'neg_preds', 'pos_useful', 'tgt_found'):
                if required_key not in train_dict:
                    error_msg = f"The key '{required_key}' is missing from the training dictionary."
                    raise ValueError(error_msg)

            logits = train_dict['logits']
            pos_preds = train_dict['pos_preds']
            neg_preds = train_dict['neg_preds']
            pos_useful = train_dict['pos_useful']
            tgt_found = train_dict['tgt_found']

        # Get useful positives, found targets and positive and negative target masks
        if mode == 'self':
            tgt_mask_args = (feat_maps, tgt_boxes, neg_preds)
            tgt_mask_kwargs = {'mode': mode, 'pos_preds': pos_preds}
            pos_useful, tgt_found, pos_tgts, neg_tgts = self.get_target_masks(*tgt_mask_args, **tgt_mask_kwargs)

        else:
            tgt_mask_args = (feat_maps, tgt_boxes, neg_preds)
            tgt_mask_kwargs = {'mode': mode, 'tgt_found': tgt_found}
            pos_tgts, neg_tgts = self.get_target_masks(*tgt_mask_args, **tgt_mask_kwargs)

        analysis_dict['pos_useful'] = torch.sum(pos_useful)[None] / batch_size
        analysis_dict['pos_tgts'] = torch.sum(pos_tgts)[None] / batch_size
        analysis_dict['neg_tgts'] = torch.sum(neg_tgts)[None] / batch_size

        tgt_found = torch.cat(tgt_found, dim=0)
        num_tgts = len(tgt_found)

        if num_tgts > 0:
            analysis_dict['tgt_found'] = 100 * torch.sum(tgt_found)[None] / num_tgts
        else:
            analysis_dict['tgt_found'] = 100 * torch.ones(1).to(tgt_found)

        # Get weighted positive, negative and hill losses
        targets = torch.full_like(pos_preds, fill_value=-1, dtype=torch.int64)
        targets[pos_tgts] = 1
        targets[neg_tgts] = 0

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
        loss_dict['hill_loss'] = self.hill_weight * self.hill_loss(logits, reduction='sum')

        # Return dictionaries
        loss_dict = {f'dod_{k}': v for k, v in loss_dict.items()}
        analysis_dict = {f'dod_{k}': v for k, v in analysis_dict.items()}

        if mode == 'self' and not self.training:
            pred_dicts = []
            return pred_dicts, loss_dict, analysis_dict
        else:
            return loss_dict, analysis_dict
