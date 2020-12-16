"""
BiViNet modules and build function.
"""
import copy

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from .backbone import build_backbone
from .bicore import build_bicore
from .heads.detection import build_det_heads
from .heads.segmentation import build_seg_heads
from .utils import downsample_masks


class BiViNet(nn.Module):
    """
    Class implementing the BiViNet module.

    Attributes:
        backbone (nn.Module): Module implementing the BiViNet backbone.
        projs (nn.ModuleList): List of size [num_core_maps] implementing backbone to core projection modules.
        cores (nn.ModuleList): List of size [num_core_layers] with concatenated core modules.
        heads (nn.ModuleList): List of size [num_core_layers+1] with copies of BiViNet head modules for each layer.
    """

    def __init__(self, backbone, core_feat_sizes, core, num_core_layers, heads):
        """
        Initializes the BiViNet module.

        Args:
            backbone (nn.Module): Module implementing the BiViNet backbone.
            core_feat_sizes (List): List of size [num_core_maps] containing the feature size of each core feature map.
            core (nn.Module): Module implementing the BiViNet core.
            num_core_layers (int): Number of concatenated core layers.
            heads (List): List of size [num_heads] with BiViNet head modules.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set backbone, core and heads attributes
        self.backbone = backbone
        self.cores = nn.ModuleList([copy.deepcopy(core) for _ in range(num_core_layers)])
        self.heads = nn.ModuleList([copy.deepcopy(nn.ModuleList(heads)) for _ in range(num_core_layers+1)])

        # Build backbone to core projection modules
        f0s = backbone.feat_sizes
        f1s = core_feat_sizes[:len(f0s)]
        self.projs = nn.ModuleList(nn.Conv2d(f0, f1, kernel_size=1) for f0, f1 in zip(f0s, f1s))
        self.projs.append(nn.Conv2d(f0s[-1], core_feat_sizes[-1], kernel_size=3, stride=2, padding=1))

    @staticmethod
    def get_param_families():
        """
        Method returning the BiViNet parameter families.

        Returns:
            List of strings containing the BiViNet parameter families.
        """

        return ['backbone', 'projs', 'core', 'heads']

    @staticmethod
    def get_feat_masks(img_masks, feat_maps):
        """
        Method obtaining feature masks corresponding to each feature map encoding features from padded image regions.

        Args:
            img_masks (BoolTensor): masks encoding padded pixels of shape [batch_size, max_iH, max_iW].
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, fH, fW, feat_size].

        Returns:
            feat_masks (List): List of size [num_core_maps] with padding feature masks of shape [batch_size, fH, fW].
        """

        map_sizes = [tuple(feat_map.shape[1:-1]) for feat_map in feat_maps]
        padding_maps = downsample_masks(img_masks, map_sizes)
        feat_masks = [padding_map > 0.5 for padding_map in padding_maps]

        return feat_masks

    def train_evaluate(self, feat_maps, feat_masks, tgt_dict, optimizer, layer_id, **kwargs):
        """
        Method performing the training/evaluation step for the given feature maps.

        Loss and analysis dictionaries are computed from the input feature maps using the module's heads.
        The model parameters are updated during training by backpropagating the loss terms from the loss dictionary.

        Args:
            feat_maps (List): List of size [num_core_maps] with feature maps of shape [batch_size, fH, fW, feat_size].
            feat_masks (List): List of size [num_core_maps] with padding feature masks of shape [batch_size, fH, fW].
            tgt_dict (Dict): Target dictionary containing following keys:
                - labels (LongTensor): tensor of shape [num_targets_total] containing the class indices;
                - boxes (FloatTensor): boxes of shape [num_targets_total, 4] in (left, top, right, bottom) format;
                - sizes (LongTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries;
                - binary_maps (List, optional): binary segmentation maps of shape [batch_size, fH, fW];
                - semantic_maps (List): semantic segmentation maps of shape [batch_size, fH, fW].

            optimizer (torch.optim.Optimizer): Optional optimizer updating the BiViNet parameters during training.
            layer_id (int): Layer index, with 0 for the backbone layer and higher integers for the core layers.
            kwargs(Dict): Dictionary of keyword arguments, potentially containing following keys:
                - max_grad_norm (float): maximum norm of optimizer update during training (clipped if larger).

        Returns:
            loss_dict (Dict): Dictionary of different weighted loss terms used for backpropagation during training.
            analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.
        """

        # Initialize loss and analysis dictionaries
        loss_dict = {}
        analysis_dict = {}

        # Populate loss and analysis dictionaries from head outputs
        for head in self.heads[layer_id]:
            head_loss_dict, head_analysis_dict = head(feat_maps, feat_masks, tgt_dict, **kwargs)
            loss_dict.update({f'{k}_{layer_id}': v for k, v in head_loss_dict.items()})
            analysis_dict.update({f'{k}_{layer_id}': v for k, v in head_analysis_dict.items()})

        # Add total layer loss to analysis dictionary
        with torch.no_grad():
            analysis_dict[f'loss_{layer_id}'] = sum(loss_dict.values())

        # Return loss and analysis dictionaries (validation only)
        if optimizer is None:
            return loss_dict, analysis_dict

        # Backpropagate loss (training only)
        loss = sum(loss_dict.values())
        loss.backward()

        return loss_dict, analysis_dict

    def forward(self, images, tgt_dict=None, optimizer=None, **kwargs):
        """
        Forward method of the BiViNet module.

        Args:
            images (NestedTensor): NestedTensor consisting of:
                - images.tensor (FloatTensor): padded images of shape [batch_size, 3, max_iH, max_iW];
                - images.mask (BoolTensor): masks encoding padded pixels of shape [batch_size, max_iH, max_iW].

            tgt_dict (Dict): Optional target dictionary used during training and validation containing following keys:
                - labels (LongTensor): tensor of shape [num_targets_total] containing the class indices;
                - boxes (FloatTensor): boxes of shape [num_targets_total, 4] in (left, top, right, bottom) format;
                - sizes (LongTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries;
                - masks (ByteTensor): padded segmentation masks of shape [num_targets_total, max_iH, max_iW].

            optimizer (torch.optim.Optimizer): Optional optimizer updating the BiViNet parameters during training.
            kwargs(Dict): Dictionary of keyword arguments, potentially containing following keys:
                - max_grad_norm (float): maximum norm of optimizer update during training (clipped if larger);
                - visualize (bool): boolean indicating whether in visualization mode or not.

       Returns:
            * If tgt_dict is not None and optimizer is not None (i.e. during training):
                loss_dict (Dict): Dictionary of different weighted loss terms used for backpropagation during training.
                analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

            * If tgt_dict is not None and optimizer is None (i.e. during validation):
                pred_dict (Dict): Dictionary containing different predictions from the last core layer.
                loss_dict (Dict): Dictionary of different weighted loss terms used for backpropagation during training.
                analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

            * If in validation mode (see above) with the keyword argument 'visualize=True' (i.e. during visualization):
                images_dict (Dict): Dictionary of different annotated images based on predictions and targets.
                loss_dict (Dict): Dictionary of different weighted loss terms used for backpropagation during training.
                analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

            * If tgt_dict is None (i.e. during testing):
                pred_dict (Dict): Dictionary containing different predictions from the last core layer.
        """

        # Reset gradients of model parameters (training only)
        if optimizer is not None:
            optimizer.zero_grad()

        # Get backbone feature maps
        feat_maps, _ = self.backbone(images)

        # Get initial core feature maps by projecting backbone feature maps
        map_ids = [min(i, len(feat_maps)-1) for i in range(len(self.projs))]
        feat_maps = [proj(feat_maps[i]).permute(0, 2, 3, 1) for i, proj in zip(map_ids, self.projs)]

        # Prepare heads and target dictionary for current forward pass
        for head_copies in zip(*self.heads):
            tgt_dict, attr_dict, buffer_dict = head_copies[0].forward_init(feat_maps, tgt_dict)
            [setattr(head, k, v) for head in head_copies for k, v in attr_dict.items()]
            [getattr(head, k).copy_(v) for head in head_copies for k, v in buffer_dict.items()]

        # Get feature masks and train/evaluate initial core feature maps (trainval only)
        if tgt_dict is not None:
            feat_masks = BiViNet.get_feat_masks(images.mask, feat_maps)
            train_eval_args = (feat_masks, tgt_dict, optimizer)
            loss_dict, analysis_dict = self.train_evaluate(feat_maps, *train_eval_args, layer_id=0, **kwargs)

        # Iteratively detach, update and train from and/or evaluate core feature maps
        for i, core in enumerate(self.cores, 1):

            # Set correct requires_grad attributes and detach core feature maps (training only)
            if optimizer is not None:
                feat_maps = [feat_map.detach() for feat_map in feat_maps]

            # Update core feature maps (trainval only)
            feat_maps = core(feat_maps)

            # Train/evaluate updated core feature maps (trainval only)
            if tgt_dict is not None:
                layer_loss_dict, layer_analysis_dict = self.train_evaluate(feat_maps, *train_eval_args, i, **kwargs)
                loss_dict.update(layer_loss_dict)
                analysis_dict.update(layer_analysis_dict)

        # Get prediction dictionary (validation/testing only)
        if optimizer is None:
            pred_dict = {k: v for head in self.heads[0] for k, v in head(feat_maps).items()}

        # Return prediction dictionary (testing only)
        if tgt_dict is None:
            return pred_dict

        # Return desired dictionaries (validation/visualization only)
        if optimizer is None:

            # Return prediction, loss and analysis dictionaries (validation only)
            if not kwargs.setdefault('visualize', False):
                return pred_dict, loss_dict, analysis_dict

            # Get and return annotated images, loss and analysis dictionaries (visualization only)
            images_dict = {}
            for head in self.heads[0]:
                images_dict = {**images_dict, **head.visualize(images, pred_dict, tgt_dict)}

            return images_dict, loss_dict, analysis_dict

        # Clip gradient when positive maximum norm is provided (training only)
        if kwargs.setdefault('max_grad_norm', -1.0) > 0.0:
            clip_grad_norm_(self.parameters(), kwargs['max_grad_norm'])

        # Update model parameters (training only)
        optimizer.step()

        # Compute average heads state dictionary (training only)
        avg_heads_state_dict = {}
        for key_values in zip(*[heads.state_dict().items() for heads in self.heads]):
            keys, values = list(zip(*key_values))
            avg_heads_state_dict[keys[0]] = sum(values) / len(values)

        # Load average heads state dictionary into different heads (training only)
        [heads.load_state_dict(avg_heads_state_dict) for heads in self.heads]

        return loss_dict, analysis_dict


def build_bivinet(args):
    """
    Build BiViNet module from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        bivinet (BiViNet): The specified BiViNet module.
    """

    # Check command-line arguments
    check = args.max_resolution_id > args.min_resolution_id
    assert_msg = "'--max_resolution_id' should be larger than '--min_resolution_id'"
    assert check, assert_msg

    # Get core feature sizes
    map_ids = range(args.min_resolution_id, args.max_resolution_id+1)
    core_feat_sizes = [min((args.base_feat_size * 2**i, args.max_feat_size)) for i in map_ids]

    # Build backbone, core and desired heads
    backbone = build_backbone(args)
    core = build_bicore(args)
    heads = [*build_det_heads(args), *build_seg_heads(args)]

    # Build BiViNet module
    bivinet = BiViNet(backbone, core_feat_sizes, core, args.num_core_layers, heads)

    return bivinet
