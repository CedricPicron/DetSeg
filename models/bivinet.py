"""
BiViNet module and build function.
"""
from copy import deepcopy

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from .backbone import build_backbone
from .cores.build import build_core
from .heads.detection import build_det_heads
from .heads.segmentation import build_seg_heads
from .utils import downsample_masks


class BiViNet(nn.Module):
    """
    Class implementing the BiViNet module.

    Attributes:
        backbone (nn.Module): Module implementing the BiViNet backbone.
        core (nn.Module): Module implementing the BiViNet core.
        heads (nn.ModuleList): List of size [num_head_copies] containing copied lists of BiViNet head modules.
        multi_step (bool): Boolean indicating whether to use multi-step parameter updates with gradient blocking.
    """

    def __init__(self, backbone, core, heads, multi_step):
        """
        Initializes the BiViNet module.

        Args:
            backbone (nn.Module): Module implementing the BiViNet backbone.
            core (nn.Module): Module implementing the BiViNet core.
            heads (List): List of size [num_heads] with BiViNet head modules.
            multi_step (bool): Boolean indicating whether to use multi-step parameter updates with gradient blocking.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set backbone, core and multi step attributes
        self.backbone = backbone
        self.core = core
        self.multi_step = multi_step

        # Set number of core layers and heads attributes
        self.num_core_layers = getattr(core, 'num_layers', 1)
        num_head_copies = self.num_core_layers + 1 if multi_step else 1
        self.heads = nn.ModuleList([deepcopy(nn.ModuleList(heads)) for _ in range(num_head_copies)])

    @staticmethod
    def get_param_families():
        """
        Method returning the BiViNet parameter families.

        Returns:
            List of strings containing the BiViNet parameter families.
        """

        return ['backbone', 'core', 'heads']

    @staticmethod
    def get_feat_masks(img_masks, feat_maps):
        """
        Method obtaining feature masks corresponding to each feature map encoding features from padded image regions.

        Args:
            img_masks (BoolTensor): masks encoding padded pixels of shape [batch_size, max_iH, max_iW].
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, fH, fW, feat_size].

        Returns:
            feat_masks (List): List of size [num_maps] with masks of active features of shape [batch_size, fH, fW].
        """

        map_sizes = [tuple(feat_map.shape[1:-1]) for feat_map in feat_maps]
        padding_maps = downsample_masks(img_masks, map_sizes)
        feat_masks = [padding_map > 0.5 for padding_map in padding_maps]

        return feat_masks

    def train_evaluate(self, feat_maps, feat_masks, tgt_dict, optimizer, step_id=None, **kwargs):
        """
        Method performing the training/evaluation step for the given feature maps.

        Loss and analysis dictionaries are computed from the input feature maps using the module's heads.
        The model parameters are updated during training by backpropagating the loss terms from the loss dictionary.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, fH, fW, feat_size].
            feat_masks (List): List of size [num_maps] with masks of active features of shape [batch_size, fH, fW].
            tgt_dict (Dict): Target dictionary with ground-truth information used for loss computation and evaluation.
            optimizer (torch.optim.Optimizer): Optimizer updating the BiViNet parameters during training.
            step_id (int): Optional integer indicating the core step in multi-step mode.

            kwargs(Dict): Dictionary of keyword arguments, potentially containing following key:
                - max_grad_norm (float): maximum norm of optimizer update during training (clipped if larger).

        Returns:
            loss_dict (Dict): Dictionary of different weighted loss terms used for backpropagation during training.
            analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.
        """

        # Initialize loss and analysis dictionaries
        loss_dict = {}
        analysis_dict = {}

        # Get head copies id
        head_copies_id = step_id if self.multi_step else 0

        # Populate loss and analysis dictionaries from head outputs
        for head in self.heads[head_copies_id]:
            head_loss_dict, head_analysis_dict = head(feat_maps, feat_masks, tgt_dict, **kwargs)

            if self.multi_step:
                loss_dict.update({f'{k}_{step_id}': v for k, v in head_loss_dict.items()})
                analysis_dict.update({f'{k}_{step_id}': v for k, v in head_analysis_dict.items()})

        # Add total step loss to analysis dictionary in multi-step mode
        with torch.no_grad():
            if self.multi_step:
                analysis_dict[f'loss_{step_id}'] = sum(loss_dict.values())

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
            images (Images): Images structure containing the batched images.
            tgt_dict (Dict): Optional dictionary with ground-truth information used during training and validation.
            optimizer (torch.optim.Optimizer): Optional optimizer updating the BiViNet parameters during training.

            kwargs (Dict): Dictionary of keyword arguments, potentially containing following keys:
                - max_grad_norm (float): maximum norm of optimizer update during training (clipped if larger);
                - visualize (bool): boolean indicating whether in visualization mode or not.

       Returns:
            * If tgt_dict is not None and optimizer is not None (i.e. during training):
                loss_dict (Dict): Dictionary of different weighted loss terms used for backpropagation during training.
                analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

            * If tgt_dict is not None and optimizer is None (i.e. during validation):
                pred_dict (Dict): Dictionary containing different predictions (from last step if in multi-step mode).
                loss_dict (Dict): Dictionary of different weighted loss terms used for backpropagation during training.
                analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

            * If in validation mode (see above) with the keyword argument 'visualize=True' (i.e. during visualization):
                images_dict (Dict): Dictionary of different annotated images based on predictions and targets.
                loss_dict (Dict): Dictionary of different weighted loss terms used for backpropagation during training.
                analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

            * If tgt_dict is None (i.e. during testing):
                pred_dict (Dict): Dictionary containing different predictions (from last step if in multi-step mode).
        """

        # Reset gradients of model parameters (training only)
        if optimizer is not None:
            optimizer.zero_grad()

        # Get backbone feature maps
        feat_maps, _ = self.backbone(images)

        # Prepare heads and target dictionary for current forward pass
        for head_copies in zip(*self.heads):
            tgt_dict, attr_dict, buffer_dict = head_copies[0].forward_init(feat_maps, tgt_dict)
            [setattr(head, k, v) for head in head_copies for k, v in attr_dict.items()]
            [getattr(head, k).copy_(v) for head in head_copies for k, v in buffer_dict.items()]

        # Get feature masks (trainval only)
        if tgt_dict is not None:
            feat_masks = BiViNet.get_feat_masks(images.masks, feat_maps)
            train_eval_args = (feat_masks, tgt_dict, optimizer)

        # Train/evaluate initial core feature maps in multi-step mode (trainval only)
        if self.multi_step and tgt_dict is not None:
            feat_maps = self.core(feat_maps, step_id=0)
            loss_dict, analysis_dict = self.train_evaluate(feat_maps, *train_eval_args, step_id=0, **kwargs)

        # Get or update core feature maps
        for i in enumerate(self.num_core_layers, 1):

            # Detach core feature maps in multi-step mode (training only)
            if self.multi_step and optimizer is not None:
                feat_maps = [feat_map.detach() for feat_map in feat_maps]

            # Get or update core feature maps
            feat_maps = self.core(feat_maps, step_id=i)

            # Train/evaluate updated core feature maps in multi-step mode (trainval only)
            if self.multi_step and tgt_dict is not None:
                layer_dicts = self.train_evaluate(feat_maps, *train_eval_args, step_id=i, **kwargs)
                loss_dict.update(layer_dicts[0])
                analysis_dict.update(layer_dicts[1])

        # Train/evaluate core feature maps if not in multi-step mode (trainval only)
        if not self.multi_step and tgt_dict is not None:
            layer_dicts = self.train_evaluate(feat_maps, *train_eval_args, **kwargs)
            loss_dict = layer_dicts[0]
            analysis_dict = layer_dicts[1]

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

        # Average and synchronize head copies in multi-step mode (training only)
        if self.multi_step:
            avg_heads_state_dict = {}

            for key_values in zip(*[heads.state_dict().items() for heads in self.heads]):
                keys, values = list(zip(*key_values))
                avg_heads_state_dict[keys[0]] = sum(values) / len(values)

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

    # Build backbone
    backbone = build_backbone(args)
    args.backbone_feat_sizes = backbone.feat_sizes

    # Build core
    core = build_core(args)

    # Build detection and segmentation heads
    heads = [*build_det_heads(args), *build_seg_heads(args)]

    # Build BiViNet module
    bivinet = BiViNet(backbone, core, heads, args.bvn_multi_step)

    return bivinet
