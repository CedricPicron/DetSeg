"""
Bidirectional Vision Network (BVN) architecture.
"""
from copy import deepcopy

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from models.build import MODELS
from models.functional.downsample import downsample_masks


@MODELS.register_module()
class BVN(nn.Module):
    """
    Class implementing the BVN module.

    Attributes:
        backbone (nn.Module): Module implementing the backbone.
        core (nn.Module): Module implementing the core.
        step_mode (str): String chosen from {multi, single} containing the BVN step mode.
        num_core_layers (int): Integer containing the number of consecutive core layers.

        If step mode is 'multi':
            heads (nn.ModuleList): List of size [num_head_copies] containing copied dictionaries of head modules.

        If step mode is 'single':
            heads (nn.ModuleDict): Dictionary of size [num_heads] with head modules.

        sync_heads: Boolean indicating whether to synchronize heads copies in multi-step mode.
    """

    def __init__(self, backbone, core, step_mode, heads, sync_heads):
        """
        Initializes the BVN module.

        Args:
            backbone (nn.Module): Module implementing the backbone.
            core (nn.Module): Module implementing the core.
            step_mode (str): String chosen from {multi, single} containing the BVN step mode.
            heads (Dict): Dictionary of size [num_heads] with head modules.
            sync_heads: Boolean indicating whether to synchronize heads copies in multi-step mode.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set backbone, core and multi step attributes
        self.backbone = backbone
        self.core = core
        self.step_mode = step_mode

        # Set number of core layers attribute
        self.num_core_layers = getattr(core, 'num_layers', 0)

        # Set heads attribute
        if step_mode == 'multi':
            num_head_copies = self.num_core_layers + 1
            self.heads = nn.ModuleList([deepcopy(nn.ModuleDict(heads)) for _ in range(num_head_copies)])

        elif step_mode == 'single':
            self.heads = nn.ModuleDict(heads)

        # Set synchronize heads attribute
        self.sync_heads = sync_heads

    @staticmethod
    def get_param_families():
        """
        Method returning the BVN parameter families.

        Returns:
            List of strings containing the BVN parameter families.
        """

        return ['backbone', 'core', 'heads']

    @staticmethod
    def get_feat_masks(img_masks, feat_maps):
        """
        Method obtaining feature masks corresponding to each feature map encoding features from padded image regions.

        Args:
            img_masks (BoolTensor): masks encoding padded pixels of shape [batch_size, max_iH, max_iW].
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

        Returns:
            feat_masks (List): List of size [num_maps] with masks of active features of shape [batch_size, fH, fW].
        """

        map_sizes = [tuple(feat_map.shape[2:]) for feat_map in feat_maps]
        padding_maps = downsample_masks(img_masks, map_sizes)
        feat_masks = [padding_map > 0.5 for padding_map in padding_maps]

        return feat_masks

    def train_evaluate(self, feat_maps, feat_masks, tgt_dict, optimizer, step_id=0, **kwargs):
        """
        Method performing the training/evaluation step for the given feature maps.

        Loss and analysis dictionaries are computed from the input feature maps using the module's heads.
        The model parameters are updated during training by backpropagating the loss terms from the loss dictionary.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, fH, fW, feat_size].
            feat_masks (List): List of size [num_maps] with masks of active features of shape [batch_size, fH, fW].
            tgt_dict (Dict): Target dictionary with ground-truth information used for loss computation and evaluation.
            optimizer (torch.optim.Optimizer): Optimizer updating the BVN parameters during training.
            step_id (int): Optional integer indicating the core step in multi-step mode (default=0).

            kwargs (Dict): Dictionary of keyword arguments, potentially containing following key:
                - extended_analysis (bool): boolean indicating whether to perform extended analyses or not.

        Returns:
            loss_dict (Dict): Dictionary of different weighted loss terms used for backpropagation during training.
            analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.
        """

        # Initialize loss and analysis dictionaries
        loss_dict = {}
        analysis_dict = {}

        # Get heads
        heads = self.heads[step_id] if self.step_mode == 'multi' else self.heads

        # Populate loss and analysis dictionaries from head outputs
        for head in heads.values():
            head_loss_dict, head_analysis_dict = head(feat_maps, feat_masks=feat_masks, tgt_dict=tgt_dict, **kwargs)

            if self.step_mode == 'multi':
                loss_dict.update({f'{k}_{step_id}': v for k, v in head_loss_dict.items()})
                analysis_dict.update({f'{k}_{step_id}': v for k, v in head_analysis_dict.items()})

            elif self.step_mode == 'single':
                loss_dict.update(head_loss_dict)
                analysis_dict.update(head_analysis_dict)

        # Add total step loss to analysis dictionary in multi-step mode
        with torch.no_grad():
            if self.step_mode == 'multi':
                analysis_dict[f'loss_{step_id}'] = sum(loss_dict.values())

        # Return loss and analysis dictionaries (validation only)
        if optimizer is None:
            return loss_dict, analysis_dict

        # Backpropagate loss (training only)
        loss = sum(loss_dict.values())
        loss.backward()

        return loss_dict, analysis_dict

    def forward(self, images, tgt_dict=None, optimizer=None, max_grad_norm=-1, visualize=False, **kwargs):
        """
        Forward method of the BVN module.

        Args:
            images (Images): Images structure containing the batched images.
            tgt_dict (Dict): Target dictionary with ground-truth information used during trainval (default=None).
            optimizer (torch.optim.Optimizer): Optimizer updating the BVN parameters during training (default=None).
            max_grad_norm (float): Maximum gradient norm of parameters throughout model (default=-1).
            visualize (bool): Boolean indicating whether to compute dictionary with visualizations (default=False).

            kwargs (Dict): Dictionary of keyword arguments, potentially containing following key:
                - extended_analysis (bool): boolean indicating whether to perform extended analyses or not.

       Returns:
            * If tgt_dict is not None and optimizer is not None (i.e. during training):
                loss_dict (Dict): Dictionary of different weighted loss terms used for backpropagation during training.
                analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

            * If tgt_dict is not None and optimizer is None (i.e. during validation):
                pred_dicts (List): List of dictionaries with predictions (from last step if in multi-step mode).
                loss_dict (Dict): Dictionary of different weighted loss terms used for backpropagation during training.
                analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

                * If additionally the keyword argument 'visualize' is True:
                    images_dict (Dict): Dictionary of different annotated images based on predictions and targets.

            * If tgt_dict is None (i.e. during testing):
                pred_dicts (List): List of dictionaries with predictions (from last step if in multi-step mode).
                analysis_dict (Dict): Empty dictionary.

        Raises:
            RuntimeError: Error when visualization is requested for BVN model with more than one head.
        """

        # Reset gradients of model parameters (training only)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        # Get backbone feature maps
        feat_maps = self.backbone(images)

        # 1) Multi-step mode
        if self.step_mode == 'multi':

            # Get initial core feature maps
            feat_maps = self.core(feat_maps, step_id=0)

            # Prepare heads and target dictionary for current forward pass
            for head_key in self.heads[0]:
                head_copies = [heads_copies[head_key] for heads_copies in self.heads]
                tgt_dict, attr_dict, buffer_dict = head_copies[0].forward_init(feat_maps, tgt_dict)
                [setattr(head, k, v) for head in head_copies for k, v in attr_dict.items()]
                [getattr(head, k).copy_(v) for head in head_copies for k, v in buffer_dict.items()]

            # Train/evaluate initial core feature maps (trainval only)
            if tgt_dict is not None:
                feat_masks = BVN.get_feat_masks(images.masks, feat_maps)
                train_eval_args = (feat_masks, tgt_dict, optimizer)
                loss_dict, analysis_dict = self.train_evaluate(feat_maps, *train_eval_args, step_id=0, **kwargs)

            # Iterate over core layers
            for i in range(1, self.num_core_layers+1):

                # Detach core feature maps (training only)
                if optimizer is not None:
                    feat_maps = [feat_map.detach() for feat_map in feat_maps]

                # Update core feature maps
                feat_maps = self.core(feat_maps, step_id=i)

                # Train/evaluate updated core feature maps (trainval only)
                if tgt_dict is not None:
                    layer_dicts = self.train_evaluate(feat_maps, *train_eval_args, step_id=i, **kwargs)
                    loss_dict.update(layer_dicts[0])
                    analysis_dict.update(layer_dicts[1])

        # 2) Single-step mode
        if self.step_mode == 'single':

            # Get core feature maps
            feat_maps = self.core(feat_maps)

            # Prepare heads and target dictionary for current forward pass
            for head in self.heads.values():
                tgt_dict, attr_dict, buffer_dict = head.forward_init(images, feat_maps, tgt_dict)
                [setattr(head, k, v) for k, v in attr_dict.items()]
                [getattr(head, k).copy_(v) for k, v in buffer_dict.items()]

            # Train/evaluate core feature maps (trainval only)
            if tgt_dict is not None:
                feat_masks = BVN.get_feat_masks(images.masks, feat_maps)
                train_eval_args = (feat_masks, tgt_dict, optimizer)
                loss_dict, analysis_dict = self.train_evaluate(feat_maps, *train_eval_args, **kwargs)

        # Get prediction dictionaries (validation/testing only)
        if optimizer is None:
            pred_heads = self.heads[-1] if self.step_mode == 'multi' else self.heads
            pred_kwargs = {'visualize': visualize, **kwargs}
            pred_dicts = [pred_dict for head in pred_heads.values() for pred_dict in head(feat_maps, **pred_kwargs)]

        # Return prediction dictionaries and empty analysis dictionary (testing only)
        if tgt_dict is None:
            analysis_dict = {}
            return pred_dicts, analysis_dict

        # Return desired dictionaries (validation/visualization only)
        if optimizer is None:

            # Return prediction, loss and analysis dictionaries (validation only)
            if not visualize:
                return pred_dicts, loss_dict, analysis_dict

            # Get and return annotated images, loss and analysis dictionaries (visualization only)
            if len(pred_heads) == 1:
                images_dict = head.visualize(images, pred_dicts, tgt_dict)
            else:
                error_msg = f"BVN only supports visualization for models with a single head (got {len(pred_heads)})."
                raise RuntimeError(error_msg)

            return pred_dicts, loss_dict, analysis_dict, images_dict

        # Update model parameters (training only)
        clip_grad_norm_(self.parameters(), max_grad_norm) if max_grad_norm > 0 else None
        optimizer.step()

        # If requested, synchronize head copies in multi-step mode (training only)
        if self.step_mode == 'multi' and self.sync_heads:
            avg_heads_state_dict = {}

            for key_values in zip(*[heads.state_dict().items() for heads in self.heads]):
                keys, values = list(zip(*key_values))
                avg_heads_state_dict[keys[0]] = sum(values) / len(values)

            [heads.load_state_dict(avg_heads_state_dict) for heads in self.heads]

        return loss_dict, analysis_dict
