"""
Backbone-Core-Heads (BCH) architecture.
"""
from collections import ChainMap
from itertools import chain

from torch import nn
from torch.nn.utils import clip_grad_norm_

from models.build import MODELS


@MODELS.register_module()
class BCH(nn.Module):
    """
    Class implementing the BCH architecture, consisting of a backbone, a core and heads modules.

    Attributes:
        backbone (nn.Module): Module implementing the backbone.
        core (nn.Module): Module implementing the core.
        heads (nn.ModuleDict): Dictionary of size [num_heads] with head modules.
    """

    def __init__(self, backbone, core, heads):
        """
        Initializes the BCH module.

        Args:
            backbone (nn.Module): Module implementing the backbone.
            core (nn.Module): Module implementing the core.
            heads (Dict): Dictionary of size [num_heads] with head modules.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set backbone, core and heads attributes
        self.backbone = backbone
        self.core = core
        self.heads = nn.ModuleDict(heads)

    @staticmethod
    def get_param_families():
        """
        Method returning the BCH parameter families.

        Returns:
            List of strings containing the BCH parameter families.
        """

        return ['backbone', 'core', 'heads']

    def forward(self, images, tgt_dict=None, optimizer=None, max_grad_norm=-1, visualize=False, **kwargs):
        """
        Forward method of the BCH module.

        Args:
            images (Images): Images structure containing the batched images.
            tgt_dict (Dict): Target dictionary with ground-truth information used during trainval (default=None).
            optimizer (torch.optim.Optimizer): Optimizer updating the BCH parameters during training (default=None).
            max_grad_norm (float): Maximum gradient norm of parameters throughout model (default=-1).
            visualize (bool): Boolean indicating whether to compute dictionary with visualizations (default=False).

            kwargs (Dict): Dictionary of keyword arguments, potentially containing following key:
                - extended_analysis (bool): boolean indicating whether to perform extended analyses or not.

        Returns:
            output_dicts (List): List of output dictionaries, potentially containing following items:
                - pred_dicts (List): list of dictionaries with predictions;
                - loss_dict (Dict): dictionary of different loss terms used for backpropagation during training;
                - analysis_dict (Dict): dictionary of different analyses used for logging purposes only;
                - images_dict (Dict): dictionary of different annotated images based on predictions and targets.

        Raises:
            TypeError: Error when an optimizer is provided, but no target dictionary.
        """

        # Check inputs
        if tgt_dict is None and optimizer is not None:
            error_msg = "A target dictionary must be provided together with the provided optimizer."
            raise TypeError(error_msg)

        # Apply backbone
        backbone_feat_maps = self.backbone(images)

        # Apply core
        core_feat_maps = self.core(backbone_feat_maps, images=images)

        # Apply heads and merge non-prediction dictionaries originating from different heads
        head_kwargs = {'tgt_dict': tgt_dict, 'images': images, 'visualize': visualize, **kwargs}
        head_dicts = [head(core_feat_maps, **head_kwargs) for head in self.heads.values()]

        if self.training:
            output_dicts = [dict(ChainMap(*dicts)) for dicts in zip(*head_dicts)]
        else:
            zipped_dicts = list(zip(*head_dicts))
            pred_dicts = list(chain.from_iterable(zipped_dicts[0]))
            non_pred_dicts = [dict(ChainMap(*dicts)) for dicts in zipped_dicts[1:]]
            output_dicts = [pred_dicts, *non_pred_dicts]

        # Update model parameters and return loss and analysis dictionaries during training
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

            loss_dict = output_dicts[0]
            loss = sum(loss_dict.values())
            loss.backward()

            clip_grad_norm_(self.parameters(), max_grad_norm) if max_grad_norm > 0 else None
            optimizer.step()

        return output_dicts
