"""
Graph-Connecting Trees (GCT) architecture.
"""

from torch import nn


class GCT(nn.Module):
    """
    Class implementing the Graph-Connecting Trees (GCT) architecture.

    Attributes:
        backbone (nn.Module): Module implementing the backbone.
    """

    def __init__(self, backbone):
        """
        Initializes the GCT module.

        Args:
            backbone (nn.Module): Module implementing the backbone.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set backbone attribute
        self.backbone = backbone

    @staticmethod
    def get_param_families():
        """
        Method returning the GCT parameter families.

        Returns:
            List of strings containing the GCT parameter families.
        """

        return ['backbone']

    def forward(self, images, tgt_dict=None, optimizer=None, max_grad_norm=-1, visualize=False, **kwargs):
        """
        Forward method of the GCT module.

        Args:
            images (Images): Images structure containing the batched images.
            tgt_dict (Dict): Target dictionary with ground-truth information used during trainval (default=None).
            optimizer (torch.optim.Optimizer): Optimizer updating the GCT parameters during training (default=None).
            max_grad_norm (float): Maximum gradient norm of parameters throughout model (default=-1).
            visualize (bool): Boolean indicating whether to compute dictionary with visualizations (default=False).
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            output_dicts (List): List of output dictionaries, potentially containing following items:
                - pred_dicts (List): list of dictionaries with predictions;
                - loss_dict (Dict): dictionary of different loss terms used for backpropagation during training;
                - analysis_dict (Dict): dictionary of different analyses used for logging purposes only;
                - images_dict (Dict): dictionary of different annotated images based on predictions and targets.

        Raises:
            RuntimeError: Error when an optimizer is provided, but no target dictionary.
        """

        # Check inputs
        if tgt_dict is None and optimizer is not None:
            error_msg = "An optimizer is provided, but no target dictionary to learn from."
            raise RuntimeError(error_msg)

        # Apply backbone
        backbone_feat_maps = self.backbone(images)

        return backbone_feat_maps
