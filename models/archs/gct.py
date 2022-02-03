"""
Graph-Connecting Trees (GCT) architecture.
"""

import torch
from torch import nn

from models.build import build_model, MODELS
from models.functional.graph import map_to_graph


@MODELS.register_module()
class GCT(nn.Module):
    """
    Class implementing the Graph-Connecting Trees (GCT) architecture.

    Attributes:
        map (nn.Module): Module performing the initial map-based processing.
        struc (nn.Module): Module computing the initial structure features from normalized node locations.
        graph (nn.Module): Module performing the subsequent graph-based processing.
    """

    def __init__(self, map_cfg, graph_cfg):
        """
        Initializes the GCT module.

        Args:
            map_cfg (Dict): Configuration dictionary specifying the initial map-based processing module.
            graph_cfg (Dict): Configuration dictionary specifying the subsequent graph-based processing module.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build initial map-based processing module
        self.map = build_model(map_cfg)

        # Build subsequent graph-based processing module
        self.graph = build_model(graph_cfg)

    @staticmethod
    def get_param_families():
        """
        Method returning the GCT parameter families.

        Returns:
            List of strings containing the GCT parameter families.
        """

        return ['map', 'graph']

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
            ValueError: Error when the output graph index is larger than graph list length.
        """

        # Check inputs
        if tgt_dict is None and optimizer is not None:
            error_msg = "An optimizer is provided, but no target dictionary to learn from."
            raise RuntimeError(error_msg)

        # Perform map-based processing
        feat_map = self.map(images)
        feat_map = feat_map[-1] if isinstance(feat_map, (list, tuple)) else feat_map

        # Get initial graph from feature map
        graph = map_to_graph(feat_map)
        graph['con_feats'] = graph.pop('node_feats')

        # Get initial structure features
        graph['struc_feats'] = graph.pop('node_xy')

        # Get initial edge weights
        num_edges = graph['edge_ids'].size(dim=1)
        graph['edge_weights'] = torch.ones(num_edges, dtype=torch.float, device=feat_map.device)

        # Get list with graphs
        graph_list = self.graph(graph)

        return graph_list
