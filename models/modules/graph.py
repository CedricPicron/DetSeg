"""
Collection of modules related to graphs.
"""

import torch
from torch import nn

from models.build import build_model, MODELS
from models.functional.graph import edge_dot_product


@MODELS.register_module()
class EdgeDotProduct(nn.Module):
    """
    Class implementing the EdgeDotProduct module computing dot products between source and target features of edges.

    Attributes:
        implementation (str): String containing the type of implementation (default='pytorch-custom').
    """

    def __init__(self, implementation='pytorch-custom'):
        """
        Initializes the EdgeDotProduct module.

        Args:
            implementation (str): String containing the type of implementation (default='pytorch-custom').
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set implementation attribute
        self.implementation = implementation

    def forward(self, node_src_feats, node_tgt_feats=None, edge_ids=None, **kwargs):
        """
        Forward method of the EdgeDotProduct module.

        Args:
            node_src_feats (FloatTensor): Node source features of shape [num_nodes, src_feat_size].
            node_tgt_feats (FloatTensor): Node target features of shape [num_nodes, tgt_feat_size] (default=None).
            edge_ids (LongTensor): Node indices for each (directed) edge of shape [2, num_edges] (default=None).
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            edge_dot_prods (FloatTensor): Tensor containing the edge dot products of shape [num_edges].

        Raises:
            ValueError: Error when no 'edge_ids' are provided.
        """

        # Check whether 'node_tgt_feats' are provided
        if node_tgt_feats is None:
            node_tgt_feats = node_src_feats

        # Check whether 'edge_ids' are provided
        if edge_ids is None:
            error_msg = "The 'edge_ids' input argument must be provided, but is missing."
            raise ValueError(error_msg)

        # Compute edge dot products
        edge_dot_prods = edge_dot_product(node_src_feats, node_tgt_feats, edge_ids, self.implementation)

        return edge_dot_prods


@MODELS.register_module()
class GraphToGraph(nn.Module):
    """
    Class implementing the Tree-based Graph-to-Graph module.

    Attributes:
        node_score (nn.Module): Module computing the unnormalized node scores.
        edge_score (nn.Module): Module computing the unnormalized edge scores.
  """

    def __init__(self, node_score_cfg, edge_score_cfg):
        """
        Initializes the GraphToGraph module.

        Args:
            node_score_cfg (Dict): Configuration dictionary specifying the node score network.
            edge_score_cfg (Dict): Configuration dictionary specifying the edge score network.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build networks computing the unnormalized node and edge scores
        self.node_score = build_model(node_score_cfg)
        self.edge_score = build_model(edge_score_cfg)

    def forward(self, node_feats, node_xy, node_adj_ids, edge_ids, **kwargs):
        """
        Forward method of the GraphToGraph module.

        Args:
            node_feats (FloatTensor): Graph node features of shape [num_nodes, feat_size].
            node_xy (FloatTensor): Node locations in normalized (x, y) format of shape [num_nodes, 2].
            node_adj_ids (List): List of size [num_nodes] with lists of adjacent node indices (including itself).
            edge_ids (LongTensor): Tensor containing the node indices for each (directed) edge of shape [2, num_edges].
            kwargs (Dict): Dictionary of unused keyword arguments.
        """

        # 1. Get normalized node and edge scores
        node_scores = torch.sigmoid(self.node_score(node_feats.squeeze(dim=1)))
        edge_scores = torch.sigmoid(self.edge_score(node_feats, edge_ids=edge_ids))

        # 2. Perform node grouping

        # 3. Get new node features

        # 4. Get new edges

        return node_scores, edge_scores
