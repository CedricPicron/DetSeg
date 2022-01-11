"""
Collection of modules related to graphs.
"""

from torch import nn

from models.functional.net import get_net_multi


class GraphToGraph(nn.Module):
    """
    Class implementing the Tree-based Graph-to-Graph module.

    Attributes:
        node_score (Sequential): Module computing the node scores.
        edge_score (Sequential): Module computing the edge scores.
  """

    def __init__(self, node_score_dicts, edge_score_dicts):
        """
        Initializes the GraphToGraph module.

        Args:
            node_score_dicts (Dict): Dictionary of multiple network dictionaries specifying the node score network.
            edge_score_dicts (Dict): Dictionary of multiple network dictionaries specifying the edge score network.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Get networks computing the node and edge scores
        self.node_score = get_net_multi(node_score_dicts)
        self.edge_score = get_net_multi(edge_score_dicts)

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

        # Get node and edge scores
        node_scores = self.node_score(node_feats)
        edge_scores = self.edge_score(node_feats, node_adj_ids=node_adj_ids, edge_ids=edge_ids)

        return node_scores, edge_scores
