"""
Collection of modules related to graphs.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter

from models.build import build_model, MODELS
from models.functional.graph import node_to_edge
from models.functional.sparse import sparse_dense_mm


@MODELS.register_module()
class GraphToGraph(nn.Module):
    """
    Class implementing the GraphToGraph module.

    Attributes:
        edge_score (nn.Module): Module implementing the edge score network.
        num_node_updates (int): Number of node feature updates using the normalized edge score matrix.
        max_group_iters (int): Maximum number of iterations during node grouping.
  """

    def __init__(self, edge_score_cfg, num_node_updates=5, max_group_iters=100):
        """
        Initializes the GraphToGraph module.

        Args:
            edge_score_cfg (Dict): Configuration dictionary specifying the edge score network.
            num_node_updates (int): Number of node feature updates using the normalized edge score matrix (default=5).
            max_group_iters (int): Maximum number of iterations during node grouping (default=100).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build edge score network
        self.edge_score = build_model(edge_score_cfg)

        # Set additional attributes
        self.num_node_updates = num_node_updates
        self.max_group_iters = max_group_iters

    def forward(self, in_graph):
        """
        Forward method of the GraphToGraph module.

        Args:
            in_graph (Dict): Input graph dictionary containing at least following keys:
                con_feats (FloatTensor): content features of shape [num_nodes, con_feat_size];
                struc_feats (FloatTensor): structure features of shape [num_nodes, struc_feat_size];
                edge_ids (LongTensor): node indices for each (directed) edge of shape [2, num_edges];
                edge_weights (FloatTensor): weights for each (directed) edge of shape [num_edges];
                node_xy (FloatTensor): node locations in normalized (x, y) format of shape [num_nodes, 2];
                node_batch_ids (LongTensor): node batch indices of shape [num_nodes].

        Returns:
            out_graph (Dict): Output graph dictionary containing following keys:
                con_feats (FloatTensor): content features of shape [num_nodes, con_feat_size];
                struc_feats (FloatTensor): structure features of shape [num_nodes, struc_feat_size];
                edge_ids (LongTensor): node indices for each (directed) edge of shape [2, num_edges];
                edge_weights (FloatTensor): weights for each (directed) edge of shape [num_edges];
                node_xy (FloatTensor): node locations in normalized (x, y) format of shape [num_nodes, 2];
                node_batch_ids (LongTensor): node batch indices of shape [num_nodes].
        """

        # Unpack input graph dictionary
        con_feats = in_graph['con_feats']
        struc_feats = in_graph['struc_feats']
        edge_ids = in_graph['edge_ids']

        # Get useful graph properties
        num_nodes = len(con_feats)
        device = edge_ids.device

        # Get unnormalized edge scores
        pruned_edge_ids = edge_ids[:, edge_ids[1] > edge_ids[0]]
        edge_scores = self.edge_score(con_feats, edge_ids=pruned_edge_ids).squeeze(dim=1)
        edge_scores = F.relu(edge_scores, inplace=True)

        # Get different tensors with edge indices
        comp_edge_ids = pruned_edge_ids.flipud()
        self_edge_ids = torch.arange(num_nodes, device=device).unsqueeze(dim=0).expand(2, -1)

        group_edge_ids = pruned_edge_ids[:, edge_scores > 0]
        group_edge_ids = torch.cat([group_edge_ids, self_edge_ids], dim=1)

        # Get normalized edge scores
        edge_ids = torch.cat([pruned_edge_ids, comp_edge_ids, self_edge_ids], dim=1)
        edge_scores = torch.cat([edge_scores, edge_scores, torch.ones(num_nodes, device=device)], dim=0)

        sort_ids = torch.argsort(edge_ids[0] * num_nodes + edge_ids[1], dim=0)
        edge_ids = edge_ids[:, sort_ids]
        edge_scores = edge_scores[sort_ids]

        node_sums = scatter(edge_scores, edge_ids[0], dim=0, reduce='sum')
        edge_scores = edge_scores / node_sums[edge_ids[0]]

        # Get new content and structure features
        for _ in range(self.num_node_updates):
            con_feats = sparse_dense_mm(edge_ids, edge_scores, (num_nodes, num_nodes), con_feats)
            struc_feats = sparse_dense_mm(edge_ids, edge_scores, (num_nodes, num_nodes), struc_feats)

        # Aggregate new content and structure features
        group_ids = torch.arange(num_nodes, device=device)

        for _ in range(self.max_group_iters):
            old_group_ids = group_ids.clone()
            group_ids = scatter(group_ids[group_edge_ids[0]], group_edge_ids[1], dim=0, reduce='min')

            if torch.equal(old_group_ids, group_ids):
                break

        group_ids = torch.unique(group_ids, return_inverse=True)[1]
        con_feats = scatter(con_feats, group_ids, dim=0, reduce='mean')
        struc_feats = scatter(struc_feats, group_ids, dim=0, reduce='mean')

        # Construct output graph dictionary
        out_graph = {}
        out_graph['con_feats'] = con_feats
        out_graph['struc_feats'] = struc_feats

        return out_graph


@MODELS.register_module()
class NodeToEdge(nn.Module):
    """
    Class implementing the NodesToEdge module computing edge features from node source and target features.

    Attributes:
        reduction (str): String containing the reduction operation.
        implementation (str): String containing the type of implementation.
    """

    def __init__(self, reduction='mul', implementation='pytorch-custom'):
        """
        Initializes the NodeToEdge module.

        Args:
            reduction (str): String containing the reduction operation (default='mul').
            implementation (str): String containing the type of implementation (default='pytorch-custom').
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set reduction and implementation attributes
        self.reduction = reduction
        self.implementation = implementation

    def forward(self, node_src_feats, node_tgt_feats=None, edge_ids=None):
        """
        Forward method of the NodeToEdge module.

        Args:
            node_src_feats (FloatTensor): Node source features of shape [num_nodes, src_feat_size].
            node_tgt_feats (FloatTensor): Node target features of shape [num_nodes, tgt_feat_size] (default=None).
            edge_ids (LongTensor): Node indices for each (directed) edge of shape [2, num_edges] (default=None).

        Returns:
            edge_feats (FloatTensor): Tensor containing the edge features of shape [num_edges, edge_feat_size].

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

        # Get edge features
        edge_feats = node_to_edge(node_src_feats, node_tgt_feats, edge_ids, self.reduction, self.implementation)

        return edge_feats
