"""
Collection of modules related to graphs.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_sparse import coalesce, spmm

from models.build import build_model, MODELS
from models.functional.graph import node_to_edge


@MODELS.register_module()
class GraphToGraph(nn.Module):
    """
    Class implementing the GraphToGraph module.

    Attributes:
        con (nn.Module): Module implementing the content update network.
        struc (nn.Module): Module implementing the structure update network.
        edge_score (nn.Module): Module implementing the edge score network.
        node_weight_iters (int): Number of iterations during node weight computation.
        max_group_iters (int): Maximum number of iterations during node grouping.
  """

    def __init__(self, con_cfg, struc_cfg, edge_score_cfg, node_weight_iters=5, max_group_iters=100):
        """
        Initializes the GraphToGraph module.

        Args:
            con_cfg (Dict): Configuration dictionary specifying the content update network.
            struc_cfg (Dict): Configuration dictionary specifying the structure update network.
            edge_score_cfg (Dict): Configuration dictionary specifying the edge score network.
            node_weight_iters (int): Number of iterations during node weight computation (default=5).
            max_group_iters (int): Maximum number of iterations during node grouping (default=100).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build content, structure and edge score networks
        self.con = build_model(con_cfg)
        self.struc = build_model(struc_cfg)
        self.edge_score = build_model(edge_score_cfg)

        # Set additional attributes
        self.node_weight_iters = node_weight_iters
        self.max_group_iters = max_group_iters

    def forward(self, in_graph):
        """
        Forward method of the GraphToGraph module.

        Args:
            in_graph (Dict): Input graph dictionary containing at least following keys:
                con_feats (FloatTensor): content features of shape [num_in_nodes, con_feat_size];
                struc_feats (FloatTensor): structure features of shape [num_in_nodes, struc_feat_size];
                edge_ids (LongTensor): node indices for each (directed) edge of shape [2, num_in_edges];
                edge_weights (FloatTensor): weights for each (directed) edge of shape [num_in_edges];
                node_batch_ids (LongTensor): node batch indices of shape [num_in_nodes].

        Returns:
            out_graph (Dict): Output graph dictionary containing following keys:
                con_feats (FloatTensor): content features of shape [num_out_nodes, con_feat_size];
                struc_feats (FloatTensor): structure features of shape [num_out_nodes, struc_feat_size];
                edge_ids (LongTensor): node indices for each (directed) edge of shape [2, num_out_edges];
                edge_weights (FloatTensor): weights for each (directed) edge of shape [num_out_edges];
                node_batch_ids (LongTensor): node batch indices of shape [num_out_nodes];
                group_ids (LongTensor): group (i.e. output node) indices of input nodes of shape [num_in_nodes].
        """

        # Unpack input graph dictionary
        con_feats = in_graph['con_feats']
        struc_feats = in_graph['struc_feats']
        edge_ids = in_graph['edge_ids']
        edge_weights = in_graph['edge_weights']
        node_batch_ids = in_graph['node_batch_ids']

        # Update structure and content features
        struc_feats = self.struc(struc_feats, edge_ids=edge_ids, edge_weights=edge_weights)
        con_feats = self.con(con_feats, edge_ids=edge_ids, edge_weights=edge_weights, struc_feats=struc_feats)

        # Get useful graph properties
        num_nodes = len(con_feats)
        device = edge_ids.device

        # Get unnormalized edge scores
        pruned_edge_ids = edge_ids[:, edge_ids[1] > edge_ids[0]]
        edge_scores = self.edge_score(con_feats, edge_ids=pruned_edge_ids).squeeze(dim=1)
        edge_scores = F.relu(edge_scores, inplace=True)

        # Get normalized edge scores
        comp_edge_ids = pruned_edge_ids.flipud()
        self_edge_ids = torch.arange(num_nodes, device=device).unsqueeze(dim=0).expand(2, -1)

        edge_ids = torch.cat([pruned_edge_ids, comp_edge_ids, self_edge_ids], dim=1)
        edge_scores = torch.cat([edge_scores, edge_scores, torch.ones(num_nodes, device=device)], dim=0)

        sort_ids = torch.argsort(edge_ids[0] + num_nodes * edge_ids[1], dim=0)
        edge_ids = edge_ids[:, sort_ids]
        edge_scores = edge_scores[sort_ids]

        node_sums = scatter(edge_scores, edge_ids[1], dim=0, reduce='sum')
        edge_scores = edge_scores / node_sums[edge_ids[1]]

        # Get unnormalized node weights
        node_weights = torch.ones(num_nodes, 1, device=device)

        for _ in range(self.node_weight_iters):
            node_weights = spmm(edge_ids, edge_scores, num_nodes, num_nodes, node_weights)

        # Get group indices, group sizes and number of groups
        group_edge_ids = edge_ids[:, edge_scores > 0]
        group_ids = torch.arange(num_nodes, device=device)

        for _ in range(self.max_group_iters):
            old_group_ids = group_ids.clone()
            group_ids = scatter(group_ids[group_edge_ids[0]], group_edge_ids[1], dim=0, reduce='min')

            if torch.equal(old_group_ids, group_ids):
                break

        group_ids, group_sizes = torch.unique(group_ids, return_inverse=True, return_counts=True)[1:]
        num_groups = len(group_sizes)

        # Get normalized node weights
        node_weights = node_weights / group_sizes[group_ids, None]

        # Get new content and structure features
        con_feats = scatter(node_weights * con_feats, group_ids, dim=0, reduce='sum')
        struc_feats = scatter(node_weights * struc_feats, group_ids, dim=0, reduce='sum')

        # Get new edge indices and edge weights
        edge_weights = node_weights[edge_ids[0]].squeeze(dim=1) * edge_weights
        edge_ids, edge_weights = coalesce(group_ids[edge_ids], edge_weights, num_groups, num_groups, op='add')

        # Get new node batch indices
        node_batch_ids = scatter(node_batch_ids, group_ids, dim=0, reduce='min')

        # Construct output graph dictionary
        out_graph = {}
        out_graph['con_feats'] = con_feats
        out_graph['struc_feats'] = struc_feats
        out_graph['edge_ids'] = edge_ids
        out_graph['edge_weights'] = edge_weights
        out_graph['node_batch_ids'] = node_batch_ids
        out_graph['group_ids'] = group_ids

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
