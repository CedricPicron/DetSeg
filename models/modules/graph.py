"""
Collection of modules related to graphs.
"""

from torch import nn

from models.build import build_model, MODELS
from models.functional.graph import node_to_edge


@MODELS.register_module()
class GraphToGraph(nn.Module):
    """
    Class implementing the GraphToGraph module.

    Attributes:
        edge_score (nn.Module): Module implementing the edge score network.
  """

    def __init__(self, edge_score_cfg):
        """
        Initializes the GraphToGraph module.

        Args:
            edge_score_cfg (Dict): Configuration dictionary specifying the edge score network.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build edge score network
        self.edge_score = build_model(edge_score_cfg)

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
        edge_ids = in_graph['edge_ids']

        # Compute edge scores
        pruned_edge_ids = edge_ids[:, edge_ids[1] > edge_ids[0]]
        edge_scores = self.edge_score(con_feats, edge_ids=pruned_edge_ids)

        return edge_scores


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
