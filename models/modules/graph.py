"""
Collection of modules related to graphs.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
from torch_scatter import scatter
from torch_sparse import coalesce, spmm

from models.build import build_model, MODELS
from models.functional.graph import node_to_edge
from models.functional.sparse import sparse_dense_mm


@MODELS.register_module()
class GraphAttn(nn.Module):
    """
    Class implementing the GraphAttn module.

    Attributes:
        norm (nn.Module): Optional pre-normalization module.
        act_fn (nn.Module): Optional pre-activation module.

        struc_proj (nn.Linear): Optional module projecting structure features to input feature space.
        qry_proj (nn.Linear): Module projecting input features to query features.
        key_proj (nn.Linear): Module projecting input features to key features.
        val_proj (nn.Linear): Module projecting input features to value features.
        out_proj (nn.Linear): Module projecting weighted value features to output features.

        num_heads (int): Integer containing the number of attention heads.
        skip (bool): Boolean indicating whether skip connection is used or not.
    """

    def __init__(self, in_size, struc_size=None, norm='', act_fn='', qk_size=-1, val_size=-1, out_size=-1, num_heads=8,
                 skip=True):
        """
        Initializes the GraphAttn module.

        Args:
            in_size (int): Size of input features.
            struc_size (int): Size of input structure features (default=None).
            norm (str): String containing the type of pre-normalization module (default='').
            act_fn (str): String containing the type of pre-activation module (default='').
            qk_size (int): Size of query and key features (default=-1).
            val_size (int): Size of value features (default=-1).
            out_size (int): Size of output features (default=-1).
            num_heads (int): Integer containing the number of attention heads (default=8).
            skip (bool): Boolean indicating whether skip connection is used or not (default=True).

        Raises:
            ValueError: Error when unsupported type of normalization is provided.
            ValueError: Error when unsupported type of activation function is provided.
            ValueError: Error when the number of heads does not divide the query and key feature size.
            ValueError: Error when the number of heads does not divide the value feature size.
            ValueError: Error when input and output feature sizes are different when skip connection is used.
            ValueError: Error when the output feature size is not specified when no skip connection is used.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Initialization of optional normalization module
        if not norm:
            pass
        elif norm == 'layer':
            self.norm = nn.LayerNorm(in_size)
        else:
            error_msg = f"The GraphAttn module does not support the '{norm}' normalization type."
            raise ValueError(error_msg)

        # Initialization of optional module with activation function
        if not act_fn:
            pass
        elif act_fn == 'gelu':
            self.act_fn = nn.GELU()
        elif act_fn == 'relu':
            self.act_fn = nn.ReLU(inplace=False) if not norm and skip else nn.ReLU(inplace=True)
        else:
            error_msg = f"The GraphAttn module does not support the '{act_fn}' activation function."

        # Get and check query and key feature size
        qk_size = qk_size if qk_size != -1 else in_size

        if qk_size % num_heads != 0:
            error_msg = f"The number of heads ({num_heads}) must divide the query and key feature size ({qk_size})."
            raise ValueError(error_msg)

        # Get and check value feature size
        val_size = val_size if val_size != -1 else in_size

        if val_size % num_heads != 0:
            error_msg = f"The number of heads ({num_heads}) must divide the value feature size ({val_size})."
            raise ValueError(error_msg)

        # Get and check output feature size
        if skip and out_size == -1:
            out_size = in_size

        elif skip and in_size != out_size:
            error_msg = f"Input ({in_size}) and output ({out_size}) sizes must match when skip connection is used."
            raise ValueError(error_msg)

        elif not skip and out_size == -1:
            error_msg = "The output feature size must be specified when no skip connection is used."
            raise ValueError(error_msg)

        # Initialization of projection modules
        if struc_size is not None:
            self.struc_proj = nn.Linear(struc_size, in_size)

        self.qry_proj = nn.Linear(in_size, qk_size)
        xavier_uniform_(self.qry_proj.weight)
        zeros_(self.qry_proj.bias)

        self.key_proj = nn.Linear(in_size, qk_size)
        xavier_uniform_(self.key_proj.weight)
        zeros_(self.key_proj.bias)

        self.val_proj = nn.Linear(in_size, val_size)
        xavier_uniform_(self.val_proj.weight)
        zeros_(self.val_proj.bias)

        self.out_proj = nn.Linear(val_size, out_size, bias=False)
        zeros_(self.out_proj.weight)

        # Set additional attributes
        self.num_heads = num_heads
        self.skip = skip

    def forward(self, in_feats, edge_ids, edge_weights=None, struc_feats=None):
        """
        Forward method of the GraphAttn module.

        Args:
            in_feats (FloatTensor): Input node features of shape [num_nodes, in_size].
            edge_ids (LongTensor): Node indices for each (directed) edge of shape [2, num_edges].
            edge_weights (FloatTensor): Weights for each (directed) edge of shape [num_edges] (default=None).
            struc_feats (FloatTensor): Structure node features of shape [num_nodes, struc_size] (default=None).

        Returns:
            out_feats (FloatTensor): Output node features of shape [num_nodes, out_size].

        Raises:
            ValueError: Error when input and structure sizes are different without structure projection initialization.
        """

        # Get number of nodes
        num_nodes = len(in_feats)

        # Apply optional normalization and activation function modules
        delta_feats = in_feats
        delta_feats = self.norm(delta_feats) if hasattr(self, 'norm') else delta_feats
        delta_feats = self.act_fn(delta_feats) if hasattr(self, 'act_fn') else delta_feats

        # Get structure-enhanced delta features
        if struc_feats is not None:
            if hasattr(self, 'struc_proj'):
                delta_struc_feats = delta_feats + self.struc_proj(struc_feats)

            elif in_feats.size(dim=1) == struc_feats.size(dim=1):
                delta_struc_feats = delta_feats + struc_feats

            else:
                error_msg = "The input and structure feature sizes must be equal when no structure feature size was "
                error_msg += f"provided during initialization (got {in_feats.size(1)} and {struc_feats.size(1)})."
                raise ValueError(error_msg)

        # Get query, key and value features
        qry_feats = self.qry_proj(delta_struc_feats)
        key_feats = self.key_proj(delta_struc_feats)
        val_feats = self.val_proj(delta_feats)

        # Get attention weights
        qry_feats = qry_feats / math.sqrt(qry_feats.size(dim=1) // self.num_heads)
        attn_weights = node_to_edge(qry_feats, key_feats, edge_ids, reduction='mul-sum', num_groups=self.num_heads)
        attn_weights = torch.sigmoid(attn_weights)

        if edge_weights is not None:
            attn_weights = edge_weights[:, None] * attn_weights

        # Get weighted value features
        val_feats = sparse_dense_mm(edge_ids, attn_weights, (num_nodes, num_nodes), val_feats)

        # Get output features
        delta_feats = self.out_proj(val_feats)
        out_feats = in_feats + delta_feats if self.skip else delta_feats

        return out_feats


@MODELS.register_module()
class GraphProjector(nn.Module):
    """
    Class implementing the GraphProjector module.

    Attributes:
        con_proj (nn.Linear): Linear module projecting input content features to output content features.
        struc_proj (nn.Linear): Linear module projecting input structure features to output structure features.
    """

    def __init__(self, con_in_size, con_out_size, struc_in_size, struc_out_size):
        """
        Initializes the GraphProjector module.

        Args:
            con_in_size (int): Integer containing the input content feature size.
            con_out_size (int): Integer containing the output content feature size.
            struc_in_size (int): Integer containing the input structure feature size.
            struc_out_size (int): Integer containing the output structure feature size.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build linear projection modules
        self.con_proj = nn.Linear(con_in_size, con_out_size)
        self.struc_proj = nn.Linear(struc_in_size, struc_out_size)

    def forward(self, in_graph):
        """
        Forward method of the GraphProjector module.

        Args:
            in_graph (Dict): Input graph dictionary containing at least following keys:
                con_feats (FloatTensor): input content features of shape [num_nodes, con_in_size];
                struc_feats (FloatTensor): input structure features of shape [num_nodes, struc_in_size].

            out_graph (Dict): Output graph dictionary containing at least following keys:
                con_feats (FloatTensor): output content features of shape [num_nodes, con_out_size];
                struc_feats (FloatTensor): output structure features of shape [num_nodes, struc_out_size].
        """

        # Get output content and structure features
        out_graph = in_graph.copy()
        out_graph['con_feats'] = self.con_proj(in_graph['con_feats'])
        out_graph['struc_feats'] = self.struc_proj(in_graph['struc_feats'])

        return out_graph


@MODELS.register_module()
class GraphToGraph(nn.Module):
    """
    Class implementing the GraphToGraph module.

    Attributes:
        con_cross (nn.Module): Module implementing the content cross-update network.
        edge_score (nn.Module): Module implementing the edge score network.
        con_self (nn.Module): Module implementing the content self-update network.
        struc_self (nn.Module): Module implementing the structure self-update network.
        node_weight_iters (int): Number of iterations during node weight computation.
        max_group_iters (int): Maximum number of iterations during node grouping.
  """

    def __init__(self, con_cross_cfg, edge_score_cfg, con_self_cfg, struc_self_cfg, node_weight_iters=5,
                 max_group_iters=100):
        """
        Initializes the GraphToGraph module.

        Args:
            con_cross_cfg (Dict): Configuration dictionary specifying the content cross-update network.
            edge_score_cfg (Dict): Configuration dictionary specifying the edge score network.
            con_self_cfg (Dict): Configuration dictionary specifying the content self-update network.
            struc_self_cfg (Dict): Configuration dictionary specifying the structure self-update network.
            node_weight_iters (int): Number of iterations during node weight computation (default=5).
            max_group_iters (int): Maximum number of iterations during node grouping (default=100).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build sub-networks
        self.con_cross = build_model(con_cross_cfg)
        self.edge_score = build_model(edge_score_cfg)
        self.con_self = build_model(con_self_cfg)
        self.struc_self = build_model(struc_self_cfg)

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

        # Get useful graph properties
        num_nodes = len(con_feats)
        device = edge_ids.device

        # Update content features using neighboring features
        con_feats = self.con_cross(con_feats, edge_ids=edge_ids, edge_weights=edge_weights, struc_feats=struc_feats)

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

        con_feats = self.con_self(con_feats)
        struc_feats = self.struc_self(struc_feats)

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
