"""
Collection of modules related to graphs.
"""
import math

import torch
from torch import nn
from torch.nn.init import xavier_uniform_, zeros_
from torch_scatter import scatter, scatter_max, scatter_mean, scatter_min
from torch_sparse import coalesce

from models.build import build_model, MODELS
from models.functional.activation import custom_step
from models.functional.graph import node_to_edge
from models.functional.sparse import sparse_dense_mm
from models.functional.utils import custom_ones


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
        pos_proj (nn.Linear): Module projecting relative node locations to relative position features.
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

        self.pos_proj = nn.Linear(2, qk_size)
        xavier_uniform_(self.pos_proj.weight)
        zeros_(self.pos_proj.bias)

        self.out_proj = nn.Linear(val_size, out_size, bias=False)
        zeros_(self.out_proj.weight)

        # Set additional attributes
        self.num_heads = num_heads
        self.skip = skip

    def forward(self, in_feats, edge_ids, edge_weights=None, node_cxcy=None, struc_feats=None):
        """
        Forward method of the GraphAttn module.

        Args:
            in_feats (FloatTensor): Input node features of shape [num_nodes, in_size].
            edge_ids (LongTensor): Node indices for each (directed) edge of shape [2, num_edges].
            struc_feats (FloatTensor): Structure node features of shape [num_nodes, struc_size] (default=None).
            node_cxcy (FloatTensor): Normalized node center locations of shape [num_nodes, 2] (default=None).
            edge_weights (FloatTensor): Weights for each (directed) edge of shape [num_edges] (default=None).

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

        # Get query, key and value features
        qry_feats = key_feats = val_feats = delta_feats

        if struc_feats is not None:
            if hasattr(self, 'struc_proj'):
                qry_feats = key_feats = delta_feats + self.struc_proj(struc_feats)

            elif in_feats.size(dim=1) == struc_feats.size(dim=1):
                qry_feats = key_feats = delta_feats + struc_feats

            else:
                error_msg = "The input and structure feature sizes must be equal when no structure feature size was "
                error_msg += f"provided during initialization (got {in_feats.size(1)} and {struc_feats.size(1)})."
                raise ValueError(error_msg)

        qry_feats = self.qry_proj(qry_feats)
        key_feats = self.key_proj(key_feats)
        val_feats = self.val_proj(val_feats)

        # Get relative position features
        qry_cxcy = node_cxcy[edge_ids[0]]
        tgt_cxcy = node_cxcy[edge_ids[1]]
        pos_feats = self.pos_proj(tgt_cxcy - qry_cxcy)

        # Get attention weights
        qry_feats = qry_feats / math.sqrt(qry_feats.size(dim=1) // self.num_heads)
        node_to_edge_kwargs = {'reduction': 'mul-sum', 'num_groups': self.num_heads, 'off_edge_tgt': pos_feats}
        attn_weights = node_to_edge(qry_feats, key_feats, edge_ids, **node_to_edge_kwargs)
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
    """

    def __init__(self, con_in_size, con_out_size):
        """
        Initializes the GraphProjector module.

        Args:
            con_in_size (int): Integer containing the input content feature size.
            con_out_size (int): Integer containing the output content feature size.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build linear content projection module
        self.con_proj = nn.Linear(con_in_size, con_out_size)

    def forward(self, in_graph):
        """
        Forward method of the GraphProjector module.

        Args:
            in_graph (Dict): Input graph dictionary containing at least following key:
                con_feats (FloatTensor): input content features of shape [num_nodes, con_in_size];

            out_graph (Dict): Output graph dictionary containing at least following key:
                con_feats (FloatTensor): output content features of shape [num_nodes, con_out_size];
        """

        # Get output content features
        out_graph = in_graph.copy()
        out_graph['con_feats'] = self.con_proj(in_graph['con_feats'])

        return out_graph


@MODELS.register_module()
class GraphToGraph(nn.Module):
    """
    Class implementing the GraphToGraph module.

    Attributes:
        con_self (nn.Module): Module implementing the content self-update network.
        con_cross (nn.Module): Module implementing the content cross-update network.
        edge_score (nn.Module): Module implementing the edge score network.
        con_cxcy (nn.Module): Module implementing the content cxcy-update network.
        struc_cxcy (nn.Module): Module implementing the structure cxcy-update network.
        left_zero_grad_thr (float): Left zero gradient threshold of the custom step function.
        right_zero_grad_thr (float): Right zero gradient threshold of the custom step function.
        max_group_iters (int): Maximum number of iterations during node grouping.
        con_agg_type (str): String containing the content aggregation type.
        struc_agg_type (str): String containing the structure aggregation type.
        con_weight (nn.Module): Optional module specifying the content weight network.
        struc_weight (nn.Module): Optional module specifying the structure weight network.
  """

    def __init__(self, con_self_cfg, con_cross_cfg, edge_score_cfg, con_cxcy_cfg, struc_cxcy_cfg,
                 left_zero_grad_thr=-0.1, right_zero_grad_thr=0.1, max_group_iters=100, con_agg_type='mean',
                 struc_agg_type='mean', con_weight_cfg=None, struc_weight_cfg=None):
        """
        Initializes the GraphToGraph module.

        Args:
            con_self_cfg (Dict): Configuration dictionary specifying the content self-update network.
            con_cross_cfg (Dict): Configuration dictionary specifying the content cross-update network.
            edge_score_cfg (Dict): Configuration dictionary specifying the edge score network.
            con_cxcy_cfg (Dict): Configuration dicationary specifying the content cxcy-update network.
            struc_cxcy_cfg (Dict): Configuration dictionary specifying the structure cxcy-update network.
            left_zero_grad_thr (float): Left zero gradient threshold of the custom step function (default=-0.1).
            right_zero_grad_thr (float): Right zero gradient threshold of the custom step function (default=0.1).
            max_group_iters (int): Maximum number of iterations during node grouping (default=100).
            con_agg_type (str): String containing the content aggregation type (default='mean').
            struc_agg_type (str): String containing the structure aggregation type (defaul='mean').
            con_weight_cfg (Dict): Configuration dictionary specifying the content weight network (default=None).
            struc_weight_cfg (Dict): Configuration dictionary specifying the structure weight network (default=None).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build mandatory sub-networks
        self.con_self = build_model(con_self_cfg)
        self.con_cross = build_model(con_cross_cfg)
        self.edge_score = build_model(edge_score_cfg)
        self.con_cxcy = build_model(con_cxcy_cfg)
        self.struc_cxcy = build_model(struc_cxcy_cfg)

        # Set additional attributes
        self.left_zero_grad_thr = left_zero_grad_thr
        self.right_zero_grad_thr = right_zero_grad_thr
        self.max_group_iters = max_group_iters
        self.con_agg_type = con_agg_type
        self.struc_agg_type = struc_agg_type

        # Build optional sub-networks
        if con_agg_type == 'weighted_sum':
            if con_weight_cfg is not None:
                self.con_weight = build_model(con_weight_cfg)
            else:
                error_msg = "A content weight configuration dictionary must be specified when using the "
                error_msg += "'weighted_sum' content aggregation type."
                raise ValueError(error_msg)

        if struc_agg_type == 'weighted_sum':
            if struc_weight_cfg is not None:
                self.struc_weight = build_model(struc_weight_cfg)
            else:
                error_msg = "A structure weight configuration dictionary must be specified when using the "
                error_msg += "'weighted_sum' structure aggregation type."
                raise ValueError(error_msg)

    def forward(self, in_graph):
        """
        Forward method of the GraphToGraph module.

        Args:
            in_graph (Dict): Input graph dictionary containing following keys:
                graph_id (int): integer containing the graph index;
                con_feats (FloatTensor): content features of shape [num_in_nodes, con_feat_size];
                dyn_struc_feats (FloatTensor): dynamic structure features of shape [num_in_nodes, struc_feat_size];
                sta_struc_feats (FloatTensor): static structure features of shape [num_in_nodes, struc_feat_size];
                edge_ids (LongTensor): node indices for each (directed) edge of shape [2, num_in_edges];
                edge_weights (FloatTensor): weights for each (directed) edge of shape [num_in_edges];
                node_cxcy (FloatTensor): node center locations in normalized (x, y) format of shape [num_in_nodes, 2];
                node_masses (FloatTensor): node masses of shape [num_in_nodes];
                node_batch_ids (LongTensor): node batch indices of shape [num_in_nodes];
                seg_maps (LongTensor): graph-based segmentation maps of shape [batch_size, fH, fW];
                analysis_dict (Dict): dictionary of different analyses used for logging purposes only.

        Returns:
            out_graph (Dict): Output graph dictionary containing following keys:
                graph_id (int): integer containing the graph index;
                con_feats (FloatTensor): content features of shape [num_out_nodes, con_feat_size];
                dyn_struc_feats (FloatTensor): dynamic structure features of shape [num_out_nodes, struc_feat_size];
                sta_struc_feats (FloatTensor): static structure features of shape [num_out_nodes, struc_feat_size];
                edge_ids (LongTensor): node indices for each (directed) edge of shape [2, num_out_edges];
                edge_weights (FloatTensor): weights for each (directed) edge of shape [num_out_edges];
                node_cxcy (FloatTensor): node center locations in normalized (x, y) format of shape [num_out_nodes, 2];
                node_masses (FloatTensor): node masses of shape [num_out_nodes];
                node_batch_ids (LongTensor): node batch indices of shape [num_out_nodes];
                seg_maps (LongTensor): graph-based segmentation maps of shape [batch_size, fH, fW];
                analysis_dict (Dict): dictionary of different analyses used for logging purposes only.

        Raises:
            ValueError: Error when an invalid content aggregation type is provided.
            ValueError: Error when an invalid structure aggregation type is provided.
        """

        # Unpack input graph dictionary
        graph_id = in_graph['graph_id']
        con_feats = in_graph['con_feats']
        dyn_struc_feats = in_graph['dyn_struc_feats']
        sta_struc_feats = in_graph['sta_struc_feats']
        edge_ids = in_graph['edge_ids']
        edge_weights = in_graph['edge_weights']
        in_node_cxcy = in_graph['node_cxcy']
        in_node_masses = in_graph['node_masses']
        node_batch_ids = in_graph['node_batch_ids']
        seg_maps = in_graph['seg_maps']
        analysis_dict = in_graph['analysis_dict']

        # Get useful graph properties
        num_nodes = len(con_feats)
        device = edge_ids.device

        # Update content features based on own content features
        con_feats = self.con_self(con_feats)

        # Update content features using neighboring features
        cross_kwargs = {'struc_feats': dyn_struc_feats.detach(), 'node_cxcy': in_node_cxcy}
        cross_kwargs = {**cross_kwargs, 'edge_weights': edge_weights}
        con_feats = self.con_cross(con_feats, edge_ids, **cross_kwargs)

        # Get edge scores
        pruned_edge_ids = edge_ids[:, edge_ids[1] > edge_ids[0]]
        edge_scores = self.edge_score(con_feats, edge_ids=pruned_edge_ids).squeeze(dim=1)

        step_kwargs = {'left_zero_grad_thr': self.left_zero_grad_thr, 'right_zero_grad_thr': self.right_zero_grad_thr}
        edge_scores = custom_step(edge_scores, **step_kwargs)

        comp_edge_ids = pruned_edge_ids.flipud()
        self_edge_ids = torch.arange(num_nodes, device=device).unsqueeze(dim=0).expand(2, -1)

        edge_ids = torch.cat([pruned_edge_ids, comp_edge_ids, self_edge_ids], dim=1)
        edge_scores = torch.cat([edge_scores, edge_scores, torch.ones(num_nodes, device=device)], dim=0)

        sort_ids = torch.argsort(edge_ids[0] * num_nodes + edge_ids[1], dim=0)
        edge_ids = edge_ids[:, sort_ids]
        edge_scores = edge_scores[sort_ids]

        # Get group indices, group sizes and number of groups
        group_edge_ids = edge_ids[:, edge_scores > 0]
        group_ids = torch.arange(num_nodes, device=device)

        for iter_id in range(self.max_group_iters):
            old_group_ids = group_ids.clone()
            group_ids = scatter(group_ids[group_edge_ids[0]], group_edge_ids[1], dim=0, reduce='min')

            if torch.equal(old_group_ids, group_ids):
                break

        group_ids, group_sizes = torch.unique(group_ids, return_inverse=True, return_counts=True)[1:]
        num_groups = len(group_sizes)

        analysis_dict[f'group_iters_{graph_id+1}'] = iter_id + 1
        analysis_dict[f'num_nodes_{graph_id+1}'] = num_groups

        # Get new node center locations and masses
        out_node_masses = scatter(in_node_masses, group_ids, dim=0, reduce='sum')
        node_weights = in_node_masses / out_node_masses[group_ids]
        out_node_cxcy = scatter(node_weights[:, None] * in_node_cxcy, group_ids, dim=0, reduce='sum')

        # Update content and structure features based on new center location
        delta_cxcy = out_node_cxcy[group_ids] - in_node_cxcy
        delta_con_feats = self.con_cxcy(delta_cxcy)
        delta_struc_feats = self.struc_cxcy(delta_cxcy)

        con_feats = con_feats + delta_con_feats
        dyn_struc_feats = dyn_struc_feats + delta_struc_feats.detach()
        sta_struc_feats = sta_struc_feats + delta_struc_feats

        # Get node scores
        new_edge_ids = group_ids[edge_ids]
        same_group = new_edge_ids[0] == new_edge_ids[1]
        node_scores = scatter(edge_scores[same_group], edge_ids[0, same_group], dim=0, reduce='sum')
        node_scores = custom_ones(node_scores)

        # Get aggregated content features
        if self.con_agg_type == 'weighted_sum':
            con_weights = self.con_weight(con_feats)
            con_sums = scatter(con_weights, group_ids, dim=0, reduce='sum')
            con_weights = con_weights / con_sums[group_ids, :]
            con_feats = scatter(con_weights * con_feats, group_ids, dim=0, reduce='sum')

        elif self.con_agg_type in ('max', 'mean', 'min'):
            con_feats = scatter(con_feats, group_ids, dim=0, reduce=self.con_agg_type)

        else:
            error_msg = f"Invalid content aggregation type (got {self.con_agg_type})."
            raise ValueError(error_msg)

        # Get aggregated static structure features
        if self.struc_agg_type == 'weighted_sum':
            struc_weights = self.struc_weight(sta_struc_feats)
            struc_sums = scatter(struc_weights, group_ids, dim=0, reduce='sum')
            struc_weights = struc_weights / struc_sums[group_ids, :]
            sta_struc_feats = scatter(struc_weights * sta_struc_feats, group_ids, dim=0, reduce='sum')

        elif self.struc_agg_type == 'max':
            sta_struc_feats, max_ids = scatter_max(sta_struc_feats, group_ids, dim=0)
            struc_weights = torch.zeros(num_nodes, max_ids.size(dim=1), device=device)
            struc_weights.scatter_(dim=0, index=max_ids, src=torch.ones(*max_ids.size(), device=device))

        elif self.struc_agg_type == 'mean':
            sta_struc_feats = scatter_mean(sta_struc_feats, group_ids, dim=0)
            struc_weights = torch.ones(num_nodes, device=device) / group_sizes[group_ids]
            struc_weights = struc_weights[:, None]

        elif self.struc_agg_type == 'min':
            sta_struc_feats, min_ids = scatter_min(sta_struc_feats, group_ids, dim=0)
            struc_weights = torch.zeros(num_nodes, min_ids.size(dim=1), device=device)
            struc_weights.scatter_(dim=0, index=min_ids, src=torch.ones(*min_ids.size(), device=device))

        else:
            error_msg = f"Invalid structure aggregation type (got {self.struc_agg_type})."
            raise ValueError(error_msg)

        # Get aggregated dynamic structure features (part 1)
        dyn_struc_weights = node_scores[:, None] * struc_weights.detach()
        dyn_struc_feats = scatter(dyn_struc_weights * dyn_struc_feats, group_ids, dim=0, reduce='sum')

        # Get aggregated dynamic structure features (part 2)

        # Get new (coalesced) edge indices and edge weights
        edge_ids, edge_weights = coalesce(new_edge_ids, edge_weights, num_groups, num_groups, op='add')

        # Get new node batch indices
        node_batch_ids = scatter(node_batch_ids, group_ids, dim=0, reduce='min')

        # Get new graph-based segmentation maps
        seg_maps = group_ids[seg_maps]

        # Construct output graph dictionary
        out_graph = {}
        out_graph['graph_id'] = graph_id + 1
        out_graph['con_feats'] = con_feats
        out_graph['dyn_struc_feats'] = dyn_struc_feats
        out_graph['sta_struc_feats'] = sta_struc_feats
        out_graph['edge_ids'] = edge_ids
        out_graph['edge_weights'] = edge_weights
        out_graph['node_cxcy'] = out_node_cxcy
        out_graph['node_masses'] = out_node_masses
        out_graph['node_batch_ids'] = node_batch_ids
        out_graph['seg_maps'] = seg_maps
        out_graph['analysis_dict'] = analysis_dict

        return out_graph


@MODELS.register_module()
class NodeToEdge(nn.Module):
    """
    Class implementing the NodesToEdge module computing edge features from node source and target features.

    Attributes:
        reduction (str): String containing the reduction operation.
        num_groups (int): Integer containing the number of groups during 'mul-sum' reduction.
        implementation (str): String containing the type of implementation.
    """

    def __init__(self, reduction='mul', num_groups=1, implementation='pytorch-custom'):
        """
        Initializes the NodeToEdge module.

        Args:
            reduction (str): String containing the reduction operation (default='mul').
            num_groups (int): Integer containing the number of groups during 'mul-sum' reduction (default=1).
            implementation (str): String containing the type of implementation (default='pytorch-custom').
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attributes
        self.reduction = reduction
        self.num_groups = num_groups
        self.implementation = implementation

    def forward(self, node_src_feats, node_tgt_feats=None, edge_ids=None, off_edge_src=None, off_edge_tgt=None):
        """
        Forward method of the NodeToEdge module.

        Args:
            node_src_feats (FloatTensor): Node source features of shape [num_nodes, src_feat_size].
            node_tgt_feats (FloatTensor): Node target features of shape [num_nodes, tgt_feat_size] (default=None).
            edge_ids (LongTensor): Node indices for each (directed) edge of shape [2, num_edges] (default=None).
            off_edge_src (FloatTensor): Offset edge source features of shape [num_edges, src_feat_size] (default=None).
            off_edge_tgt (FloatTensor): Offset edge target features of shape [num_edges, tgt_feat_size] (default=None).

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
        attrs = {'reduction': self.reduction, 'num_groups': self.num_groups, 'implementation': self.implementation}
        edge_feats = node_to_edge(node_src_feats, node_tgt_feats, edge_ids, off_edge_src, off_edge_tgt, **attrs)

        return edge_feats
