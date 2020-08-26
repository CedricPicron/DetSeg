"""
Transformer modules and build function.
"""
import copy
import sys
from typing import Optional

import numpy as np
import torch
from torch import nn, Tensor
import torch.autograd.Function as Function
import torch.nn.functional as F


class Transformer(nn.Module):
    """
    Class implementing the Transformer module.

    Attributes:
        feature_dim (int): Feature size.
        num_heads (int): Number of attention heads.
        encoder (TransformerEncoder): Global multi-head self-attention encoder.
    """

    def __init__(self, num_encoder_layers=6, num_group_layers=6,
                 feature_dim=256, num_heads=8, mha_dropout=0.1,
                 ffn_hidden_dim=2048, ffn_dropout=0.1,
                 return_intermediate=False):
        """
        Initializes the Transformer module.

        Args:
            num_encoder_layers (int): Number of encoder layers.
            num_group_layers (int): Mumber of group layers.
            feature_dim (int): Feature size.
            num_heads (int): Number of attention heads.
            mha_dropout (float): Dropout used during multi-head attention (MHA).
            ffn_hidden_dim (int): Hidden dimension of feedforward network (FFN).
            ffn_dropout (float): Dropout used during feedforward network (FFN).
            return_intermediate (bool): Return intermediate group layer outputs.
        """

        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads

        encoder_layer = TransformerEncoderLayer(feature_dim, num_heads, mha_dropout, ffn_hidden_dim, ffn_dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets module parameters to random initial values.

        All multi-dimensional parameters are being reset following the uniform xavier initialization.
        """

        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, in_features, padding_mask, pos_encoding):
        """
        Forward method of Transformer module.

        Args:
            in_features (Tensor): Input features of shape [batch_size x feature_dim x H x W].
            padding_mask (Tensor): Boolean mask with padding information of shape [batch_size x H x W].
            pos_encoding (Tensor): Position encoding of shape [batch_size x feature_dim x H x W].

        Returns:
            out_features (Tensor): Output features of shape [batch_size x feature_dim x H x W].
        """

        # flatten from NxCxHxW to HWxNxC
        in_features = in_features.flatten(2).permute(2, 0, 1)
        pos_encoding = pos_encoding.flatten(2).permute(2, 0, 1)

        # flatten from NxHxW to NxHW
        padding_mask = padding_mask.flatten(1)

        # compute encoder features with shape HWxNxC
        encoder_features = self.encoder(in_features, feature_mask=padding_mask, pos=pos_encoding)

        # out_features = None.transpose(1, 2)

        return encoder_features


class TransformerEncoder(nn.Module):
    """
    Concatenation of encoder layers.

    Attributes:
        layers (nn.ModulesList): List of encoder layers being concatenated.
        num_layers (int): Number of concatenated encoder layers.
    """

    def __init__(self, encoder_layer, num_layers):
        """
        Initializes the TransformerEncoder module.

        Args:
            encoder_layer (TransformerEncoderLayer): Encoder layer module to be concatenated.
            num_layers (int): Number of concatenated encoder layers.
        """

        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, in_features: Tensor, feature_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None):
        """
        Forward method of TransformerEncoder module.

        Args:
            in_features (Tensor): Input features of shape [num_features x batch_size x feature_dim].
            feature_mask (Tensor): Boolean mask encoding inactive features of shape [batch_size, num_features].
            pos (Tensor): Position encoding of shape [num_features x batch_size x feature_dim].

        Returns:
            out_features (Tensor): Output features of shape [num_features x batch_size x feature_dim].
        """

        out_features = in_features
        for layer in self.layers:
            out_features = layer(out_features, src_mask=feature_mask, pos=pos)

        return out_features


class TransformerEncoderLayer(nn.Module):
    """
    Encoder layer with global multi-head self-attention and FFN.

    Attributes:
        self_attn (nn.MultiheadAtttenion): Multi-head attetion (MHA) module.
        drop1 (nn.Dropout): Dropout module after global MHA.
        norm1 (nn.LayerNorm): Layernorm after MHA skip connection.
        linear1 (nn.Linear): First FFN linear layer.
        dropout (nn.Dropout): Dropout module after first FFN layer.
        linear2 (nn.Linear): Second FFN linear layer.
        dropout2 (nn.Dropout): Dropout module after second FFN layer.
        norm2 (nn.LayerNorm): Layernorm after FFN skip connection.
    """

    def __init__(self, feature_dim, num_heads, mha_dropout=0.1, ffn_hidden_dim=2048, ffn_dropout=0.1):
        super().__init__()

        # Implementation of global multi-head self-attention
        self.self_attn = nn.MultiheadAttention(feature_dim, num_heads, dropout=mha_dropout)
        self.dropout1 = nn.Dropout(mha_dropout)
        self.norm1 = nn.LayerNorm(feature_dim)

        # Implementation of feedforward network (FFN)
        self.linear1 = nn.Linear(feature_dim, ffn_hidden_dim)
        self.dropout = nn.Dropout(ffn_dropout)
        self.linear2 = nn.Linear(ffn_hidden_dim, feature_dim)
        self.dropout2 = nn.Dropout(ffn_dropout)
        self.norm2 = nn.LayerNorm(feature_dim)

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src, src_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None):
        # Global multi-head self-attention with position encoding
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, key_padding_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward network (FFN)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class GroupDecoder(nn.Module):
    """
    Concatenation of group decoder layers.

    Attributes:
        layers (nn.ModulesList): List of group decoder layers being concatenated.
        num_layers (int): Number of concatenated group decoder layers.

    * Initialization (first iteration and layer):
        in: encoder_features -> shape = [H*W, batch_size, feature_dim]

        feat_idx_edge_idx = init_fe(H, W, batch_size) -> shape = [num_features, 4]
        edge_idx_feat_idx = init_ef(H, W, batch_size) -> shape = [num_edges, 2]

        out: features = encoder_features.view(-1, feature_dim) -> shape = [H*W*batch_size, feat_dim]
        out: graph = Graph(feat_idx_edge_idx, edge_idxe_feat_idx) -> create Graph instance

        note: num_feat = H*W*batch_size at first iteration
        note: num_edges = (2*H*W - H - W) * batch_size at first iteration
    """


class GroupDecoderLayer(nn.Module):
    """
    Submodules:
        1) Grouping
        2) Cross-attention
        3) Self-attention
        4) FFN
    """


class Graph():
    """
    Class implementing the Graph object used for the feature graphs.
    """

    def __init__(self, height, width):
        height, width = np.array(height), np.array(width)
        batch_size = len(height)

        self.num_nodes = height * width
        self.num_edges = 2*height*width - height - width
        self.num_nodes_total = sum(self.num_nodes)
        self.num_edges_total = sum(self.num_edges)

        self.node_to_node = self.init_node_to_node(height, width, batch_size)
        self.node_to_edge = self.init_node_to_edge(height, width, batch_size)
        self.edge_to_node = self.init_edge_to_node(height, width, batch_size)
        self.edge_to_edge = self.init_edge_to_edge(height, width, batch_size)

    def init_node_to_node(self, height, width, bs):
        base = torch.arange(self.num_nodes_total).unsqueeze(1)
        height = torch.cat([height[i] * torch.ones(self.num_nodes[i]) for i in range(bs)], dim=0)
        node_to_node = torch.cat((base-1, base-height, base+1, base+height), dim=1)

        offset = np.concatenate(([0], np.cumsum(self.num_nodes)[:-1]), axis=0)
        fill_idx_left = torch.cat([width[i] * torch.arange(height[i]) for i in range(bs)], dim=0)
        fill_idx_top = torch.cat([offset[i] + torch.arange(width[i]) for i in range(bs)], dim=0)
        fill_idx_right = width[0]-1 + fill_idx_left
        fill_idx_bottom = offset[1]-width[0] + fill_idx_top

        node_fill_idx = self.num_nodes_total
        node_to_node[fill_idx_left, 0] = node_fill_idx
        node_to_node[fill_idx_top, 1] = node_fill_idx
        node_to_node[fill_idx_right, 2] = node_fill_idx
        node_to_node[fill_idx_bottom, 3] = node_fill_idx

        return node_to_node

    def init_node_to_edge(self, height, width, bs):
        return

    def init_edge_to_node(self, height, width, bs):
        return

    def init_edge_to_edge(self, height, width, bs):
        return

    def get_groups(self, mask):
        return

    def add_neighbor_edges(self, edge_idx):
        return

    def get_group_edge_feat(self, edge_idx):
        return

    def get_num_feat_per_group(self, edge_idx):
        return

    def get_feat_idx(self, edge_idx):
        return

    def update(self, edge_idx):
        return


class HardWeightGate(Function):

    @staticmethod
    def forward(ctx, weights, grp_num_feat):
        dtype = weights.dtype
        weights = torch.cumsum(1/grp_num_feat[:, None], dim=1) <= 1
        weights = weights.to(dtype)

        return weights

    @staticmethod
    def backward(ctx, grad_weights):
        return grad_weights, None


class Grouper(nn.Module):
    """
    * Grouping:
        ** Projection:
            in: features -> shape = [num_feat, feat_dim]
            param: P = projection matrix -> shape = [feat_dim, proj_dim]
            out: proj_features = mm(feature, P) -> shape = [num_feat, proj_dim]

        ** Pair comparison:
            in: proj_features -> shape = [num_feat, proj_dim]
            in: graph -> Graph object
            param: similarity_threshold -> float in [0, 1]

            pair = proj_features[graph.edge_idx_feat_idx, :]-> shape = [num_edges, 2, proj_dim]
            similarities = bmm(pair[:, 0, None, :], pair[:, 1, :, None]).squeeze() -> shape = [num_edges]

            edge_logits = torch.full([num_edges+1], fill=-sys.float_info.max, dtype=torch.float)
            out: edge_logits[:-1] = sigmoid(similarities - similarity_threshold) -> shape = [num_edges+1]

        ** Get soft grouping weights:
            in: edge_logits -> shape = [num_edges+1]
            in: features -> shape = [num_feat, feat_dim]
            in: graph -> Graph instance

            mask = edge_logits > 0.5 -> shape = [num_edges+1]
            grp_edge_idx = graph.get_groups(mask) -> shape = [num_groups, max_edges]
            grp_edge_idx_plus = graph.add_neighbor_edges(grp_edge_idx) -> shape = [num_groups, max_edges_plus]
            note: use fill=num_edges for both index tensors

            grp_edge_logits = edge_logits[grp_edge_idx_plus] -> shape = [num_groups, max_edges_plus]
            grp_edge_weights = softmax(grp_edge_logits, dim=-1) -> shape = [num_groups, max_edges_plus]

            grp_edge_feat = graph.get(grp_edge_idx_plus) -> shape = [num_groups, max_edges_plus, max_feat_plus]
            grp_feat_weights = 0.5 * bmm(grp_edge_weights[:, None, :], grp_edge_feat).squeeze()

            out: grp_feat_weights -> shape = [num_groups, max_feat_plus]
            out: grp_num_feat = graph.get_num_feat(grp_edge_idx) -> shape = [num_groups]

        ** Get hard grouping weights:
            Forward:
                in: grp_feat_weights -> shape = [num_groups, max_feat_plus]
                in: grp_num_feat -> shape = [num_groups]

                grp_feat_weights[:] = 1/grp_num_feat[:, None] -> shape = [num_groups, max_feat_plus]
                grp_mask = cumsum(grp_feat_weights, dim=1) > 1 -> shape = [num_groups, max_feat_plus]
                out: grp_feat_weights = grp_feat_weights[grp_mask] -> shape = [num_groups, max_feat_plus]

            Backward:
                in: grad_grp_feat_weights -> shape = [num_groups, max_feat_plus]
                out: grad_grp_feat_weights -> shape = [num_groups, max_feat_plus]

        ** Compute group means (new features):
            in: graph -> Graph instance
            in: grp_edge_idx_plus -> shape = [num_groups, max_edges_plus]
            in: grp_feat_weights -> shape = [num_groups, max_feat_plus]

            grp_feat_idx_plus = graph.get_feat_idx(grp_edge_idx_plus) -> shape = [num_groups, max_feat_plus]
            note: use fill=0 for simplicity (all integers between in [0, max_feature_plus-1] are allowed
            grp_feat = features[grp_feat_idx_plus, :] -> shape = [num_groups, max_feat_plus, feat_dim]

            out: grp_means = bmm(grp_feat_weights[:, None, :], grp_feat).squeeze() -> shape = [num_groups, feat_dim]
            out: graph.update(grp_edge_idx) -> update graph.{feat_idx_edge_idx, edge_idx_feat_idx}
    """

    def __init__(self, feat_dim=256, proj_dim=128, similarity_threshold=0.8):
        """
        Initializes the Grouper module.

        Args:
            feat_dim (int): Feature size.
            proj_dim (int): Size of projected features, used for comparison.
            similarity_threshold (float): Minimum similarity required for grouping.
        """

        super.__init__()

        self.projector = nn.Linear(feat_dim, proj_dim, bias=False)
        self.similarity_threshold = similarity_threshold
        self.get_hard_weights = HardWeightGate.apply

    def forward(self, in_features, graph):
        """
        Forward method of Grouper module.

        Args:
            in_features (Tensor): Input features of shape [num_groups_before x feat_dim].
            graph (Graph): Graph object storing the feature connections.

        Returns:
            out_features (Tensor): Output features of shape [num_groups_after x feat_dim]
        """

        # Project and normalize features
        proj_features = self.projector(in_features)
        norm_features = F.normalize(proj_features, dim=1)

        # Compute similarities and corresponding edge logits
        pair = norm_features[graph.edge_to_node, :]
        similarities = torch.bmm(pair[:, 0, None, :], pair[:, 1, :, None]).squeeze()
        edge_logits = torch.full([graph.num_edges_total+1], fill=-sys.float_info.max, dtype=torch.float)
        edge_logits[:-1] = torch.sigmoid(similarities - self.similarity_threshold)

        # Compute edge weights
        grp_edge_idx = graph.get_groups(edge_logits > 0.5)
        grp_edge_idx_plus = graph.add_neighbor_edges(grp_edge_idx)
        grp_edge_logits = edge_logits[grp_edge_idx_plus]
        grp_edge_weights = torch.softmax(grp_edge_logits, dim=-1)

        # Compute soft feature weights
        grp_edge_feat = graph.get_group_edge_feat(grp_edge_idx_plus)
        grp_soft_feat_weights = 0.5 * torch.bmm(grp_edge_weights[:, None, :], grp_edge_feat).squeeze()

        # Compute hard feature weights
        grp_num_feat = graph.get_num_feat_per_group(grp_edge_idx)
        grp_hard_feat_weights = self.get_hard_weights(grp_soft_feat_weights, grp_num_feat)

        # Compute group means (new output features)
        grp_feat_idx_plus = graph.get_feat_idx(grp_edge_idx_plus)
        grp_feat = in_features[grp_feat_idx_plus, :]
        out_features = torch.bmm(grp_hard_feat_weights[:, None, :], grp_feat).squeeze()

        # Update graph
        graph.update(grp_edge_idx)

        return out_features


def build_transformer(args):
    """
    Build transformer from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        Transformer module.
    """

    return Transformer(
        num_encoder_layers=args.num_encoder_layers,
        num_group_layers=args.num_group_layers,
        feature_dim=args.feature_dim,
        mha_dropout=args.mha_dropout,
        num_heads=args.num_heads,
        ffn_dropout=args.ffn_dropout,
        ffn_hidden_dim=args.ffn_hidden_dim,
        return_intermediate=True,
    )


if __name__ == '__main__':
    import datetime
    import time

    test1 = False
    test2 = False

    if test1:
        from main import get_parser
        args = get_parser().parse_args()
        transformer = build_transformer(args)

        detr_url = 'https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth'
        detr_state_dict = torch.hub.load_state_dict_from_url(detr_url)['model']

        from collections import OrderedDict
        encoder_state_dict = OrderedDict()

        encoder_identifier = 'transformer.encoder.'
        identifier_length = len(encoder_identifier)

        for detr_name, detr_state in detr_state_dict.items():
            if encoder_identifier in detr_name:
                new_name = detr_name[identifier_length:]
                encoder_state_dict[new_name] = detr_state

        transformer.encoder.load_state_dict(encoder_state_dict)

    if test2:
        # Test parameters
        H = 32
        W = 32
        batch_size = 4
        proj_dim = 128

        #  Input
        device = torch.device("cuda")
        num_feat = H*W*batch_size
        proj_features = torch.randn(num_feat, proj_dim, device=device, requires_grad=True)

        num_edges = (2*H*W - H - W) * batch_size
        edge_idx_feat_idx = np.random.randint(0, num_feat-1, size=(num_edges, 2))

        # Parallel
        start_time = time.time()
        pair = proj_features[edge_idx_feat_idx, :]
        similarities = torch.bmm(pair[:, 0, None, :], pair[:, 1, :, None]).squeeze()

        fake_loss = similarities.sum()
        fake_loss.backward()
        end_time = time.time()
        print(f"Memory (parallel): {torch.cuda.max_memory_allocated()/(1024*1024)} MB")
        print(f"Time (parallel): {datetime.timedelta(seconds=end_time-start_time)}")

        del fake_loss, pair, similarities
        torch.cuda.reset_peak_memory_stats()

        # Sequential
        start_time = time.time()
        similarities = torch.zeros(num_edges)
        for i, (idx1, idx2) in enumerate(edge_idx_feat_idx):
            feat1 = proj_features[idx1]
            feat2 = proj_features[idx2]
            similarities[i] = torch.dot(feat1, feat2)

        fake_loss = similarities.sum()
        fake_loss.backward()
        end_time = time.time()
        print(f"Memory (sequential): {torch.cuda.max_memory_allocated()/(1024*1024)} MB")
        print(f"Time (sequential): {datetime.timedelta(seconds=end_time-start_time)}")
