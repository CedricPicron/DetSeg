"""
Transformer modules and build function.
"""
import copy
from typing import Optional

import torch
from torch import nn, Tensor
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
        layers (nn.ModulesList): List of concatenated encoder layers.
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


class TransformerGroupDecoder(nn.Module):
    """
    Experimental.

    Submodules:
        1) Grouping
        2) Cross-attention
        3) Self-attention
        4) FFN

    Process:
        * Initialization:
            - input features of shape [H*W x batch_size x feature_dim]
            - neighbor indices:
                1) shape = [num_groups x batch_size x max_neighbors] with:
                    * num_groups = H*W
                    * max_neighbors = {4, 8} (i.e. without or with corners)
                    * index = -1 means empty neighbor



    3. Parallel (with flattened batch dimension):
        * Initialization (first iteration/layer):
            - in: encoder_features -> shape = [H*W, batch_size, feature_dim]

            - feature_to_edge = init_fe(H, W, batch_size) -> List[idx: feature_idx, List[edge_idx]]
            - edge_to_feature = init_ef(H, W, batch_size) -> List[idx: edge_idx, List[feature_idx]]

            - out: features = encoder_features.view(-1, feature_dim) -> shape = [H*W*batch_size, feature_dim]
            - out: graph = Graph(feature_to_edge, edge_to_feature) -> create Graph instance

            - note: num_features = H*W*batch_size at first iteration
            - note: num_edges = [2*H*W - H - W] * batch_size at first iteration

        * Grouping:
            ** Projection:
                - in: features -> shape = [num_features, feature_dim]
                - param: P = projection matrix -> shape = [feature_dim, proj_dim]
                - out: proj_features = mm(feature, P) -> shape = [num_features, proj_dim]

            ** Pair comparison:
                - in: proj_features -> shape = [num_features, proj_dim]
                - in: graph -> Graph object
                - param: similarity_threshold -> float in [0, 1]

                - idx_pair = np.zeros((2, num_edges))
                - [idx_pair[:, i] for i, idx_pair in enumerate(graph.edge_to_feature)]
                - pair = proj_features[idx_pair, :]-> shape = [2, num_edges, proj_dim]

                - similarities = bmm(pair[0, :, None, :], pair[1, :, :, None]).squeeze() -> shape = [num_edges]
                - out: edge_logits = sigmoid(similarities - similarity_threshold) -> shape = [num_edges]

            ** Make groups: Forward (hard decisions):
                - in: features -> shape = [num_features, feature_dim]
                - in: edge_logits -> shape = [num_edges]
                - in: graph -> Graph instance

                - save: edge_logits -> shape = [num_edges]
                - save: graph.edge_to_feature -> List[idx: edge_idx, List[feature_idx]]

                - mask = edge_logits > 0.5 -> shape = [num_edges]
                - group_to_edge, group_to_feature = get_groups(mask) -> List[idx: group_idx, List[edge/feature_idx]]
                - graph.update(group_to_edge, group_to_feature) -> update graph.{feature_to_edge, edge_to_feature}

                - group_sizes = [len(feature_idx) for feature_idx in group_to_feature] -> List[group_size]
                - group_sizes = torch.tensor(group_sizes).unsqueeze(1) -> shape = [num_groups, 1]

                - group_to_feature = list_to_tensor(group_to_feature, fill=0) -> shape = [num_groups, max_features]
                - group_features = features[group_to_feature, :] -> shape = [num_groups, max_features, feature_dim]

                - out: group_means = sum(group_features, dim=1) / group_sizes -> shape = [num_groups, feature_dim]
                - save: group_to_edge, group_to_feature -> List[idx: group_idx, List[edge/feature_idx]]

            ** Make groups: Backward (soft decisions):
                in: grad_group_means -> shape = [num_groups, feature_dim]
                ctx: edge_logits -> shape = [num_edges]
                ctx: edge_to_feature -> List[idx: edge_idx, List[feature_idx]] (from old graph)
                ctx: group_to_edge, group_to_feature -> List[idx: group_idx, List[edge/feature_idx]]

                edge_logits = cat((edge_logits, torch.tensor([-sys.float_info.max]))) -> shape = [num_edges + 1]
                group_to_edge = list_to_tensor(group_to_edge, fill=num_edges) -> shape = [num_groups, max_edges]

                group_edge_logits = edge_logits[group_to_edge] -> shape = [num_groups, max_edges]
                group_edge_weights = softmax(group_edge_logits, dim=-1) -> shape = [num_groups, max_edges]

                edge_feature = get_matrix(group_to_edge, group_to_feature) -> shape = [max_edges, max_features]
                group_feature_weights = 0.5 * mm(group_edge_weights, edge_feature) -> shape = [num_groups, max_features]

                out: grad_features -> shape = [num_features, feature_dim]
                out: grad_edge_logits -> shape = [num_edges]
    """


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
