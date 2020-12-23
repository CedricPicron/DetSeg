"""
BLA module and build function.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from models.projector import Projector


class BLA(nn.Module):
    """
    Class implementing the BLA (Bidirectional Local Attention) module.

    Attributes:
        feat_sizes (List): List of size [num_maps] containing the feature size of each map.
        num_heads_list (List): List of size [num_maps] containing the number of heads for each map.
        attn_scalings (List): List of size [num_maps] containing attention scale factors.

        in_proj_weights (nn.ParameterList): List of size [num_maps] with input projection matrices.
        in_proj_biases (nn.ParameterList): List of size [num_maps] with input projection biases.
        with_pos_feats (bool): Boolean indicating whether local position features are used and learned.
        pos_feats (nn.ParameterList, optional): List of size [num_maps] with local position features.
        out_proj_weights (nn.ParameterList): List of size [num_maps] with output projection matrices.
        out_proj_biases (nn.ParameterList): List of size [num_maps] with output projection biases.
        attn_dropouts (nn.ModuleList): List of size [num_maps] with attention dropout modules.
        attn_layernorms (nn.ModuleList): List of size [num_maps] with attention layernorm modules.

        ffn_in_projs (nn.ModuleList): List of size [num_maps] with FFN input projection modules.
        ffn_out_projs (nn.ModuleList): List of size [num_maps] with FFN output projection modules.
        ffn_in_dropouts (nn.ModuleList): List of size [num_maps] with FFN input dropout modules.
        ffn_out_dropouts (nn.ModuleList): List of size [num_maps] with FFN output dropout modules.
        ffn_layernorms (nn.ModuleList): List of size [num_maps] with FFN layernorm modules.
    """

    def __init__(self, in_feat_sizes, out_feat_sizes, num_heads_list, with_pos_feats, dropout, ffn_size_multiplier):
        """
        Initializes the BLA module.

        Args:
            in_feat_sizes (List): List of size [num_in_maps] containing the feature size of each input map.
            out_feat_sizes (List): List of size [num_out_maps] containing the feature size of each output map.
            num_heads_list (List): List of size [num_out_maps] containing the number of heads for each output map.
            with_pos_feats (bool): Boolean indicating whether local position features are used and learned.
            dropout (float): Dropout probability used throughout the module.
            ffn_size_multiplier (int): Feature size multiplier used for FFN hidden layer feature sizes.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Check inputs
        num_maps_delta = len(out_feat_sizes) - len(in_feat_sizes)
        msg = f"Fewer output maps than inputs map are not allowed (got {abs(num_maps_delta)} fewer)."
        assert num_maps_delta >= 0, msg

        msg = f"Only one more output map than input map is allowed (got {num_maps_delta} more)."
        assert num_maps_delta <= 1, msg

        check = all([feat_size % num_heads == 0 for feat_size, num_heads in zip(out_feat_sizes, num_heads_list)])
        msg = f"Feature sizes ({out_feat_sizes}) should be divisible by their number of heads ({num_heads_list})."
        assert check, msg

        # Initialize input projector
        fixed_settings = {'proj_type': 'conv1', 'conv_stride': 1}
        proj_dicts = []

        for in_map_id, out_feat_size in enumerate(out_feat_sizes):
            proj_dict = {'in_map_id': in_map_id, 'out_feat_size': out_feat_size, **fixed_settings}
            proj_dicts.append(proj_dict)

        if num_maps_delta == 1:
            fixed_settings = {'proj_type': 'conv3', 'conv_stride': 2}
            proj_dict = {'in_map_id': len(in_feat_sizes)-1, 'out_feat_size': out_feat_sizes[-1]}

        self.in_proj = Projector(in_feat_sizes, proj_dicts)

        # Set feature sizes, number of heads and attention scalings attributes
        self.feat_sizes = out_feat_sizes
        self.num_heads_list = num_heads_list
        self.attn_scalings = [float(f//num_heads)**-0.25 for f, num_heads in zip(out_feat_sizes, num_heads_list)]

        # Initialize input projection parameters
        fs = out_feat_sizes
        bot_proj_size = 4*fs[0] + 2*fs[1]
        mid_proj_sizes = [2*f0 + 5*f1 + 2*f2 for f0, f1, f2 in zip(fs[:-2], fs[1:-1], fs[2:])]
        top_proj_size = 2*fs[-2] + 4*fs[-1]
        proj_sizes = [bot_proj_size, *mid_proj_sizes, top_proj_size]

        self.attn_in_weights = nn.ParameterList([Parameter(torch.empty(p, f)) for p, f in zip(proj_sizes, fs)])
        self.attn_in_biases = nn.ParameterList([Parameter(torch.empty(p)) for p in proj_sizes])

        # Set attribute determining whether position features are used and learned
        self.with_pos_feats = with_pos_feats

        # Initialize position features if requested
        if with_pos_feats:
            bot_pos_size = 2*fs[0]
            mid_pos_size = [3*f for f in fs[1:-1]]
            top_pos_size = 2*fs[-1]

            pos_sizes = [bot_pos_size, *mid_pos_size, top_pos_size]
            self.pos_feats = nn.ParameterList([Parameter(torch.empty(p, 9)) for p in pos_sizes])

        # Initialize output projection parameters
        self.attn_out_weights = nn.ParameterList([Parameter(torch.empty(f, 3*f)) for f in fs])
        self.attn_out_biases = nn.ParameterList([Parameter(torch.empty(3*f)) for f in fs])

        # Initialize attention dropout and layernorm modules
        self.attn_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in fs])
        self.attn_layernorms = nn.ModuleList([nn.LayerNorm(f) for f in fs])

        # Initialize feedforward network linear, dropout and layernorm modules
        self.ffn_in_projs = nn.ModuleList([nn.Linear(f, ffn_size_multiplier*f) for f in fs])
        self.ffn_out_projs = nn.ModuleList([nn.Linear(ffn_size_multiplier*f, f) for f in fs])
        self.ffn_in_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in fs])
        self.ffn_out_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in fs])
        self.ffn_layernorms = nn.ModuleList([nn.LayerNorm(f) for f in fs])

        # Set default initial values of module parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets module parameters to default initial values.
        """

        [nn.init.xavier_uniform_(param) for param in self.parameters() if param.dim() > 1]
        [nn.init.constant_(bias, 0.0) for bias in self.in_proj_biases]
        [nn.init.constant_(bias, 0.0) for bias in self.out_proj_biases]

    def forward(self, feat_maps):
        """
        Forward method of the BLA module.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, fH, fW, feat_size].

        Returns:
            feat_maps (List): List of size [num_maps] of updated feature maps of shape [batch_size, fH, fW, feat_size].
        """

        # Project feature maps
        proj_maps = []
        for feat_map, in_proj_weight, in_proj_bias in zip(feat_maps, self.in_proj_weights, self.in_proj_biases):
            proj_map = F.linear(feat_map, in_proj_weight, in_proj_bias)
            proj_maps.append(proj_map)

        # Get batch size
        batch_size = feat_maps[0].shape[0]

        # Perform self-attention
        zip_list = [range(len(feat_maps)), proj_maps, self.feat_sizes, self.attn_scalings]
        zip_list = [*zip_list, self.num_heads_list, self.out_proj_weights, self.out_proj_biases]

        self_attn_maps = []
        for i, proj_map, f, scale, num_heads, out_weight, out_bias in zip(*zip_list):
            H, W = proj_map.shape[1:-1]

            query_map = scale*proj_map[:, :, :, :f]
            query_map = query_map.view(batch_size, H, W, num_heads, 1, -1)

            key_map = scale*proj_map[:, :, :, f:2*f]
            key_map = F.pad(key_map.permute(0, 3, 1, 2), (1, 1, 1, 1), mode='replicate').permute(0, 2, 3, 1)

            sizes = [batch_size, H, W, f, 3, 3]
            strides = [*key_map.stride(), key_map.stride()[1], key_map.stride()[2]]
            key_map = key_map.as_strided(sizes, strides).reshape(batch_size, H, W, f, 9)
            key_map = key_map + self.pos_feats[i][:f, :] if self.with_pos_feats else key_map
            key_map = key_map.view(batch_size, H, W, num_heads, -1, 9)

            value_map = proj_map[:, :, :, 2*f:3*f]
            value_map = F.pad(value_map.permute(0, 3, 1, 2), (1, 1, 1, 1), mode='replicate').permute(0, 2, 3, 1)
            value_map = value_map.as_strided(sizes, strides).reshape(batch_size, H, W, f, 9)
            value_map = value_map.view(batch_size, H, W, num_heads, -1, 9)

            attn_weights = F.softmax(torch.matmul(query_map, key_map), dim=-1)
            weighted_map = torch.sum(attn_weights * value_map, dim=-1).view(batch_size, H, W, -1)
            self_attn_map = F.linear(weighted_map, out_weight[:, :f], out_bias[:f])
            self_attn_maps.append(self_attn_map)

        # Perform top-down cross-attention
        last_map_bools = [i == len(feat_maps)-1 for i in range(len(feat_maps))]
        interpolation_kwargs = {'mode': 'bilinear', 'align_corners': True}

        zip_list = [range(len(feat_maps)-1), proj_maps[:-1], proj_maps[1:], self.feat_sizes[:-1], self.feat_sizes[1:]]
        zip_list = [*zip_list, self.attn_scalings[:-1], last_map_bools[1:], self.num_heads_list[:-1]]
        zip_list = [*zip_list, self.out_proj_weights[:-1], self.out_proj_biases[:-1]]

        top_down_attn_maps = []
        for i, proj_map1, proj_map2, f, g, scale, last_map, num_heads, out_weight, out_bias in zip(*zip_list):
            H1, W1 = proj_map1.shape[1:-1]
            H2, W2 = proj_map2.shape[1:-1]

            query_map = scale*proj_map1[:, ::2, ::2, 3*f:4*f]
            query_map = query_map.view(batch_size, H2, W2, num_heads, 1, -1)

            key_map = scale*proj_map2[:, :, :, 3*g:3*g+f] if last_map else scale*proj_map2[:, :, :, 4*g:4*g+f]
            key_map = F.pad(key_map.permute(0, 3, 1, 2), (1, 1, 1, 1), mode='replicate').permute(0, 2, 3, 1)

            sizes = [batch_size, H2, W2, f, 3, 3]
            strides = [*key_map.stride(), key_map.stride()[1], key_map.stride()[2]]
            key_map = key_map.as_strided(sizes, strides).reshape(batch_size, H2, W2, f, 9)
            key_map = key_map + self.pos_feats[i][f:2*f, :] if self.with_pos_feats else key_map
            key_map = key_map.view(batch_size, H2, W2, num_heads, -1, 9)

            value_map = proj_map2[:, :, :, 3*g+f:3*g+2*f] if last_map else proj_map2[:, :, :, 4*g+f:4*g+2*f]
            value_map = F.pad(value_map.permute(0, 3, 1, 2), (1, 1, 1, 1), mode='replicate').permute(0, 2, 3, 1)
            value_map = value_map.as_strided(sizes, strides).reshape(batch_size, H2, W2, f, 9)
            value_map = value_map.view(batch_size, H2, W2, num_heads, -1, 9)

            attn_weights = F.softmax(torch.matmul(query_map, key_map), dim=-1)
            weighted_map = torch.sum(attn_weights * value_map, dim=-1).view(batch_size, H2, W2, -1)
            top_down_attn_map = F.linear(weighted_map, out_weight[:, f:2*f], out_bias[f:2*f])

            pH, pW = (int(H1 % 2 == 0), int(W1 % 2 == 0))
            top_down_attn_map = top_down_attn_map.permute(0, 3, 1, 2)
            top_down_attn_map = F.pad(top_down_attn_map, (0, pW, 0, pH), mode='replicate')
            top_down_attn_map = F.interpolate(top_down_attn_map, size=(H1+pH, W1+pW), **interpolation_kwargs)
            top_down_attn_map = top_down_attn_map[:, :, :H1, :W1].permute(0, 2, 3, 1)
            top_down_attn_maps.append(top_down_attn_map)

        top_down_attn_maps.append(torch.zeros_like(feat_maps[-1]))

        # Perform bottom-up cross-attention
        first_map_bools = [i == 0 for i in range(len(feat_maps))]

        zip_list = [range(1, len(feat_maps)), proj_maps[:-1], proj_maps[1:], self.feat_sizes[:-1], self.feat_sizes[1:]]
        zip_list = [*zip_list, self.attn_scalings[1:], first_map_bools[:-1], self.num_heads_list[1:]]
        zip_list = [*zip_list, self.out_proj_weights[1:], self.out_proj_biases[1:]]

        bottom_up_attn_maps = [torch.zeros_like(feat_maps[0])]
        for i, proj_map0, proj_map1, e, f, scale, first_map, num_heads, out_weight, out_bias in zip(*zip_list):
            H0, W0 = proj_map0.shape[1:-1]
            H1, W1 = proj_map1.shape[1:-1]

            query_map = scale*proj_map1[:, :, :, -f:]
            query_map = query_map.view(batch_size, H1, W1, num_heads, 1, -1)

            pH, pW = (H0 % 2, W0 % 2)
            key_map = scale*proj_map0[:, :, :, -2*f:-f] if first_map else scale*proj_map0[:, :, :, -e-2*f:-e-f]
            key_map = F.pad(key_map.permute(0, 3, 1, 2), (1, pW, 1, pH), mode='replicate').permute(0, 2, 3, 1)

            sizes = [batch_size, H1, W1, f, 3, 3]
            s0, s1, s2, s3 = key_map.stride()
            strides = [s0, 2*s1, 2*s2, s3, s1, s2]
            key_map = key_map.as_strided(sizes, strides).reshape(batch_size, H1, W1, f, 9)
            key_map = key_map + self.pos_feats[i][-f:, :] if self.with_pos_feats else key_map
            key_map = key_map.view(batch_size, H1, W1, num_heads, -1, 9)

            value_map = proj_map0[:, :, :, -f:] if first_map else proj_map0[:, :, :, -e-f:-e]
            value_map = F.pad(value_map.permute(0, 3, 1, 2), (1, pW, 1, pH), mode='replicate').permute(0, 2, 3, 1)
            value_map = value_map.as_strided(sizes, strides).reshape(batch_size, H1, W1, f, 9)
            value_map = value_map.view(batch_size, H1, W1, num_heads, -1, 9)

            attn_weights = F.softmax(torch.matmul(query_map, key_map), dim=-1)
            weighted_map = torch.sum(attn_weights * value_map, dim=-1).view(batch_size, H1, W1, -1)
            bottom_up_attn_map = F.linear(weighted_map, out_weight[:, 2*f:3*f], out_bias[2*f:3*f])
            bottom_up_attn_maps.append(bottom_up_attn_map)

        # Add attention maps together and update feature maps with additional dropout and layernorm
        zip_list = [feat_maps, self_attn_maps, top_down_attn_maps, bottom_up_attn_maps]
        zip_list = [*zip_list, self.attn_dropouts, self.attn_layernorms]

        feat_maps = []
        for feat_map, attn_map1, attn_map2, attn_map3, dropout, layernorm in zip(*zip_list):
            delta_feat_map = attn_map1 + attn_map2 + attn_map3
            feat_map = feat_map + dropout(delta_feat_map)
            feat_map = layernorm(feat_map)
            feat_maps.append(feat_map)

        # Update feature maps with feedforward network (FFN)
        zip_list = [feat_maps, self.ffn_in_projs, self.ffn_in_dropouts, self.ffn_out_projs]
        zip_list = [*zip_list, self.ffn_out_dropouts, self.ffn_layernorms]

        feat_maps = []
        for feat_map, in_proj, in_dropout, out_proj, out_dropout, layernorm in zip(*zip_list):
            delta_feat_map = out_proj(in_dropout(F.relu(in_proj(feat_map))))
            feat_map = feat_map + out_dropout(delta_feat_map)
            feat_map = layernorm(feat_map)
            feat_maps.append(feat_map)

        return feat_maps


def build_bla(args):
    """
    Build BLA module from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        bla (nn.Module): The specified BLA module.
    """

    # Check command-line arguments
    min_id, max_id = (args.bla_min_downsampling, args.bla_max_downsampling)
    assert_msg = "'--bla_max_downsampling' ({max_id}) should be larger than '--bla_min_downsampling' ({min_id})"
    assert max_id > min_id, assert_msg

    # Get input arguments
    in_feat_sizes = args.backbone.feat_sizes
    out_feat_sizes = [min((args.bla_base_feat_size * 2**i, args.bla_max_feat_size)) for i in range(min_id, max_id+1)]
    num_heads_list = [min((args.bla_base_num_heads * 2**i, args.bla_max_num_heads)) for i in range(min_id, max_id+1)]

    with_pos_feats = not args.bla_no_pos_feats
    dropout = args.bla_dropout
    ffn_size_multiplier = args.bla_ffn_size_multiplier

    # Build BLA module
    bla = BLA(in_feat_sizes, out_feat_sizes, num_heads_list, with_pos_feats, dropout, ffn_size_multiplier)

    return bla