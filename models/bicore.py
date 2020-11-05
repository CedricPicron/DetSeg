"""
Bidirectional core modules and build function.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class BiAttnConv(nn.Module):
    """
    Class implementing the bidirectional attention-based convolution module.

    Attributes:
        feat_sizes (List): List of size [num_maps] containing the feature size in each map.
        num_heads_list (List): List of size [num_maps] containing the number of heads in each map.
        attn_scalings (List): List of size [num_maps] containing attention scale factors.

        in_proj_weights (nn.ParameterList): List of size [num_maps] with input projection matrices.
        in_proj_biases (nn.ParameterList): List of size [num_maps] with input projection biases.
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

    def __init__(self, feat_sizes, num_heads_list, dropout, ffn_size_multiplier):
        """
        Initializes the BiAttnConv module.

        Args:
            feat_sizes (List): List of size [num_maps] containing the feature size in each map.
            num_heads_list (List): List of size [num_maps] containing the number of heads in each map.
            dropout (float): Dropout probability used throughout the module.
            ffn_size_multiplier (int): Feature size multiplier used for FFN hidden layer feature sizes.
        """

        # Intialization of default nn.Module
        super().__init__()

        # Check inputs
        check = len(feat_sizes) == len(num_heads_list)
        msg = f"Inconsistent lengths between 'feat_sizes' ({len(feat_sizes)}) and 'num_heads' ({len(num_heads_list)})"
        assert check, msg

        check = all([feat_size % num_heads == 0 for feat_size, num_heads in zip(feat_sizes, num_heads_list)])
        msg = f"Feature sizes ({feat_sizes}) should be divisible by their number of heads ({num_heads_list})"
        assert check, msg

        # Set feature sizes, number of heads and attention scalings attributes
        self.feat_sizes = feat_sizes
        self.num_heads_list = num_heads_list
        self.attn_scalings = [float(f//num_heads)**-0.5 for f, num_heads in zip(feat_sizes, num_heads_list)]

        # Initializing input and output projection parameters
        bot_proj_size = 4*feat_sizes[0] + 2*feat_sizes[1]
        mid_proj_sizes = [2*i + 5*j + 2*k for i, j, k in zip(feat_sizes[:-2], feat_sizes[1:-1], feat_sizes[2:])]
        top_proj_size = 2*feat_sizes[-2] + 4*feat_sizes[-1]
        proj_sizes = [bot_proj_size, *mid_proj_sizes, top_proj_size]

        self.in_proj_weights = nn.ParameterList([Parameter(torch.empty(p, f)) for p, f in zip(proj_sizes, feat_sizes)])
        self.in_proj_biases = nn.ParameterList([Parameter(torch.empty(p)) for p in proj_sizes])

        self.out_proj_weights = nn.ParameterList([Parameter(torch.empty(f, 3*f)) for f in feat_sizes])
        self.out_proj_biases = nn.ParameterList([Parameter(torch.empty(3*f)) for f in feat_sizes])

        # Initializing attention dropout and layernorm
        self.attn_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in feat_sizes])
        self.attn_layernorms = nn.ModuleList([nn.LayerNorm(f) for f in feat_sizes])

        # Initializing feedforward network linear, dropout and layernorm modules
        self.ffn_in_projs = nn.ModuleList([nn.Linear(ffn_size_multiplier*f, f) for f in feat_sizes])
        self.ffn_out_projs = nn.ModuleList([nn.Linear(f, ffn_size_multiplier*f) for f in feat_sizes])
        self.ffn_in_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in feat_sizes])
        self.ffn_out_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in feat_sizes])
        self.ffn_layernorms = nn.ModuleList([nn.LayerNorm(f) for f in feat_sizes])

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
        Forward method of the BiAttnConv module.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, H, W, feat_size].

        Returns:
            feat_maps (List): List of size [num_maps] with updated feature maps of shape [batch_size, H, W, feat_size].
        """

        # Project feature maps
        proj_maps = []
        for feat_map, in_proj_weight, in_proj_bias in zip(feat_maps, self.in_proj_weights, self.in_proj_biases):
            proj_map = F.linear(feat_map, in_proj_weight, in_proj_bias)
            proj_maps.append(proj_map)

        # Get batch size and kernel indices
        batch_size = feat_maps[0].shape[0]
        device = feat_maps[0].device
        x_kernel = torch.tensor(torch.arange(3), dtype=torch.int, device=device).repeat_interleave(3)
        y_kernel = torch.tensor(torch.arange(3), dtype=torch.int, device=device).repeat(3)

        # Perform self-attention
        zip_list = [proj_maps, self.feat_sizes, self.attn_scalings, self.num_heads_list]
        zip_list = [*zip_list, self.out_proj_weights, self.out_proj_biases]

        self_attn_maps = []
        for proj_map, f, attn_scaling, num_heads, out_proj_weight, out_proj_bias in zip(*zip_list):
            H, W = proj_map.shape[1:-1]

            query_map = attn_scaling*proj_map[:, :, :, :f]
            query_map = query_map.view(batch_size, H, W, num_heads, 1, -1)

            key_map = proj_map[:, :, :, f:2*f].permute(0, 3, 1, 2)
            key_map = F.pad(key_map, (1, 1, 1, 1)).permute(0, 2, 3, 1)
            key_map = torch.stack([key_map[:, i:H+i, j:W+j, :] for i, j in zip(x_kernel, y_kernel)], dim=-1)
            key_map = key_map.view(batch_size, H, W, num_heads, -1, 9)

            value_map = proj_map[:, :, :, 2*f:3*f].permute(0, 3, 1, 2)
            value_map = F.pad(value_map, (1, 1, 1, 1)).permute(0, 2, 3, 1)
            value_map = torch.stack([value_map[:, i:H+i, j:W+j, :] for i, j in zip(x_kernel, y_kernel)], dim=-1)
            value_map = value_map.view(batch_size, H, W, num_heads, -1, 9)

            attn_weights = F.softmax(torch.matmul(query_map, key_map), dim=-1)
            weighted_map = torch.sum(attn_weights * value_map, dim=-1).view(batch_size, H, W, -1)
            self_attn_map = F.linear(weighted_map, out_proj_weight[:, :f], out_proj_bias[:f])
            self_attn_maps.append(self_attn_map)

        # Perform top-down cross-attention
        zip_list = [proj_maps[:-1], proj_maps[1:], self.feat_sizes[:-1], self.feat_sizes[1:], self.attn_scalings]
        zip_list = [*zip_list, self.num_heads_list, self.out_proj_weights, self.out_proj_biases]

        top_down_attn_maps = []
        for proj_map1, proj_map2, f, g, attn_scaling, num_heads, out_proj_weight, out_proj_bias in zip(*zip_list):
            H1, W1 = proj_map1.shape[1:-1]
            H2, W2 = proj_map2.shape[1:-1]

            query_map = attn_scaling*proj_map1[:, :, :, 3*f:4*f]
            query_map = query_map.view(batch_size, H1, W1, num_heads, 1, -1)

            key_map = proj_map2[:, :, :, 4*f:4*f+g].permute(0, 3, 1, 2)
            key_map = F.pad(key_map, (1, 1, 1, 1)).permute(0, 2, 3, 1)
            key_map = torch.stack([key_map[:, i:H2+i, j:W2+j, :] for i, j in zip(x_kernel, y_kernel)], dim=-1)
            key_map = key_map.view(batch_size, H2, W2, num_heads, -1, 9)

            value_map = proj_map2[:, :, :, 4*f+g:4*f+2*g].permute(0, 3, 1, 2)
            value_map = F.pad(value_map, (1, 1, 1, 1)).permute(0, 2, 3, 1)
            value_map = torch.stack([value_map[:, i:H+i, j:W+j, :] for i, j in zip(x_kernel, y_kernel)], dim=-1)
            value_map = value_map.view(batch_size, H, W, num_heads, -1, 9)

            attn_weights = F.softmax(torch.matmul(query_map, key_map), dim=-1)
            weighted_map = torch.sum(attn_weights * value_map, dim=-1).view(batch_size, H, W, -1)
            top_down_attn_map = F.linear(weighted_map, out_proj_weight[:, f:2*f], out_proj_bias[f:2*f])
            top_down_attn_maps.append(top_down_attn_map)

        # Perform bottom-up cross-attention
        bottom_up_attn_maps = []
        for proj_map0, proj_map1, e, f, attn_scaling, num_heads, out_proj_weight, out_proj_bias in zip(*zip_list):
            H1, W1 = proj_map1.shape[1:-1]
            H2, W2 = proj_map2.shape[1:-1]

            query_map = attn_scaling*proj_map1[:, :, :, -2*e-f:-2*e]
            query_map = query_map.view(batch_size, H, W, num_heads, 1, -1)

            key_map = proj_map0[:, :, :, -2*e:-e].permute(0, 3, 1, 2)
            key_map = F.pad(key_map, (1, 1, 1, 1)).permute(0, 2, 3, 1)
            key_map = torch.stack([key_map[:, i:H+i, j:W+j, :] for i, j in zip(x_kernel, y_kernel)], dim=-1)
            key_map = key_map.view(batch_size, H, W, num_heads, -1, 9)

            value_map = proj_map0[:, :, :, -e:].permute(0, 3, 1, 2)
            value_map = F.pad(value_map, (1, 1, 1, 1)).permute(0, 2, 3, 1)
            value_map = torch.stack([value_map[:, i:H+i, j:W+j, :] for i, j in zip(x_kernel, y_kernel)], dim=-1)
            value_map = value_map.view(batch_size, H, W, num_heads, -1, 9)

            attn_weights = F.softmax(torch.matmul(query_map, key_map), dim=-1)
            weighted_map = torch.sum(attn_weights * value_map, dim=-1).view(batch_size, H, W, -1)
            bottom_up_attn_map = F.linear(weighted_map, out_proj_weight[:, f:2*f], out_proj_bias[f:2*f])
            bottom_up_attn_maps.append(bottom_up_attn_map)

        # Add attention maps together and update feature maps with additional dropout and layernorm
        zip_list = [feat_maps, self_attn_maps, top_down_attn_maps, bottom_up_attn_maps]
        zip_list = [*zip_list, self.attn_dropouts, self.attn_layernorms]

        for feat_map, attn_map1, attn_map2, attn_map3, dropout, layernorm in zip(*zip_list):
            delta_feat_map = attn_map1 + attn_map2 + attn_map3
            feat_map = feat_map + dropout(delta_feat_map)
            feat_map = layernorm(feat_map)

        # Update feature maps with feedforward network (FFN)
        zip_list = [feat_maps, self.ffn_in_projs, self.ffn_in_dropouts, self.ffn_out_projs]
        zip_list = [*zip_list, self.ffn_out_dropouts, self.ffn_layernorms]

        for feat_map, in_proj, in_dropout, out_proj, out_dropout, layernorm in zip(*zip_list):
            delta_feat_map = out_proj(in_dropout(F.relu(in_proj(feat_map))))
            feat_map = feat_map + out_dropout(delta_feat_map)
            feat_map = layernorm(feat_map)

        return feat_maps


def build_bicore(args):
    """
    Build BiCore module from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        bicore (nn.Module): The specified BiCore module.
    """

    # Check command-line arguments
    check = args.max_resolution_id >= args.min_resolution_id
    msg = "'--max_resolution_id' should be larger than '--min_resolution_id'"
    assert check, msg

    # Get feature sizes and number of heads list
    map_ids = range(args.min_resolution_id, args.max_resolution_id+1)
    feat_sizes = [min((args.base_feat_size * 2**i, args.max_feat_size)) for i in map_ids]
    num_heads_list = [min((args.base_num_heads * 2**i, args.max_num_heads)) for i in map_ids]

    # Get BiCore type
    if args.bicore_type == 'bi_attn_conv':
        bicore = BiAttnConv(feat_sizes, num_heads_list, args.bicore_dropout, args.ffn_size_multiplier)
    else:
        raise ValueError(f"Unknown BiCore type '{args.bicore_type}' was provided.")

    return bicore
