"""
BLA module and build function.
"""
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from models.projector import Projector


class BLA(nn.Module):
    """
    Class implementing the BLA (Bidirectional Local Attention) module.

    Attributes:
        num_layers (int): Integer containing the number of BLA update layers.
        in_proj (Projector): Module computing the initial output feature maps from input feature maps.

        feat_sizes (List): List of size [num_out_maps] containing the feature size of each output map.
        num_heads_list (List): List of size [num_out_maps] containing the number of heads for each output map.
        attn_scalings (List): List of size [num_out_maps] containing attention scale factors.

        attn_dropout (float): Dropout probability (between 0 and 1) used during attention computation.
        ffn_dropout (float): Dropout probability (between 0 and 1) used during FFN computation.

        attn_in_weights (nn.ModuleList): List [num_layers] with lists [num_out_maps] of input projection matrices.
        attn_in_biases (nn.ModuleList): List [num_layers] with lists [num_out_maps] of input projection biases.
        attn_out_weights (nn.ModuleList): List [num_layers] with lists [num_out_maps] of output projection matrices.
        attn_out_biases (nn.ModuleList): List [num_layers] with lists [num_out_maps] of output projection biases.
        attn_layernorms (nn.ModuleList): List [num_layers] with lists [num_out_maps] of attention layernorm modules.

        pos_attn (bool): Boolean indicating whether local position features are used and learned.
        pos_feats (nn.ModuleList): Optional list [num_layers] with lists [num_out_maps] of local position features.

        ffn_in_projs (nn.ModuleList): List [num_layers] with lists [num_out_maps] of FFN input projection modules.
        ffn_out_projs (nn.ModuleList): List [num_layers] with lists [num_out_maps] of FFN output projection modules.
        ffn_layernorms (nn.ModuleList): List [num_layers] with lists [num_out_maps] of FFN layernorm modules.
    """

    def __init__(self, num_layers, in_feat_sizes, out_feat_sizes, attn_dict, ffn_dict):
        """
        Initializes the BLA module.

        Args:
            num_layers (int): Integer containing the number of BLA update layers.
            in_feat_sizes (List): List of size [num_in_maps] containing the feature size of each input map.
            out_feat_sizes (List): List of size [num_out_maps] containing the feature size of each output map.

            attn_dict (Dict): Attention dictionary containing following keys:
                - num_heads_list (List): list of size [num_out_maps] containing the number of heads per output map;
                - pos_attn (bool): boolean indicating whether local position features are used and learned;
                - dropout (float): dropout probability (between 0 and 1) used during attention computation.

            ffn_dict (Dict): FFN (feed-forward network) dictionary containing following keys:
                - size_multiplier (int): feature size multiplier used for FFN hidden layer feature sizes;
                - dropout (float): dropout probability (between 0 and 1) used during FFN computation.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Check inputs
        num_maps_delta = len(out_feat_sizes) - len(in_feat_sizes)
        msg = f"Fewer output maps than inputs map are not allowed (got {abs(num_maps_delta)} fewer)."
        assert num_maps_delta >= 0, msg

        msg = f"Only one more output map than input map is allowed (got {num_maps_delta} more)."
        assert num_maps_delta <= 1, msg

        num_heads_list = attn_dict['num_heads_list']
        check = all([feat_size % num_heads == 0 for feat_size, num_heads in zip(out_feat_sizes, num_heads_list)])
        msg = f"Feature sizes ({out_feat_sizes}) should be divisible by their number of heads ({num_heads_list})."
        assert check, msg

        # Set number of layers attribute
        self.num_layers = num_layers

        # Initialize input projector
        fixed_settings = {'proj_type': 'conv1', 'conv_stride': 1}
        proj_dicts = []

        for i in range(len(in_feat_sizes)):
            proj_dict = {'in_map_id': i, 'out_feat_size': out_feat_sizes[i], **fixed_settings}
            proj_dicts.append(proj_dict)

        if num_maps_delta == 1:
            proj_dict = {'in_map_id': len(in_feat_sizes)-1, 'out_feat_size': out_feat_sizes[-1]}
            proj_dict = {**proj_dict, 'proj_type': 'conv3', 'conv_stride': 2}
            proj_dicts.append(proj_dict)

        self.in_proj = Projector(in_feat_sizes, proj_dicts)

        # Set feature sizes, number of heads and attention scalings attributes
        self.feat_sizes = out_feat_sizes
        self.num_heads_list = num_heads_list
        self.attn_scalings = [float(f//num_heads)**-0.25 for f, num_heads in zip(out_feat_sizes, num_heads_list)]

        # Set dropout attributes
        self.attn_dropout = attn_dict['dropout']
        self.ffn_dropout = ffn_dict['dropout']

        # Initialize input projection parameters
        fs = out_feat_sizes
        bot_proj_size = 4*fs[0] + 2*fs[1]
        mid_proj_sizes = [2*f0 + 5*f1 + 2*f2 for f0, f1, f2 in zip(fs[:-2], fs[1:-1], fs[2:])]
        top_proj_size = 2*fs[-2] + 4*fs[-1]
        proj_sizes = [bot_proj_size, *mid_proj_sizes, top_proj_size]

        self.attn_in_weights = nn.ParameterList([Parameter(torch.empty(p, f)) for p, f in zip(proj_sizes, fs)])
        self.attn_in_biases = nn.ParameterList([Parameter(torch.empty(p)) for p in proj_sizes])

        # Initialize output projection parameters
        self.attn_out_weights = nn.ParameterList([Parameter(torch.empty(f, 3*f)) for f in fs])
        self.attn_out_biases = nn.ParameterList([Parameter(torch.empty(3*f)) for f in fs])

        # Initialize attention layernorm modules
        self.attn_layernorms = nn.ModuleList([nn.LayerNorm(f) for f in fs])

        # Set attribute determining whether position features are used and learned
        self.pos_attn = attn_dict['pos_attn']

        # Initialize position features if requested
        if self.pos_attn:
            bot_pos_size = 2*fs[0]
            mid_pos_size = [3*f for f in fs[1:-1]]
            top_pos_size = 2*fs[-1]

            pos_sizes = [bot_pos_size, *mid_pos_size, top_pos_size]
            self.pos_feats = nn.ParameterList([Parameter(torch.empty(p, 9)) for p in pos_sizes])

        # Initialize FFN projection and layernorm modules
        size_multiplier = ffn_dict['size_multiplier']
        self.ffn_in_projs = nn.ModuleList([nn.Linear(f, size_multiplier*f) for f in fs])
        self.ffn_out_projs = nn.ModuleList([nn.Linear(size_multiplier*f, f) for f in fs])
        self.ffn_layernorms = nn.ModuleList([nn.LayerNorm(f) for f in fs])

        # Initialize separate update layers
        for attr_name in dir(self):
            module = getattr(self, attr_name)

            if isinstance(module, nn.ModuleList) or isinstance(module, nn.ParameterList):
                setattr(self, attr_name, nn.ModuleList([deepcopy(module) for _ in range(num_layers)]))

        # Set default initial values of module parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets module parameters to default initial values.
        """

        [nn.init.xavier_uniform_(layer_weight) for weights in self.attn_in_weights for layer_weight in weights]
        [nn.init.xavier_uniform_(layer_weight) for weights in self.attn_out_weights for layer_weight in weights]
        [nn.init.zeros_(layer_bias) for biases in self.attn_in_biases for layer_bias in biases]
        [nn.init.zeros_(layer_bias) for biases in self.attn_out_biases for layer_bias in biases]
        [nn.init.zeros_(layer_pos_feats) for pos_feats in self.pos_feats for layer_pos_feats in pos_feats]

    def forward_init(self, in_feat_maps, **kwargs):
        """
        Forward initialization method of the BLA module.

        Args:
            in_feat_maps (List): Input feature maps [num_in_maps] of shape [batch_size, feat_size, fH, fW].

        Returns:
            out_feat_maps (List): Output feature maps [num_out_maps] of shape [batch_size, feat_size, fH, fW].
        """

        # Get output feature maps
        out_feat_maps = self.in_proj(in_feat_maps)

        return out_feat_maps

    def forward_update(self, in_feat_maps, layer_id, **kwargs):
        """
        Forward update method of the BLA module.

        Args:
            in_feat_maps (List): Input feature maps [num_out_maps] of shape [batch_size, feat_size, fH, fW].

        Returns:
            out_feat_maps (List): Output feature maps [num_out_maps] of shape [batch_size, feat_size, fH, fW].
        """

        # Get number of maps and batch size
        num_maps = len(in_feat_maps)
        batch_size = len(in_feat_maps[0])

        # Permute input feature maps
        in_feat_maps = [in_feat_map.permute(0, 2, 3, 1) for in_feat_map in in_feat_maps]

        # Project input feature maps
        proj_maps = []
        zip_list = [in_feat_maps, self.attn_in_weights[layer_id], self.attn_in_biases[layer_id]]

        for in_feat_map, in_weight, in_bias in zip(*zip_list):
            proj_map = F.linear(in_feat_map, in_weight, in_bias)
            proj_maps.append(proj_map)

        # Perform self-attention
        self_attn_maps = []
        zip_list = [range(num_maps), proj_maps, self.feat_sizes, self.attn_scalings]
        zip_list = [*zip_list, self.num_heads_list, self.attn_out_weights[layer_id], self.attn_out_biases[layer_id]]

        for i, proj_map, f, scale, num_heads, out_weight, out_bias in zip(*zip_list):
            H, W = proj_map.shape[1:-1]

            query_map = scale*proj_map[:, :, :, :f]
            query_map = query_map.view(batch_size, H, W, num_heads, 1, -1)

            key_map = scale*proj_map[:, :, :, f:2*f]
            key_map = F.pad(key_map.permute(0, 3, 1, 2), (1, 1, 1, 1), mode='replicate').permute(0, 2, 3, 1)

            sizes = [batch_size, H, W, f, 3, 3]
            strides = [*key_map.stride(), key_map.stride()[1], key_map.stride()[2]]
            key_map = key_map.as_strided(sizes, strides).reshape(batch_size, H, W, f, 9)
            key_map = key_map + self.pos_feats[layer_id][i][:f, :] if self.pos_attn else key_map
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
        last_map_bools = [i == num_maps-1 for i in range(num_maps)]
        interpolation_kwargs = {'mode': 'bilinear', 'align_corners': True}
        top_down_attn_maps = []

        zip_list = [range(num_maps-1), proj_maps[:-1], proj_maps[1:], self.feat_sizes[:-1], self.feat_sizes[1:]]
        zip_list = [*zip_list, self.attn_scalings[:-1], last_map_bools[1:], self.num_heads_list[:-1]]
        zip_list = [*zip_list, self.attn_out_weights[layer_id][:-1], self.attn_out_biases[layer_id][:-1]]

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
            key_map = key_map + self.pos_feats[layer_id][i][f:2*f, :] if self.pos_attn else key_map
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

        top_down_attn_maps.append(torch.zeros_like(in_feat_maps[-1]))

        # Perform bottom-up cross-attention
        first_map_bools = [i == 0 for i in range(num_maps)]
        bottom_up_attn_maps = [torch.zeros_like(in_feat_maps[0])]

        zip_list = [range(1, num_maps), proj_maps[:-1], proj_maps[1:], self.feat_sizes[:-1], self.feat_sizes[1:]]
        zip_list = [*zip_list, self.attn_scalings[1:], first_map_bools[:-1], self.num_heads_list[1:]]
        zip_list = [*zip_list, self.attn_out_weights[layer_id][1:], self.attn_out_biases[layer_id][1:]]

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
            key_map = key_map + self.pos_feats[layer_id][i][-f:, :] if self.pos_attn else key_map
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
        zip_list = [in_feat_maps, self_attn_maps, top_down_attn_maps, bottom_up_attn_maps]
        zip_list = [*zip_list,  self.attn_layernorms[layer_id]]
        feat_maps = []

        for in_feat_map, attn_map1, attn_map2, attn_map3, attn_layernorm in zip(*zip_list):
            delta_feat_map = attn_map1 + attn_map2 + attn_map3
            feat_map = in_feat_map + F.dropout(delta_feat_map, self.attn_dropout, self.training)
            feat_map = attn_layernorm(feat_map)
            feat_maps.append(feat_map)

        # Update feature maps with feedforward network (FFN)
        out_feat_maps = []
        zip_list = [feat_maps, self.ffn_in_projs[layer_id], self.ffn_out_projs[layer_id]]
        zip_list = [*zip_list, self.ffn_layernorms[layer_id]]

        for feat_map, in_proj, out_proj, ffn_layernorm in zip(*zip_list):
            hidden_feat_map = F.dropout(F.relu(in_proj(feat_map)), self.ffn_dropout, self.training)
            out_feat_map = feat_map + F.dropout(out_proj(hidden_feat_map), self.ffn_dropout, self.training)
            out_feat_map = ffn_layernorm(out_feat_map)
            out_feat_maps.append(out_feat_map.permute(0, 3, 1, 2))

        return out_feat_maps

    def forward(self, in_feat_maps, step_id=None, **kwargs):
        """
        Forward method of the BLA module.

        Args:
            If step_id is None:
                in_feat_maps (List): Input feature maps [num_in_maps] of shape [batch_size, feat_size, fH, fW].
                step_id (None): Indicates full forward pass in a single step with initialization followed by updates.

            If step_id = 0:
                in_feat_maps (List): Input feature maps [num_in_maps] of shape [batch_size, feat_size, fH, fW].
                step_id (int): Zero integer indicating output feature maps initialization.

            If step_id > 0:
                in_feat_maps (List): Input feature maps [num_out_maps] of shape [batch_size, feat_size, fH, fW].
                step_id (int): Positive integer indicating output feature maps update with layer 'step_id-1'.

        Returns:
            out_feat_maps (List): Output feature maps [num_out_maps] of shape [batch_size, feat_size, fH, fW].
        """

        # Initializes and updates output feature maps
        if step_id is None:
            out_feat_maps = self.forward_init(in_feat_maps, **kwargs)

            for layer_id in range(self.num_layers):
                out_feat_maps = self.forward_update(out_feat_maps, layer_id=layer_id, **kwargs)

        # Initializes output feature maps
        elif step_id == 0:
            out_feat_maps = self.forward_init(in_feat_maps, **kwargs)

        # Updates output feature maps
        elif step_id > 0:
            out_feat_maps = self.forward_update(in_feat_maps, layer_id=step_id-1, **kwargs)

        return out_feat_maps


class BLAv1(nn.Module):
    """
    Class implementing the BLA (Bidirectional Local Attention) module (experimental version 1).

    Experimental version 1 of the BLA module disentangles the attention parameters for each attention mechanism.
    This allows the different attention parts to be disabled when desired. It also supports disabling the FFN part.

    Attributes:
        num_layers (int): Integer containing the number of BLA update layers.
        in_proj (Projector): Module computing the initial output feature maps from input feature maps.

        feat_sizes (List): List of size [num_out_maps] containing the feature size of each output map.
        num_heads_list (List): List of size [num_out_maps] containing the number of heads for each output map.
        attn_scalings (List): List of size [num_out_maps] containing attention scale factors.

        self_attn (bool): Boolean indicating whether self-attention mechanism is enabled or not.
        self_in_projs (nn.ModuleList): List [num_layers] of lists [num_out_maps] of self-attention in-projections.
        self_out_projs (nn.ModuleList): List [num_layers] of lists [num_out_maps] of self-attention out-projections.

        td_attn (bool): Boolean indicating whether top-down attention mechanism is enabled or not.
        td_q_in_projs (nn.ModuleList): List [num_layers] of lists [num_out_maps-1] of top-down query in-projections.
        td_kv_in_projs (nn.ModuleList): List [num_layers] of lists [num_out_maps-1] of top-down key-value projections.
        td_out_projs (nn.ModuleList): List [num_layers] of lists [num_out_maps-1] of top-down out-projections.

        bu_attn (bool): Boolean indicating whether bottom-up attention mechanism is enabled or not.
        bu_q_in_projs (nn.ModuleList): List [num_layers] of lists [num_out_maps-1] of bottom-up query in-projections.
        bu_kv_in_projs (nn.ModuleList): List [num_layers] of lists [num_out_maps-1] of bottom-up key-value projections.
        bu_out_projs (nn.ModuleList): List [num_layers] of lists [num_out_maps-1] of bottom-up out-projections.

        pos_attn (bool): Boolean indicating whether local position features are enabled or not.
        self_pos_feats (nn.ModuleList): List [num_layers] of lists [num_out_maps] of self-attention position features.
        td_pos_feats (nn.ModuleList): List [num_layers] of lists [num_out_maps-1] of top-down position features.
        bu_pos_feats (nn.ModuleList): List [num_layers] of lists [num_out_maps-1] of bottom-up position features.

        attn_dropout (float): Dropout probability (between 0 and 1) used during attention computation.
        attn_layernorms (nn.ModuleList): List [num_layers] with lists [num_out_maps] of attention layernorm modules.

        ffn (bool): Boolean indicating whether FFN layers are enabled or not.
        ffn_in_projs (nn.ModuleList): List [num_layers] with lists [num_out_maps] of FFN input projection modules.
        ffn_out_projs (nn.ModuleList): List [num_layers] with lists [num_out_maps] of FFN output projection modules.

        ffn_dropout (float): Dropout probability (between 0 and 1) used during FFN computation.
        ffn_layernorms (nn.ModuleList): List [num_layers] with lists [num_out_maps] of FFN layernorm modules.
    """

    def __init__(self, num_layers, in_feat_sizes, out_feat_sizes, attn_dict, ffn_dict):
        """
        Initializes the BLA module.

        Args:
            num_layers (int): Integer containing the number of BLA update layers.
            in_feat_sizes (List): List of size [num_in_maps] containing the feature size of each input map.
            out_feat_sizes (List): List of size [num_out_maps] containing the feature size of each output map.

            attn_dict (Dict): Attention dictionary containing following keys:
                - num_heads_list (List): list of size [num_out_maps] containing the number of heads per output map;
                - self_attn (bool): boolean indicating whether self-attention mechanism is enabled or not;
                - td_attn (bool): boolean indicating whether top-down attention mechanism is enabled or not;
                - bu_attn (bool): boolean indicating whether bottom-up attention mechanism is enabled or not;
                - pos_attn (bool): boolean indicating whether local position features are enabled or not;
                - dropout (float): dropout probability (between 0 and 1) used during attention computation.

            ffn_dict (Dict): FFN (feed-forward network) dictionary containing following keys:
                - ffn (bool): boolean indicating whether FFN layers are enabled or not;
                - size_multiplier (int): feature size multiplier used for FFN hidden layer feature sizes;
                - dropout (float): dropout probability (between 0 and 1) used during FFN computation.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Check inputs
        num_maps_delta = len(out_feat_sizes) - len(in_feat_sizes)
        msg = f"Fewer output maps than inputs map are not allowed (got {abs(num_maps_delta)} fewer)."
        assert num_maps_delta >= 0, msg

        msg = f"Only one more output map than input map is allowed (got {num_maps_delta} more)."
        assert num_maps_delta <= 1, msg

        num_heads_list = attn_dict['num_heads_list']
        check = all([feat_size % num_heads == 0 for feat_size, num_heads in zip(out_feat_sizes, num_heads_list)])
        msg = f"Feature sizes ({out_feat_sizes}) should be divisible by their number of heads ({num_heads_list})."
        assert check, msg

        # Set number of layers attribute
        self.num_layers = num_layers

        # Initialize input projector
        fixed_settings = {'proj_type': 'conv1', 'conv_stride': 1}
        proj_dicts = []

        for i in range(len(in_feat_sizes)):
            proj_dict = {'in_map_id': i, 'out_feat_size': out_feat_sizes[i], **fixed_settings}
            proj_dicts.append(proj_dict)

        if num_maps_delta == 1:
            proj_dict = {'in_map_id': len(in_feat_sizes)-1, 'out_feat_size': out_feat_sizes[-1]}
            proj_dict = {**proj_dict, 'proj_type': 'conv3', 'conv_stride': 2}
            proj_dicts.append(proj_dict)

        self.in_proj = Projector(in_feat_sizes, proj_dicts)

        # Set feature sizes, number of heads and attention scalings attributes
        self.feat_sizes = out_feat_sizes
        self.num_heads_list = num_heads_list
        self.attn_scalings = [float(f//num_heads)**-0.25 for f, num_heads in zip(out_feat_sizes, num_heads_list)]

        # Set boolean enable attributes
        self.self_attn = attn_dict['self_attn']
        self.td_attn = attn_dict['td_attn']
        self.bu_attn = attn_dict['bu_attn']
        self.pos_attn = attn_dict['pos_attn']
        self.ffn = ffn_dict['ffn']

        # Rename output feature sizes for readability
        fs = out_feat_sizes

        # Initialize self-attention modules if requested
        if self.self_attn:
            self.self_in_projs = nn.ModuleList([nn.Linear(f, 3*f) for f in fs])
            self.self_out_projs = nn.ModuleList([nn.Linear(f, f) for f in fs])

            if self.pos_attn:
                self.self_pos_feats = nn.ParameterList([Parameter(torch.empty(f, 9)) for f in fs])

        # Initialize top-down attention modules if requested
        if self.td_attn:
            self.td_q_in_projs = nn.ModuleList([nn.Linear(f, f) for f in fs[:-1]])
            self.td_kv_in_projs = nn.ModuleList([nn.Linear(f2, 2*f1) for f2, f1 in zip(fs[1:], fs[:-1])])
            self.td_out_projs = nn.ModuleList([nn.Linear(f, f) for f in fs[:-1]])

            if self.pos_attn:
                self.td_pos_feats = nn.ParameterList([Parameter(torch.empty(f, 9)) for f in fs[:-1]])

        # Initialize bottom-up attention modules if requested
        if self.bu_attn:
            self.bu_q_in_projs = nn.ModuleList([nn.Linear(f, f) for f in fs[1:]])
            self.bu_kv_in_projs = nn.ModuleList([nn.Linear(f0, 2*f1) for f0, f1 in zip(fs[:-1], fs[1:])])
            self.bu_out_projs = nn.ModuleList([nn.Linear(f, f) for f in fs[1:]])

            if self.pos_attn:
                self.bu_pos_feats = nn.ParameterList([Parameter(torch.empty(f, 9)) for f in fs[1:]])

        # Initialize attention dropout and layernorm attributes
        self.attn_dropout = attn_dict['dropout']
        self.attn_layernorms = nn.ModuleList([nn.LayerNorm(f) for f in fs])

        # Initialize FFN attributes
        if self.ffn:
            size_multiplier = ffn_dict['size_multiplier']
            self.ffn_in_projs = nn.ModuleList([nn.Linear(f, size_multiplier*f) for f in fs])
            self.ffn_out_projs = nn.ModuleList([nn.Linear(size_multiplier*f, f) for f in fs])

            self.ffn_dropout = ffn_dict['dropout']
            self.ffn_layernorms = nn.ModuleList([nn.LayerNorm(f) for f in fs])

        # Initialize separate update layers
        for attr_name in dir(self):
            module = getattr(self, attr_name)

            if isinstance(module, nn.ModuleList) or isinstance(module, nn.ParameterList):
                setattr(self, attr_name, nn.ModuleList([deepcopy(module) for _ in range(num_layers)]))

        # Set default initial values of module parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets module parameters to default initial values.
        """

        for name, param in self.named_parameters():
            if any(attn_type in name for attn_type in ['self', 'td', 'bu']):
                if 'projs' in name:
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

                elif 'pos_feats' in name:
                    nn.init.zeros_(param)

    def forward_init(self, in_feat_maps, **kwargs):
        """
        Forward initialization method of the BLA module.

        Args:
            in_feat_maps (List): Input feature maps [num_in_maps] of shape [batch_size, feat_size, fH, fW].

        Returns:
            out_feat_maps (List): Output feature maps [num_out_maps] of shape [batch_size, feat_size, fH, fW].
        """

        # Get output feature maps
        out_feat_maps = self.in_proj(in_feat_maps)

        return out_feat_maps

    def forward_update(self, in_feat_maps, layer_id, **kwargs):
        """
        Forward update method of the BLA module.

        Args:
            in_feat_maps (List): Input feature maps [num_out_maps] of shape [batch_size, feat_size, fH, fW].

        Returns:
            out_feat_maps (List): Output feature maps [num_out_maps] of shape [batch_size, feat_size, fH, fW].
        """

        # Get number of maps and batch size
        num_maps = len(in_feat_maps)
        batch_size = len(in_feat_maps[0])

        # Permute input feature maps to 'linear' format and initialize delta feature maps
        in_feat_maps = [in_feat_map.permute(0, 2, 3, 1) for in_feat_map in in_feat_maps]
        delta_feat_maps = [torch.zeros_like(in_feat_map) for in_feat_map in in_feat_maps]

        # Perform self-attention if requested
        if self.self_attn:
            zip_list = [range(num_maps), in_feat_maps, self.self_in_projs[layer_id], self.attn_scalings]
            zip_list = [*zip_list, self.num_heads_list, self.self_out_projs[layer_id]]

            for i, feat_map, in_proj, scale, num_heads, out_proj in zip(*zip_list):
                H, W, f = feat_map.shape[1:]
                proj_map = in_proj(feat_map)

                query_map = scale*proj_map[:, :, :, :f]
                query_map = query_map.view(batch_size, H, W, num_heads, 1, -1)

                key_map = scale*proj_map[:, :, :, f:2*f]
                key_map = F.pad(key_map.permute(0, 3, 1, 2), (1, 1, 1, 1), mode='replicate').permute(0, 2, 3, 1)

                sizes = [batch_size, H, W, f, 3, 3]
                strides = [*key_map.stride(), key_map.stride()[1], key_map.stride()[2]]
                key_map = key_map.as_strided(sizes, strides).reshape(batch_size, H, W, f, 9)
                key_map = key_map + self.self_pos_feats[layer_id][i] if self.pos_attn else key_map
                key_map = key_map.view(batch_size, H, W, num_heads, -1, 9)

                value_map = proj_map[:, :, :, 2*f:3*f]
                value_map = F.pad(value_map.permute(0, 3, 1, 2), (1, 1, 1, 1), mode='replicate').permute(0, 2, 3, 1)
                value_map = value_map.as_strided(sizes, strides).reshape(batch_size, H, W, f, 9)
                value_map = value_map.view(batch_size, H, W, num_heads, -1, 9)

                attn_weights = F.softmax(torch.matmul(query_map, key_map), dim=-1)
                weighted_map = torch.sum(attn_weights * value_map, dim=-1).view(batch_size, H, W, -1)
                delta_feat_maps[i] += out_proj(weighted_map)

        # Perform top-down attention if requested
        if self.td_attn:
            interpolation_kwargs = {'mode': 'bilinear', 'align_corners': True}

            zip_list = [range(num_maps-1), in_feat_maps[:-1], in_feat_maps[1:], self.td_q_in_projs[layer_id]]
            zip_list = [*zip_list, self.td_kv_in_projs[layer_id], self.attn_scalings[:-1], self.num_heads_list[:-1]]
            zip_list = [*zip_list, self.td_out_projs[layer_id]]

            for i, feat_map1, feat_map2, q_in_proj, kv_in_proj, scale, num_heads, out_proj in zip(*zip_list):
                H1, W1, f1 = feat_map1.shape[1:]
                H2, W2 = feat_map2.shape[1:-1]

                query_map = scale*q_in_proj(feat_map1[:, ::2, ::2, :])
                query_map = query_map.view(batch_size, H2, W2, num_heads, 1, -1)

                key_value_map = kv_in_proj(feat_map2)
                key_map = scale*key_value_map[:, :, :, :f1]
                key_map = F.pad(key_map.permute(0, 3, 1, 2), (1, 1, 1, 1), mode='replicate').permute(0, 2, 3, 1)

                sizes = [batch_size, H2, W2, f1, 3, 3]
                strides = [*key_map.stride(), key_map.stride()[1], key_map.stride()[2]]
                key_map = key_map.as_strided(sizes, strides).reshape(batch_size, H2, W2, f1, 9)
                key_map = key_map + self.td_pos_feats[layer_id][i] if self.pos_attn else key_map
                key_map = key_map.view(batch_size, H2, W2, num_heads, -1, 9)

                value_map = key_value_map[:, :, :, f1:2*f1]
                value_map = F.pad(value_map.permute(0, 3, 1, 2), (1, 1, 1, 1), mode='replicate').permute(0, 2, 3, 1)
                value_map = value_map.as_strided(sizes, strides).reshape(batch_size, H2, W2, f1, 9)
                value_map = value_map.view(batch_size, H2, W2, num_heads, -1, 9)

                attn_weights = F.softmax(torch.matmul(query_map, key_map), dim=-1)
                weighted_map = torch.sum(attn_weights * value_map, dim=-1).view(batch_size, H2, W2, -1)
                top_down_attn_map = out_proj(weighted_map)

                pH, pW = (int(H1 % 2 == 0), int(W1 % 2 == 0))
                top_down_attn_map = top_down_attn_map.permute(0, 3, 1, 2)
                top_down_attn_map = F.pad(top_down_attn_map, (0, pW, 0, pH), mode='replicate')
                top_down_attn_map = F.interpolate(top_down_attn_map, size=(H1+pH, W1+pW), **interpolation_kwargs)
                delta_feat_maps[i] += top_down_attn_map[:, :, :H1, :W1].permute(0, 2, 3, 1)

        # Perform bottom-up attention if requested
        if self.bu_attn:
            zip_list = [range(num_maps), in_feat_maps[:-1], in_feat_maps[1:], self.bu_q_in_projs[layer_id]]
            zip_list = [*zip_list, self.bu_kv_in_projs[layer_id], self.attn_scalings[1:], self.num_heads_list[1:]]
            zip_list = [*zip_list, self.bu_out_projs[layer_id]]

            for i, feat_map0, feat_map1, q_in_proj, kv_in_proj, scale, num_heads, out_proj in zip(*zip_list):
                H0, W0 = feat_map0.shape[1:-1]
                H1, W1, f1 = feat_map1.shape[1:]

                query_map = scale*q_in_proj(feat_map1)
                query_map = query_map.view(batch_size, H1, W1, num_heads, 1, -1)

                key_value_map = kv_in_proj(feat_map0)
                key_map = scale*key_value_map[:, :, :, :f1]

                pH, pW = (H0 % 2, W0 % 2)
                key_map = F.pad(key_map.permute(0, 3, 1, 2), (1, pW, 1, pH), mode='replicate').permute(0, 2, 3, 1)

                sizes = [batch_size, H1, W1, f1, 3, 3]
                s0, s1, s2, s3 = key_map.stride()
                strides = [s0, 2*s1, 2*s2, s3, s1, s2]
                key_map = key_map.as_strided(sizes, strides).reshape(batch_size, H1, W1, f1, 9)
                key_map = key_map + self.bu_pos_feats[layer_id][i] if self.pos_attn else key_map
                key_map = key_map.view(batch_size, H1, W1, num_heads, -1, 9)

                value_map = key_value_map[:, :, :, f1:2*f1]
                value_map = F.pad(value_map.permute(0, 3, 1, 2), (1, pW, 1, pH), mode='replicate').permute(0, 2, 3, 1)
                value_map = value_map.as_strided(sizes, strides).reshape(batch_size, H1, W1, f1, 9)
                value_map = value_map.view(batch_size, H1, W1, num_heads, -1, 9)

                attn_weights = F.softmax(torch.matmul(query_map, key_map), dim=-1)
                weighted_map = torch.sum(attn_weights * value_map, dim=-1).view(batch_size, H1, W1, -1)
                delta_feat_maps[i+1] += out_proj(weighted_map)

        # Update input feature maps with attention-based delta maps
        out_feat_maps = []
        zip_list = [in_feat_maps, delta_feat_maps, self.attn_layernorms[layer_id]]

        for in_feat_map, delta_feat_map, attn_layernorm in zip(*zip_list):
            out_feat_map = in_feat_map + F.dropout(delta_feat_map, self.attn_dropout, self.training)
            out_feat_map = attn_layernorm(out_feat_map)
            out_feat_maps.append(out_feat_map)

        # Update output feature maps with feedforward network (FFN) if requested
        if self.ffn:
            zip_list = [range(num_maps), out_feat_maps, self.ffn_in_projs[layer_id], self.ffn_out_projs[layer_id]]
            zip_list = [*zip_list, self.ffn_layernorms[layer_id]]

            for i, feat_map, in_proj, out_proj, ffn_layernorm in zip(*zip_list):
                hidden_feat_map = F.dropout(F.relu(in_proj(feat_map)), self.ffn_dropout, self.training)
                feat_map = feat_map + F.dropout(out_proj(hidden_feat_map), self.ffn_dropout, self.training)
                out_feat_maps[i] = ffn_layernorm(feat_map)

        # Permute output feature maps back to 'convolution' format
        out_feat_maps = [out_feat_map.permute(0, 3, 1, 2) for out_feat_map in out_feat_maps]

        return out_feat_maps

    def forward(self, in_feat_maps, step_id=None, **kwargs):
        """
        Forward method of the BLA module.

        Args:
            If step_id is None:
                in_feat_maps (List): Input feature maps [num_in_maps] of shape [batch_size, feat_size, fH, fW].
                step_id (None): Indicates full forward pass in a single step with initialization followed by updates.

            If step_id = 0:
                in_feat_maps (List): Input feature maps [num_in_maps] of shape [batch_size, feat_size, fH, fW].
                step_id (int): Zero integer indicating output feature maps initialization.

            If step_id > 0:
                in_feat_maps (List): Input feature maps [num_out_maps] of shape [batch_size, feat_size, fH, fW].
                step_id (int): Positive integer indicating output feature maps update with layer 'step_id-1'.

        Returns:
            out_feat_maps (List): Output feature maps [num_out_maps] of shape [batch_size, feat_size, fH, fW].
        """

        # Initializes and updates output feature maps
        if step_id is None:
            out_feat_maps = self.forward_init(in_feat_maps, **kwargs)

            for layer_id in range(self.num_layers):
                out_feat_maps = self.forward_update(out_feat_maps, layer_id=layer_id, **kwargs)

        # Initializes output feature maps
        elif step_id == 0:
            out_feat_maps = self.forward_init(in_feat_maps, **kwargs)

        # Updates output feature maps
        elif step_id > 0:
            out_feat_maps = self.forward_update(in_feat_maps, layer_id=step_id-1, **kwargs)

        return out_feat_maps


class BLAv2(nn.Module):
    """
    Class implementing the BLA (Bidirectional Local Attention) module (experimental version 2).

    Same as experimental version 1, but with linear layers replaced by convolutions with kernel size 1.

    Attributes:
        num_layers (int): Integer containing the number of BLA update layers.
        in_proj (Projector): Module computing the initial output feature maps from input feature maps.

        feat_sizes (List): List of size [num_out_maps] containing the feature size of each output map.
        num_heads_list (List): List of size [num_out_maps] containing the number of heads for each output map.
        attn_scalings (List): List of size [num_out_maps] containing attention scale factors.

        self_attn (bool): Boolean indicating whether self-attention mechanism is enabled or not.
        self_in_projs (nn.ModuleList): List [num_layers] of lists [num_out_maps] of self-attention in-projections.
        self_out_projs (nn.ModuleList): List [num_layers] of lists [num_out_maps] of self-attention out-projections.

        td_attn (bool): Boolean indicating whether top-down attention mechanism is enabled or not.
        td_q_in_projs (nn.ModuleList): List [num_layers] of lists [num_out_maps-1] of top-down query in-projections.
        td_kv_in_projs (nn.ModuleList): List [num_layers] of lists [num_out_maps-1] of top-down key-value projections.
        td_out_projs (nn.ModuleList): List [num_layers] of lists [num_out_maps-1] of top-down out-projections.

        bu_attn (bool): Boolean indicating whether bottom-up attention mechanism is enabled or not.
        bu_q_in_projs (nn.ModuleList): List [num_layers] of lists [num_out_maps-1] of bottom-up query in-projections.
        bu_kv_in_projs (nn.ModuleList): List [num_layers] of lists [num_out_maps-1] of bottom-up key-value projections.
        bu_out_projs (nn.ModuleList): List [num_layers] of lists [num_out_maps-1] of bottom-up out-projections.

        pos_attn (bool): Boolean indicating whether local position features are enabled or not.
        self_pos_feats (nn.ModuleList): List [num_layers] of lists [num_out_maps] of self-attention position features.
        td_pos_feats (nn.ModuleList): List [num_layers] of lists [num_out_maps-1] of top-down position features.
        bu_pos_feats (nn.ModuleList): List [num_layers] of lists [num_out_maps-1] of bottom-up position features.

        attn_dropout (float): Dropout probability (between 0 and 1) used during attention computation.
        attn_layernorms (nn.ModuleList): List [num_layers] with lists [num_out_maps] of attention layernorm modules.

        ffn (bool): Boolean indicating whether FFN layers are enabled or not.
        ffn_in_projs (nn.ModuleList): List [num_layers] with lists [num_out_maps] of FFN input projection modules.
        ffn_out_projs (nn.ModuleList): List [num_layers] with lists [num_out_maps] of FFN output projection modules.

        ffn_dropout (float): Dropout probability (between 0 and 1) used during FFN computation.
        ffn_layernorms (nn.ModuleList): List [num_layers] with lists [num_out_maps] of FFN layernorm modules.
    """

    def __init__(self, num_layers, in_feat_sizes, out_feat_sizes, attn_dict, ffn_dict):
        """
        Initializes the BLA module.

        Args:
            num_layers (int): Integer containing the number of BLA update layers.
            in_feat_sizes (List): List of size [num_in_maps] containing the feature size of each input map.
            out_feat_sizes (List): List of size [num_out_maps] containing the feature size of each output map.

            attn_dict (Dict): Attention dictionary containing following keys:
                - num_heads_list (List): list of size [num_out_maps] containing the number of heads per output map;
                - self_attn (bool): boolean indicating whether self-attention mechanism is enabled or not;
                - td_attn (bool): boolean indicating whether top-down attention mechanism is enabled or not;
                - bu_attn (bool): boolean indicating whether bottom-up attention mechanism is enabled or not;
                - pos_attn (bool): boolean indicating whether local position features are enabled or not;
                - dropout (float): dropout probability (between 0 and 1) used during attention computation.

            ffn_dict (Dict): FFN (feed-forward network) dictionary containing following keys:
                - ffn (bool): boolean indicating whether FFN layers are enabled or not;
                - size_multiplier (int): feature size multiplier used for FFN hidden layer feature sizes;
                - dropout (float): dropout probability (between 0 and 1) used during FFN computation.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Check inputs
        num_maps_delta = len(out_feat_sizes) - len(in_feat_sizes)
        msg = f"Fewer output maps than inputs map are not allowed (got {abs(num_maps_delta)} fewer)."
        assert num_maps_delta >= 0, msg

        msg = f"Only one more output map than input map is allowed (got {num_maps_delta} more)."
        assert num_maps_delta <= 1, msg

        num_heads_list = attn_dict['num_heads_list']
        check = all([feat_size % num_heads == 0 for feat_size, num_heads in zip(out_feat_sizes, num_heads_list)])
        msg = f"Feature sizes ({out_feat_sizes}) should be divisible by their number of heads ({num_heads_list})."
        assert check, msg

        # Set number of layers attribute
        self.num_layers = num_layers

        # Initialize input projector
        fixed_settings = {'proj_type': 'conv1', 'conv_stride': 1}
        proj_dicts = []

        for i in range(len(in_feat_sizes)):
            proj_dict = {'in_map_id': i, 'out_feat_size': out_feat_sizes[i], **fixed_settings}
            proj_dicts.append(proj_dict)

        if num_maps_delta == 1:
            proj_dict = {'in_map_id': len(in_feat_sizes)-1, 'out_feat_size': out_feat_sizes[-1]}
            proj_dict = {**proj_dict, 'proj_type': 'conv3', 'conv_stride': 2}
            proj_dicts.append(proj_dict)

        self.in_proj = Projector(in_feat_sizes, proj_dicts)

        # Set feature sizes, number of heads and attention scalings attributes
        self.feat_sizes = out_feat_sizes
        self.num_heads_list = num_heads_list
        self.attn_scalings = [float(f//num_heads)**-0.25 for f, num_heads in zip(out_feat_sizes, num_heads_list)]

        # Set boolean enable attributes
        self.self_attn = attn_dict['self_attn']
        self.td_attn = attn_dict['td_attn']
        self.bu_attn = attn_dict['bu_attn']
        self.pos_attn = attn_dict['pos_attn']
        self.ffn = ffn_dict['ffn']

        # Rename output feature sizes for readability
        fs = out_feat_sizes

        # Initialize self-attention modules if requested
        if self.self_attn:
            self.self_in_projs = nn.ModuleList([nn.Conv2d(f, 3*f, 1) for f in fs])
            self.self_out_projs = nn.ModuleList([nn.Conv2d(f, f, 1) for f in fs])

            if self.pos_attn:
                self.self_pos_feats = nn.ParameterList([Parameter(torch.empty(f, 9)) for f in fs])

        # Initialize top-down attention modules if requested
        if self.td_attn:
            self.td_q_in_projs = nn.ModuleList([nn.Conv2d(f, f, 1) for f in fs[:-1]])
            self.td_kv_in_projs = nn.ModuleList([nn.Conv2d(f2, 2*f1, 1) for f2, f1 in zip(fs[1:], fs[:-1])])
            self.td_out_projs = nn.ModuleList([nn.Conv2d(f, f, 1) for f in fs[:-1]])

            if self.pos_attn:
                self.td_pos_feats = nn.ParameterList([Parameter(torch.empty(f, 9)) for f in fs[:-1]])

        # Initialize bottom-up attention modules if requested
        if self.bu_attn:
            self.bu_q_in_projs = nn.ModuleList([nn.Conv2d(f, f, 1) for f in fs[1:]])
            self.bu_kv_in_projs = nn.ModuleList([nn.Conv2d(f0, 2*f1, 1) for f0, f1 in zip(fs[:-1], fs[1:])])
            self.bu_out_projs = nn.ModuleList([nn.Conv2d(f, f, 1) for f in fs[1:]])

            if self.pos_attn:
                self.bu_pos_feats = nn.ParameterList([Parameter(torch.empty(f, 9)) for f in fs[1:]])

        # Initialize attention dropout and layernorm attributes
        self.attn_dropout = attn_dict['dropout']
        self.attn_layernorms = nn.ModuleList([nn.LayerNorm(f) for f in fs])

        # Initialize FFN attributes
        if self.ffn:
            size_multiplier = ffn_dict['size_multiplier']
            self.ffn_in_projs = nn.ModuleList([nn.Conv2d(f, size_multiplier*f, 1) for f in fs])
            self.ffn_out_projs = nn.ModuleList([nn.Conv2d(size_multiplier*f, f, 1) for f in fs])

            self.ffn_dropout = ffn_dict['dropout']
            self.ffn_layernorms = nn.ModuleList([nn.LayerNorm(f) for f in fs])

        # Initialize separate update layers
        for attr_name in dir(self):
            module = getattr(self, attr_name)

            if isinstance(module, nn.ModuleList) or isinstance(module, nn.ParameterList):
                setattr(self, attr_name, nn.ModuleList([deepcopy(module) for _ in range(num_layers)]))

        # Set default initial values of module parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets module parameters to default initial values.
        """

        for name, param in self.named_parameters():
            if any(attn_type in name for attn_type in ['self', 'td', 'bu']):
                if 'projs' in name:
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

                elif 'pos_feats' in name:
                    nn.init.zeros_(param)

    def forward_init(self, in_feat_maps, **kwargs):
        """
        Forward initialization method of the BLA module.

        Args:
            in_feat_maps (List): Input feature maps [num_in_maps] of shape [batch_size, feat_size, fH, fW].

        Returns:
            out_feat_maps (List): Output feature maps [num_out_maps] of shape [batch_size, feat_size, fH, fW].
        """

        # Get output feature maps
        out_feat_maps = self.in_proj(in_feat_maps)

        return out_feat_maps

    def forward_update(self, in_feat_maps, layer_id, **kwargs):
        """
        Forward update method of the BLA module.

        Args:
            in_feat_maps (List): Input feature maps [num_out_maps] of shape [batch_size, feat_size, fH, fW].

        Returns:
            out_feat_maps (List): Output feature maps [num_out_maps] of shape [batch_size, feat_size, fH, fW].
        """

        # Get number of maps and batch size
        num_maps = len(in_feat_maps)
        batch_size = len(in_feat_maps[0])

        # Initialize delta feature maps
        delta_feat_maps = [torch.zeros_like(in_feat_map) for in_feat_map in in_feat_maps]

        # Perform self-attention if requested
        if self.self_attn:
            zip_list = [range(num_maps), in_feat_maps, self.self_in_projs[layer_id], self.attn_scalings]
            zip_list = [*zip_list, self.num_heads_list, self.self_out_projs[layer_id]]

            for i, feat_map, in_proj, scale, num_heads, out_proj in zip(*zip_list):
                f, H, W = feat_map.shape[1:]
                proj_map = in_proj(feat_map)

                query_map = scale*proj_map[:, :f, :, :].permute(0, 2, 3, 1)
                query_map = query_map.view(batch_size, H, W, num_heads, 1, -1)

                key_map = scale*proj_map[:, f:2*f, :, :]
                key_map = F.pad(key_map, (1, 1, 1, 1), mode='replicate').permute(0, 2, 3, 1)

                sizes = [batch_size, H, W, f, 3, 3]
                strides = [*key_map.stride(), key_map.stride()[1], key_map.stride()[2]]
                key_map = key_map.as_strided(sizes, strides).reshape(batch_size, H, W, f, 9)
                key_map = key_map + self.self_pos_feats[layer_id][i] if self.pos_attn else key_map
                key_map = key_map.view(batch_size, H, W, num_heads, -1, 9)

                value_map = proj_map[:, 2*f:3*f, :, :]
                value_map = F.pad(value_map, (1, 1, 1, 1), mode='replicate').permute(0, 2, 3, 1)
                value_map = value_map.as_strided(sizes, strides).reshape(batch_size, H, W, f, 9)
                value_map = value_map.view(batch_size, H, W, num_heads, -1, 9)

                attn_weights = F.softmax(torch.matmul(query_map, key_map), dim=-1)
                weighted_map = torch.sum(attn_weights * value_map, dim=-1).view(batch_size, H, W, -1)
                delta_feat_maps[i] += out_proj(weighted_map.permute(0, 3, 1, 2))

        # Perform top-down attention if requested
        if self.td_attn:
            interpolation_kwargs = {'mode': 'bilinear', 'align_corners': True}

            zip_list = [range(num_maps-1), in_feat_maps[:-1], in_feat_maps[1:], self.td_q_in_projs[layer_id]]
            zip_list = [*zip_list, self.td_kv_in_projs[layer_id], self.attn_scalings[:-1], self.num_heads_list[:-1]]
            zip_list = [*zip_list, self.td_out_projs[layer_id]]

            for i, feat_map1, feat_map2, q_in_proj, kv_in_proj, scale, num_heads, out_proj in zip(*zip_list):
                f1, H1, W1 = feat_map1.shape[1:]
                H2, W2 = feat_map2.shape[2:]

                query_map = scale*q_in_proj(feat_map1[:, :, ::2, ::2]).permute(0, 2, 3, 1)
                query_map = query_map.view(batch_size, H2, W2, num_heads, 1, -1)

                key_value_map = kv_in_proj(feat_map2)
                key_map = scale*key_value_map[:, :f1, :, :]
                key_map = F.pad(key_map, (1, 1, 1, 1), mode='replicate').permute(0, 2, 3, 1)

                sizes = [batch_size, H2, W2, f1, 3, 3]
                strides = [*key_map.stride(), key_map.stride()[1], key_map.stride()[2]]
                key_map = key_map.as_strided(sizes, strides).reshape(batch_size, H2, W2, f1, 9)
                key_map = key_map + self.td_pos_feats[layer_id][i] if self.pos_attn else key_map
                key_map = key_map.view(batch_size, H2, W2, num_heads, -1, 9)

                value_map = key_value_map[:, f1:2*f1, :, :]
                value_map = F.pad(value_map, (1, 1, 1, 1), mode='replicate').permute(0, 2, 3, 1)
                value_map = value_map.as_strided(sizes, strides).reshape(batch_size, H2, W2, f1, 9)
                value_map = value_map.view(batch_size, H2, W2, num_heads, -1, 9)

                attn_weights = F.softmax(torch.matmul(query_map, key_map), dim=-1)
                weighted_map = torch.sum(attn_weights * value_map, dim=-1).view(batch_size, H2, W2, -1)
                top_down_attn_map = out_proj(weighted_map.permute(0, 3, 1, 2))

                pH, pW = (int(H1 % 2 == 0), int(W1 % 2 == 0))
                top_down_attn_map = F.pad(top_down_attn_map, (0, pW, 0, pH), mode='replicate')
                top_down_attn_map = F.interpolate(top_down_attn_map, size=(H1+pH, W1+pW), **interpolation_kwargs)
                delta_feat_maps[i] += top_down_attn_map[:, :, :H1, :W1]

        # Perform bottom-up attention if requested
        if self.bu_attn:
            zip_list = [range(num_maps), in_feat_maps[:-1], in_feat_maps[1:], self.bu_q_in_projs[layer_id]]
            zip_list = [*zip_list, self.bu_kv_in_projs[layer_id], self.attn_scalings[1:], self.num_heads_list[1:]]
            zip_list = [*zip_list, self.bu_out_projs[layer_id]]

            for i, feat_map0, feat_map1, q_in_proj, kv_in_proj, scale, num_heads, out_proj in zip(*zip_list):
                H0, W0 = feat_map0.shape[2:]
                f1, H1, W1 = feat_map1.shape[1:]

                query_map = scale*q_in_proj(feat_map1).permute(0, 2, 3, 1)
                query_map = query_map.view(batch_size, H1, W1, num_heads, 1, -1)

                key_value_map = kv_in_proj(feat_map0)
                key_map = scale*key_value_map[:, :f1, :, :]
                key_map = F.pad(key_map, (1, W0 % 2, 1, H0 % 2), mode='replicate').permute(0, 2, 3, 1)

                sizes = [batch_size, H1, W1, f1, 3, 3]
                s0, s1, s2, s3 = key_map.stride()
                strides = [s0, 2*s1, 2*s2, s3, s1, s2]
                key_map = key_map.as_strided(sizes, strides).reshape(batch_size, H1, W1, f1, 9)
                key_map = key_map + self.bu_pos_feats[layer_id][i] if self.pos_attn else key_map
                key_map = key_map.view(batch_size, H1, W1, num_heads, -1, 9)

                value_map = key_value_map[:, f1:2*f1, :, :]
                value_map = F.pad(value_map, (1, W0 % 2, 1, H0 % 2), mode='replicate').permute(0, 2, 3, 1)
                value_map = value_map.as_strided(sizes, strides).reshape(batch_size, H1, W1, f1, 9)
                value_map = value_map.view(batch_size, H1, W1, num_heads, -1, 9)

                attn_weights = F.softmax(torch.matmul(query_map, key_map), dim=-1)
                weighted_map = torch.sum(attn_weights * value_map, dim=-1).view(batch_size, H1, W1, -1)
                delta_feat_maps[i+1] += out_proj(weighted_map.permute(0, 3, 1, 2))

        # Update input feature maps with attention-based delta maps
        out_feat_maps = []
        zip_list = [in_feat_maps, delta_feat_maps, self.attn_layernorms[layer_id]]

        for in_feat_map, delta_feat_map, attn_layernorm in zip(*zip_list):
            out_feat_map = in_feat_map + F.dropout(delta_feat_map, self.attn_dropout, self.training)
            out_feat_map = attn_layernorm(out_feat_map.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            out_feat_maps.append(out_feat_map)

        # Update output feature maps with feedforward network (FFN) if requested
        if self.ffn:
            zip_list = [range(num_maps), out_feat_maps, self.ffn_in_projs[layer_id], self.ffn_out_projs[layer_id]]
            zip_list = [*zip_list, self.ffn_layernorms[layer_id]]

            for i, feat_map, in_proj, out_proj, ffn_layernorm in zip(*zip_list):
                hidden_feat_map = F.dropout(F.relu(in_proj(feat_map)), self.ffn_dropout, self.training)
                feat_map = feat_map + F.dropout(out_proj(hidden_feat_map), self.ffn_dropout, self.training)
                out_feat_maps[i] = ffn_layernorm(feat_map.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return out_feat_maps

    def forward(self, in_feat_maps, step_id=None, **kwargs):
        """
        Forward method of the BLA module.

        Args:
            If step_id is None:
                in_feat_maps (List): Input feature maps [num_in_maps] of shape [batch_size, feat_size, fH, fW].
                step_id (None): Indicates full forward pass in a single step with initialization followed by updates.

            If step_id = 0:
                in_feat_maps (List): Input feature maps [num_in_maps] of shape [batch_size, feat_size, fH, fW].
                step_id (int): Zero integer indicating output feature maps initialization.

            If step_id > 0:
                in_feat_maps (List): Input feature maps [num_out_maps] of shape [batch_size, feat_size, fH, fW].
                step_id (int): Positive integer indicating output feature maps update with layer 'step_id-1'.

        Returns:
            out_feat_maps (List): Output feature maps [num_out_maps] of shape [batch_size, feat_size, fH, fW].
        """

        # Initializes and updates output feature maps
        if step_id is None:
            out_feat_maps = self.forward_init(in_feat_maps, **kwargs)

            for layer_id in range(self.num_layers):
                out_feat_maps = self.forward_update(out_feat_maps, layer_id=layer_id, **kwargs)

        # Initializes output feature maps
        elif step_id == 0:
            out_feat_maps = self.forward_init(in_feat_maps, **kwargs)

        # Updates output feature maps
        elif step_id > 0:
            out_feat_maps = self.forward_update(in_feat_maps, layer_id=step_id-1, **kwargs)

        return out_feat_maps


def build_bla(args):
    """
    Build BLA module from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        bla (nn.Module): The specified BLA module.
    """

    # Get genaral build arguments
    num_layers = args.bla_num_layers
    in_feat_sizes = args.backbone_feat_sizes

    min_id, max_id = (args.min_downsampling, args.max_downsampling)
    out_feat_sizes = [min((args.bla_base_feat_size * 2**i, args.bla_max_feat_size)) for i in range(min_id, max_id+1)]

    num_heads_list = [min((args.bla_base_num_heads * 2**i, args.bla_max_num_heads)) for i in range(min_id, max_id+1)]
    pos_attn = ~args.bla_disable_pos
    attn_dropout = args.bla_attn_dropout
    attn_dict = {'num_heads_list': num_heads_list, 'pos_attn': pos_attn, 'dropout': attn_dropout}

    ffn_size_multiplier = args.bla_ffn_size_multiplier
    ffn_dropout = args.bla_ffn_dropout
    ffn_dict = {'size_multiplier': ffn_size_multiplier, 'dropout': ffn_dropout}

    # Build BLA module of requested version
    if args.bla_version == 'main':
        bla = BLA(num_layers, in_feat_sizes, out_feat_sizes, attn_dict, ffn_dict)

    elif args.bla_version in ['v1', 'v2']:
        self_attn = not args.bla_disable_self
        td_attn = not args.bla_disable_td
        bu_attn = not args.bla_disable_bu

        attn_dict = {**attn_dict, 'self_attn': self_attn, 'td_attn': td_attn, 'bu_attn': bu_attn}
        ffn_dict = {**ffn_dict, 'ffn': not args.bla_disable_ffn}

        if args.bla_version == 'v1':
            bla = BLAv1(num_layers, in_feat_sizes, out_feat_sizes, attn_dict, ffn_dict)
        elif args.bla_version == 'v2':
            bla = BLAv2(num_layers, in_feat_sizes, out_feat_sizes, attn_dict, ffn_dict)

    return bla
