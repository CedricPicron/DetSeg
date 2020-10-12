"""
Decoder modules and build function.
"""
from collections import OrderedDict
import copy
import random

import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class GlobalDecoder(nn.Module):
    """
    Class implementing the GlobalDecoder module.

    Attributes:
        feat_dim (int): Feature dimension used in the decoder.
        num_slots (int): Number of slots per image.
        num_layers (int): Number of concatenated decoder layers.
        return_all (bool): Whether to return all decoder predictions.
        layers (nn.ModulesList): List of decoder layers being concatenated.
        norm (nn.LayerNorm): Final layer normalization before output.
        slot_embeds (nn.Embedding): Learned positional slot embeddings.
        trained (bool): Whether global decoder is trained or not.
    """

    def __init__(self, decoder_layer, decoder_dict, feat_dim, num_slots):
        """
        Initializes the GlobalDecoder module.

        Args:
            decoder_layer (nn.Module): Decoder layer module to be concatenated.
            decoder_dict (Dict): Dictionary containing the decoder parameters:
                - num_layers (int): number of concatenated decoder layers;
                - return_all (bool): whether to return all decoder predictions;
                - train (bool): whether decoder should be trained or not.
            feat_dim (int): Feature dimension used in the decoder.
            num_slots (int): Number of slots per image.
        """

        super().__init__()
        self.feat_dim = feat_dim
        self.num_slots = num_slots
        self.num_layers = decoder_dict['num_layers']
        self.return_all = decoder_dict['return_all']

        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(self.num_layers)])
        self.norm = nn.LayerNorm(feat_dim)
        self.requires_grad_(decoder_dict['train'])
        self.slot_embeds = nn.Embedding(num_slots, feat_dim)
        self.trained = decoder_dict['train']

    def load_from_original_detr(self, state_dict):
        """
        Loads decoder from state_dict of an original Facebook DETR model.

        state_dict (Dict): Dictionary containing Facebook's model parameters and persistent buffers.
        """

        decoder_identifier = 'transformer.decoder.'
        identifier_length = len(decoder_identifier)
        decoder_state_dict = OrderedDict()

        for original_name, state in state_dict.items():
            if decoder_identifier in original_name:
                new_name = original_name[identifier_length:]
                decoder_state_dict[new_name] = state

        decoder_state_dict['slot_embeds.weight'] = state_dict['query_embed.weight']
        self.load_state_dict(decoder_state_dict)

    def reset_parameters(self):
        """
        Resets all multi-dimensional module parameters according to the uniform xavier initialization.
        """

        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, features, feature_masks, pos_encodings):
        """
        Forward method of the GlobalDecoder module.

        Args:
            features (FloatTensor): Features of shape [H*W, batch_size, feat_dim].
            feature_masks (BoolTensor): Boolean masks encoding inactive features of shape [batch_size, H, W].
            pos_encodings (FloatTensor): Position encodings of shape [H*W, batch_size, feat_dim].

        Returns:
            slots (FloatTensor): Object slots of shape [num_pred_sets, num_slots_total, feat_dim].
            batch_idx (IntTensor): Slot batch indices (in ascending order) of shape [num_pred_sets, num_slots_total].
        """

        batch_size = feature_masks.shape[0]
        feature_masks = feature_masks.flatten(1)
        slot_embeds = self.slot_embeds.weight.unsqueeze(1).repeat(1, batch_size, 1)
        slots = torch.zeros_like(slot_embeds)
        slots_list = []

        for layer in self.layers:
            slots = layer(slots, slot_embeds, features, feature_masks, pos_encodings)
            slots_list.append(self.norm(slots)) if self.return_all else None

        if not self.return_all:
            slots_list.append(self.norm(slots))

        num_pred_sets = len(slots_list)
        slots_list.reverse()
        slots = torch.stack(slots_list, dim=0)
        slots = slots.transpose(1, 2).reshape(num_pred_sets, batch_size*self.num_slots, self.feat_dim)

        batch_idx = torch.arange(batch_size*self.num_slots, device=slots.device) // self.num_slots
        batch_idx = batch_idx[None, :].expand(num_pred_sets, -1)

        return slots, batch_idx, None


class GlobalDecoderLayer(nn.Module):
    """
    Class implementing the GlobalDecoderLayer module.

    Decoder layer with global multi-head self-attention, followed by cross-attention and a feedforward network (FFN).

    Attributes:
        self_attn (nn.MultiheadAtttenion): Multi-head attention module used for self-attenion.
        dropout1 (nn.Dropout): Dropout module after self-attention.
        norm1 (nn.LayerNorm): Layernorm module after self-attention skip connection.
        multihead_attn (nn.MultiheadAtttenion): Multi-head attention module used for cross-attenion.
        dropout2 (nn.Dropout): Dropout module after cross-attention.
        norm2 (nn.LayerNorm): Layernorm module after cross-attention skip connection.
        linear1 (nn.Linear): First FFN linear layer.
        dropout (nn.Dropout): Dropout module after first FFN layer.
        linear2 (nn.Linear): Second FFN linear layer.
        dropout3 (nn.Dropout): Dropout module after second FFN layer.
        norm3 (nn.LayerNorm): Layernorm module after FFN skip connection.
    """

    def __init__(self, feat_dim, mha_dict, ffn_dict):
        """
        Initializes the GlobalDecoderLayer module.

        Args:
            feat_dim (int): feat_dim (int): Feature dimension used in the decoder layer.
            mha_dict (Dict): Dict containing parameters of the MultiheadAttention module:
                - num_heads (int): number of attention heads;
                - dropout (float): dropout probability used throughout the MultiheadAttention module.
            ffn_dict (Dict): Dict containing parameters of the FFN module:
                - hidden_dim (int): number of hidden dimensions in the FFN hidden layers;
                - dropout (float): dropout probability used throughout the FFN module.
        """

        # Intialization of default nn.Module
        super().__init__()

        # Initialization of multi-head attention module for self-attention
        num_heads = mha_dict['num_heads']
        mha_dropout = mha_dict['dropout']

        self.self_attn = nn.MultiheadAttention(feat_dim, num_heads, dropout=mha_dropout)
        self.dropout1 = nn.Dropout(mha_dropout)
        self.norm1 = nn.LayerNorm(feat_dim)

        # Initialization of multi-head attention module for cross-attention
        self.multihead_attn = nn.MultiheadAttention(feat_dim, num_heads, dropout=mha_dropout)
        self.dropout2 = nn.Dropout(mha_dropout)
        self.norm2 = nn.LayerNorm(feat_dim)

        # Initialization of feedforward network (FFN) module
        ffn_hidden_dim = ffn_dict['hidden_dim']
        ffn_dropout = ffn_dict['dropout']

        self.linear1 = nn.Linear(feat_dim, ffn_hidden_dim)
        self.dropout = nn.Dropout(ffn_dropout)
        self.linear2 = nn.Linear(ffn_hidden_dim, feat_dim)
        self.dropout3 = nn.Dropout(ffn_dropout)
        self.norm3 = nn.LayerNorm(feat_dim)

    def forward(self, slots, slot_embeds, features, feature_masks, pos_encodings):
        """
        Forward method of the GlobalDecoderLayer module.

        Args:
            slots (FloatTensor): Object slots of shape [num_slots, batch_size, feat_dim].
            slot_embeds (FloatTensor): Slot embeddings of shape [num_slots, batch_size, feat_dim].
            features (FloatTensor): Features of shape [H*W, batch_size, feat_dim].
            feature_masks (BoolTensor): Boolean masks encoding inactive features of shape [batch_size, H, W].
            pos_encodings (FloatTensor): Position encodings of shape [H*W, batch_size, feat_dim].

        Returns:
            slots (FloatTensor): Updated object slots of shape [num_slots, batch_size, feat_dim].
        """

        # Global multi-head self-attention with position encoding
        queries = slots + slot_embeds
        keys = slots + slot_embeds
        values = slots

        delta_slots = self.self_attn(queries, keys, values)[0]
        slots = slots + self.dropout1(delta_slots)
        slots = self.norm1(slots)

        # Global multi-head cross-attention with position encoding
        queries = slots + slot_embeds
        keys = features + pos_encodings
        values = features

        delta_slots = self.multihead_attn(queries, keys, values, key_padding_mask=feature_masks)[0]
        slots = slots + self.dropout2(delta_slots)
        slots = self.norm2(slots)

        # Feedforward network (FFN)
        delta_slots = self.linear2(self.dropout(F.relu(self.linear1(slots))))
        slots = slots + self.dropout3(delta_slots)
        slots = self.norm3(slots)

        return slots


class SampleDecoder(nn.Module):
    """
    Class implementing the SampleDecoder module.

    Attributes:
        feat_dim (int): Feature dimension used in the decoder.
        num_init_slots (int): Number of initial slots per image.
        num_layers (int): Number of concatenated decoder layers.
        return_all (bool): Whether to return all decoder predictions.
        trained (bool): Whether the sample decoder is trained or not.
        layers (nn.ModuleList): List of sample decoder layers being concatenated.
    """

    def __init__(self, decoder_layer, decoder_dict, feat_dim, num_init_slots):
        """
        Initializes the SampleDecoder module.

        Args:
            decoder_layer (nn.Module): Sample decoder layer module to be concatenated.
            decoder_dict (Dict): Dictionary containing the decoder parameters:
                - num_layers (int): number of concatenated decoder layers;
                - return_all (bool): whether to return all decoder predictions;
                - train (bool): whether decoder should be trained or not;
                - no_curio_sharing: whether curiosity kernels should be shared between layers or not.
            feat_dim (int): Feature dimension used in the decoder.
            num_init_slots (int): Number of initial slots per image.

        Raises:
            ValueError: Raised when both the number of layers and layer iterations equal one in training mode,
                        as the segmentation and curiosity modules cannot be learned in this case.
        """

        super().__init__()
        self.feat_dim = feat_dim
        self.num_init_slots = num_init_slots
        self.num_layers = decoder_dict['num_layers']
        self.return_all = decoder_dict['return_all']
        self.trained = decoder_dict['train']

        self.layers = self.build_layers(decoder_layer, decoder_dict['no_curio_sharing'])
        self.requires_grad_(self.trained)

        if decoder_layer.num_iterations == 1 and decoder_dict['no_curio_sharing']:
            self.layers[-1].cross_attention.curio_kernel.requires_grad_(False)

        if self.trained and self.num_layers == 1 and decoder_layer.num_iterations == 1:
            raise ValueError("The number of layers and layer iterations cannot both equal one in training mode.")

    def build_layers(self, decoder_layer, no_curio_sharing):
        """
        Build layers of sample decoder module.

        Some modules are shared between layers (segmentation modules), while some are not (MHA modules).
        Whether or not curiosity kernels are shared, depends on the 'no_curio_sharing' input argument.

        Args:
            decoder_layer (nn.Module): Sample decoder layer module to be concatenated.
            no_curio_sharing (bool): Whether curiosity kernels should be shared between layers or not.

        Returns:
            layers (nn.ModuleList): List of sample decoder layers being concatenated.
        """

        # Initialize module list
        layers = nn.ModuleList()

        # Construct layer per layer
        for layer_id in range(self.num_layers):
            layer = copy.deepcopy(decoder_layer)

            # Share segmentation modules between layers
            layer.cross_attention.slot_proj = decoder_layer.cross_attention.slot_proj
            layer.cross_attention.feat_proj = decoder_layer.cross_attention.feat_proj

            # Share curiosity modules between layers if desired
            if not no_curio_sharing:
                layer.cross_attention.curio_kernel = decoder_layer.cross_attention.curio_kernel

            layers.append(layer)

        return layers

    def forward_init(self, feature_masks, pos_encodings, mask_fill=-1e6):
        """
        Initializes slots, segmentation/curiosity maps and other useful stuff.

        Args:
            feature_masks (BoolTensor): Boolean masks encoding inactive features of shape [batch_size, H, W].
            pos_encodings (FloatTensor): Position encodings of shape [H*W, batch_size, feat_dim].

        Returns:
            slots (FloatTensor): Initial slots of shape [1, num_slots_total, feat_dim].
            batch_idx (IntTensor): Batch indices corresponding to the slots of shape [num_slots_total].
            seg_maps (FloatTensor): Initial segmentation maps of shape [num_slots_total, H, W].
            curio_maps (FloatTensor): Initial curiosity maps of shape [num_slots_total, H, W].
            max_mask_entries (int): Maximum masked entries of mask from feature_masks.
        """

        batch_size, H, W = feature_masks.shape
        num_slots_total = self.num_init_slots * batch_size
        device = pos_encodings.device

        # Uniform sampling of initial slots within non-padded regions
        batch_idx = torch.arange(num_slots_total, device=device) // self.num_init_slots
        modified_masks = ~feature_masks.view(batch_size, H*W) * torch.randint(9, size=(batch_size, H*W), device=device)
        _, sorted_idx = torch.sort(modified_masks, dim=1, descending=True)
        flat_pos_idx = sorted_idx[:, :self.num_init_slots].flatten()
        slots = pos_encodings[flat_pos_idx, batch_idx, :].unsqueeze(0)

        # Initialize segmentation maps
        seg_maps = torch.full((num_slots_total, H, W), 0.5, device=device, dtype=torch.float32)

        # Initialize curiosity maps
        height_vector = torch.arange(-H+1, H, device=device, dtype=torch.int64)
        width_vector = torch.arange(-W+1, W, device=device, dtype=torch.int64)
        xy_grid = torch.stack(torch.meshgrid(height_vector, width_vector), dim=0)
        gauss_kernel = torch.exp(-torch.norm(xy_grid.to(dtype=torch.float32), p=2, dim=0)/4.0)

        curio_maps = torch.zeros(num_slots_total, 3*H, 3*W, device=device)
        slot_idx = torch.arange(num_slots_total)[:, None, None]
        pos_idx = torch.stack([flat_pos_idx // W, flat_pos_idx % W], dim=0)[:, :, None, None]

        curio_maps[slot_idx, pos_idx[0, :]+xy_grid[0]+H, pos_idx[1, :]+xy_grid[1]+W] = gauss_kernel
        curio_maps = curio_maps[:, H:2*H, W:2*W].contiguous()
        curio_maps.masked_fill_(feature_masks[batch_idx], mask_fill)

        # Compute maximum masked entries of mask from feature_masks
        max_mask_entries = torch.max(torch.sum(feature_masks.view(batch_size, H*W), dim=1)).item()

        return slots, batch_idx, seg_maps, curio_maps, max_mask_entries

    def forward(self, features, feature_masks, pos):
        """
        Forward method of the SampleDecoder module.

        Args:
            features (FloatTensor): Features of shape [H*W, batch_size, feat_dim].
            feature_masks (BoolTensor): Boolean masks encoding inactive features of shape [batch_size, H, W].
            pos (FloatTensor): Position encodings of shape [H*W, batch_size, feat_dim].

        Returns:
            slots (FloatTensor): Object slots of shape [num_pred_sets, num_slots_total, feat_dim].
            batch_idx (IntTensor): Slot batch indices (in ascending order) of shape [num_pred_sets, num_slots_total].
            seg_maps (FloatTensor): Segmentation maps at last decoder layer of shape [num_slots_total, H, W].
        """

        # Initialize slots, segmentation/curiosity maps and other useful stuff
        slots, batch_idx, seg_maps, curio_maps, *args = self.forward_init(feature_masks, pos)
        slots_list = []

        # Compute slots and segmentation maps
        for layer_id, layer in enumerate(self.layers):
            slots, seg_maps, curio_maps = layer(features, pos, slots, batch_idx, seg_maps, curio_maps, *args)
            slots_list.append(slots) if self.return_all else None

        # Add final slots to slots list if not already done
        if not self.return_all:
            slots_list.append(slots)

        # Some post-processsing
        num_pred_sets = len(slots_list)
        slots_list.reverse()
        slots = torch.stack(slots_list, dim=0)
        slots = slots.transpose(1, 2).reshape(num_pred_sets, -1, self.feat_dim)
        batch_idx = batch_idx[None, :].expand(num_pred_sets, -1)

        return slots, batch_idx, seg_maps


class SampleDecoderLayer(nn.Module):
    """
    Class implementing the SampleDecoderLayer module.

    Attributes:
        cross_attention (nn.Module): The cross-attention module.
        self_attention (nn.Module): The self-attention module.
        ffn (nn.Module): The FFN (feedforward network) module.
        num_iterations (int): Number of iterations before returning.
        iteration_type (str): String containing one of following iteration types:
            - outside: iterate over whole, i.e. over cross-attention, self-attention and ffn together;
            - inside: iterate over cross-attention only and end with single self-attention and ffn.
    """

    def __init__(self, cross_attention, self_attention, ffn, iter_dict):
        """
        Initializes the SampleDecoderLayer module.

        Args:
            cross_attention (nn.Module): The cross-attention module.
            self_attention (nn.Module): The self-attention module.
            ffn (nn.Module): The FFN (feedforward network) module.
            iter_dict (Dict): Dictionary containing following iteration parameters:
                - num_iterations (int): number of iterations before returning;
                - iteration_type (str): string containing one of following iteration types:
                    - outside: iterate over whole, i.e. over cross-attention, self-attention and ffn together;
                    - inside: iterate over cross-attention only and end with single self-attention and ffn.
        """

        super().__init__()
        self.cross_attention = cross_attention
        self.self_attention = self_attention
        self.ffn = ffn

        self.num_iterations = iter_dict['num_iterations']
        self.iteration_type = iter_dict['type']

    def forward(self, features, pos, slots, batch_idx, seg_maps, curio_maps, *args):
        """
        Forward method of the SampleDecoderLayer module.

        Args:
            features (FloatTensor): Features of shape [H*W, batch_size, feat_dim].
            pos (FloatTensor): Position encodings of shape [H*W, batch_size, feat_dim].
            slots (FloatTensor): Object slots of shape [1, num_slots_total, feat_dim].
            batch_idx (IntTensor): Batch indices corresponding to the slots of shape [num_slots_total].
            seg_maps (FloatTensor): Segmentation maps of shape [num_slots_total, H, W].
            curio_maps (FloatTensor): Curiosity maps of shape [num_slots_total, H, W].
            args (List): List containing additional input arguments:
                - max_mask_entries (int): Maximum number of masked entries in mask from feature_masks.

        Returns:
            slots (FloatTensor): Updated object slots of shape [1, num_slots_total, feat_dim].
            seg_maps (FloatTensor): Updated segmentation maps of shape [num_slots_total, H, W].
            curio_maps (FloatTensor): Updated curiosity maps of shape [num_slots_total, H, W].
        """

        if self.iteration_type == 'outside':
            for _ in range(self.num_iterations):
                outputs = self.cross_attention(features, pos, slots, batch_idx, seg_maps, curio_maps, *args)
                slots, seg_maps, curio_maps = outputs

                slots, seg_maps, curio_maps = self.self_attention(slots, batch_idx, seg_maps, curio_maps)
                slots = self.ffn(slots)

        elif self.iteration_type == 'inside':
            for _ in range(self.num_iterations):
                outputs = self.cross_attention(features, pos, slots, batch_idx, seg_maps, curio_maps, *args)
                slots, seg_maps, curio_maps = outputs

            slots, seg_maps, curio_maps = self.self_attention(slots, batch_idx, seg_maps, curio_maps)
            slots = self.ffn(slots)

        return slots, seg_maps, curio_maps


class SampleCrossAttention(nn.Module):
    """
    Class implementing the SampleCrossAttention module.

    Attributes:
        num_pos_samples (int): Number of positive features (i.e. high curiosity features) sampled per slot.
        num_neg_samples (int): Number of negative features (i.e. low curiosity features) sampled per slot.
        sample_type (str): String indicating whether to sample before or after input projection.

        feat_dim (int): Feature dimension and slot dimension used throughout the module.
        num_heads (int): Number of attention heads used during multi-head attention.
        head_dim (int): Dimension of projected features and slots for each head.
        dropout (float): Dropout probability used during multi-head attention.

        in_proj_weight (Parameter): Module parameter with the input projection weight of shape [3*feat_dim, feat_dim].
        in_proj_bias (Parameter): Module parameter with the input projection bias of shape [3*feat_dim].
        out_proj (nn.Linear): Module performing the output projection during multi-head attention.

        mha_delta_dropout (nn.Dropout): Dropout module used on multi-head attention output.
        mha_layer_norm (nn.LayerNorm): Layernorm module used after multi-head attention skip connection.

        curio_kernel (nn.ConvTranspose2d): Transposed convolution module spreading curiosity information.
        curio_dropout (nn.Dropout): Dropout module used during curiosity map update.
    """

    def __init__(self, feat_dim, samples_dict, mha_dict, curio_dict):
        """
        Initializes the SampleCrossAttention module.

        Args:
            feat_dim (int): Feature dimension and slot dimension used throughout the module.
            samples_dict (Dict): Dictionary containing sampling parameters:
                - num_pos_samples (int): number of positive features (i.e. high curiosity features) sampled per slot;
                - num_neg_samples (int): number of negative features (i.e. low curiosity features) sampled per slot;
                - sample_type (str): string indicating whether to sample before or after input projection.
            mha_dict (Dict): Dictionary containing parameters for the weighted multi-head attention:
                - hard_weights (bool): if true, transform soft weights into hard weights (otherwise leave unchanged);
                - num_heads (int): number of attention heads;
                - dropout (float): dropout probality used during the slots update.
            curio_dict (Dict): Dictionary containing parameters for the curiosity updates:
                - kernel_size (int): size of curiosity kernel of transposed convolution;
                - dropout (float): dropout probability used during the curiosity map update.

        Raises:
            ValueError: Raised when feature dimension is not divisible by the number of requested attention heads.
        """

        # Intialization of default nn.Module
        super().__init__()

        # Attributes defining the feature sampling procedure
        self.num_pos_samples = samples_dict['num_pos_samples']
        self.num_neg_samples = samples_dict['num_neg_samples']
        self.sample_type = samples_dict['sample_type']

        # Attributes defining the attention procedure
        self.feat_dim = feat_dim
        self.num_heads = mha_dict['num_heads']
        self.head_dim = feat_dim // self.num_heads
        self.dropout = mha_dict['dropout']

        # Check divisibility of feature dimension by head dimension
        if self.head_dim * self.num_heads == feat_dim:
            raise ValueError(f"Feature dimension {feat_dim} must be divisible by number of heads {self.num_heads}.")

        # Initializing the attention input and output projections
        self.in_proj_weight = Parameter(torch.empty(3 * feat_dim, feat_dim))
        self.in_proj_bias = Parameter(torch.empty(3 * feat_dim))
        self.out_proj = nn.Linear(feat_dim, feat_dim)

        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

        # Initializing the slot update modules
        self.mha_dropout = nn.Dropout(mha_dict['dropout'])
        self.mha_layer_norm = nn.LayerNorm(feat_dim)

        # Initializing the curiosity update modules
        self.curio_kernel = nn.ConvTranspose2d(2, 1, curio_dict['kernel_size'], padding=1)
        self.curio_dropout = nn.Dropout(curio_dict['dropout'])

    def sample(self, features, curio_maps, max_mask_entries):
        """
        Sampling features based on the curiosity maps.

        Args:
            features (FloatTensor): Features of shape [H*W, batch_size, feat_dim].
            curio_maps (FloatTensor): Curiosity maps of shape [num_slots_total, H, W].
            max_mask_entries (int): Maximum number of masked entries in map from curio_maps.

        Returns:
            feat_idx (IntTensor): Indices of positions of sampled features of shape [num_samples, num_slots_total].
                                  Here num_samples is the sum of the number of positive and negative samples per slot,
                                  with the positive samples in the front positions of the resulting tensor.
        """

        # Some renaming
        pos_samples = self.num_pos_samples
        neg_samples = self.num_neg_samples

        # Get positive sample positions
        num_slots_total, H, W = curio_maps.shape
        _, sorted_idx = torch.sort(curio_maps.view(num_slots_total, H*W), dim=1, descending=True)
        pos_idx = sorted_idx[:, :pos_samples]

        # Get negative sample positions
        neg_idx = sorted_idx[:, pos_samples:-max_mask_entries] if max_mask_entries > 0 else sorted_idx[:, pos_samples:]
        neg_idx = neg_idx[:, random.sample(range(neg_idx.shape[1]), k=neg_samples)]

        # Concatenate positive and negative sample positions
        feat_idx = torch.cat([pos_idx, neg_idx], dim=1).t()

        return feat_idx

    def update_slots(self, slots, features, pos_encodings, feat_idx, batch_idx):
        """
        Update object slots.

        Args:
            slots (FloatTensor): Object slots of shape [1, num_slots_total, feat_dim].
            features (FloatTensor): Features of shape [H*W, batch_size, feat_dim].
            pos_encodings (FloatTensor): Position encodings of shape [H*W, batch_size, feat_dim].
            feat_idx (IntTensor): Indices of positions of sampled features of shape [num_samples, num_slots_total].
            batch_idx (IntTensor): Batch indices corresponding to the slots of shape [num_slots_total].

        Returns:
            slots (FloatTensor): Updated object slots of shape [1, num_slots_total, feat_dim].
            affinities (FloatTensor): Slot-feature affinities of shape [num_samples, num_slots_total, num_heads].
        """

        # Sample features if desired
        if self.sample_type == 'before':
            features = features[feat_idx, batch_idx, :]
            pos_encodings = pos_encodings[feat_idx, batch_idx, :]

        # Get queries, keys and values
        queries = slots
        keys = features + pos_encodings
        values = features[:self.num_pos_samples] if self.sample_type == 'before' else features

        # Project queries, keys and values
        weight = self.in_proj_weight[:self.feat_dim, :]
        bias = self.in_proj_bias[:self.feat_dim]
        proj_queries = F.linear(queries, weight, bias)

        weight = self.in_proj_weight[self.feat_dim:2*self.feat_dim, :]
        bias = self.in_proj_bias[self.feat_dim:2*self.feat_dim]
        proj_keys = F.linear(keys, weight, bias)

        weight = self.in_proj_weight[2*self.feat_dim:, :]
        bias = self.in_proj_bias[2*self.feat_dim:]
        proj_values = F.linear(values, weight, bias)

        # Apply scaling
        scaling = float(self.head_dim)**-0.5
        proj_queries = scaling*proj_queries

        # Sample keys and values if desired
        if self.sample_type == 'after':
            proj_keys = proj_keys[feat_idx, batch_idx, :]
            proj_values = proj_values[feat_idx[:self.num_pos_samples], batch_idx, :]

        # Reshape and expose different heads
        num_slots_total = slots.shape[1]
        proj_queries = proj_queries.contiguous().view(1, num_slots_total*self.num_heads, self.head_dim).transpose(0, 1)
        proj_keys = proj_keys.contiguous().view(-1, num_slots_total*self.num_heads, self.head_dim).transpose(0, 1)
        proj_values = proj_values.contiguous().view(-1, num_slots_total*self.num_heads, self.head_dim).transpose(0, 1)

        # Compute slot-feature affinities
        affinities = torch.bmm(proj_queries, proj_keys.transpose(1, 2))

        # Compute attention weights (positive samples only)
        pos_affinities = affinities[:, :, :self.num_pos_samples].contiguous()
        attn_weights = F.softmax(pos_affinities, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Compute attention output
        attn_output = torch.bmm(attn_weights, proj_values)
        attn_output = attn_output.transpose(0, 1).contiguous().view(1, num_slots_total, self.feat_dim)
        attn_output = self.out_proj(attn_output)

        # Use attention output to update slots
        slots = slots + self.delta_dropout(attn_output)
        slots = self.layer_norm(slots)

        # Post-process affinities
        affinities = affinities.view(num_slots_total, self.num_heads, -1)
        affinities = affinities.permute(2, 0, 1)

        return slots, affinities

    @torch.no_grad()
    def update_seg_maps(self, seg_maps, feat_idx, affinities, pos_x=0.9, neg_x=0.1):
        """
        Update segmentation maps.

        Args:
            seg_maps (FloatTensor): Segmentation maps of shape [num_slots_total, H, W].
            feat_idx (IntTensor): Indices of positions of sampled features of shape [num_samples, num_slots_total].
            affinities (FloatTensor): Slot-feature affinities of shape [num_samples, num_slots_total, num_heads].
            pos_x (float): Segmentation probability given to position with affinity equal to mean positive affinity.
            neg_x (float): Segmentation probability given to position with affinity equal to mean negative affinity.

        Returns:
            seg_maps (FloatTensor): Updated segmentation maps of shape [num_slots_total, H, W].
        """

        # Get positive and negative affinities (averaged over heads)
        affinities = torch.mean(affinities, dim=-1)
        pos_affinities = affinities[:self.num_pos_samples]
        neg_affinities = affinities[self.num_pos_samples:]

        # Compute mean positive and negative affinities
        mean_pos_affinity = torch.mean(pos_affinities, dim=0)
        mean_neg_affinity = torch.mean(neg_affinities, dim=0)

        # Rescale affinities
        slope = (mean_pos_affinity-mean_neg_affinity) / (pos_x-neg_x)
        bias = mean_pos_affinity - slope*pos_x
        affinities = (affinities-bias) / slope

        # Compute segmentation probabilities
        seg_probs = F.sigmoid(affinities)

        # Update segmentation maps at sampled positions
        num_slots_total, H, W = seg_maps.shape
        seg_maps = seg_maps.view(num_slots_total, H*W)
        seg_maps[torch.arange(num_slots_total), feat_idx] = seg_probs
        seg_maps = seg_maps.view(num_slots_total, H, W)

        return seg_maps

    def update_curio_maps(self, curio_maps, feat_idx, seg_maps):
        """
        Update curiosity maps.

        Args:
            curio_maps (FloatTensor): Curiosity maps of shape [num_slots_total, H, W].
            feat_idx (IntTensor): Indices of sampled positions of shape [num_samples, num_slots_total].
            seg_maps (FloatTensor): Segmentation maps of shape [num_slots_total, H, W].

        Returns:
            curio_maps (FloatTensor): Updated curiosity maps of shape [num_slots_total, H, W].
        """

        # Update curiosity maps only if curiosity kernel is learned
        if not self.curio_kernel.weight.requires_grad:
            return curio_maps

        # Compute curiosity map changes
        seg_maps_with_inverse = torch.cat([seg_maps.unsqueeze(1), 1-seg_maps.unsqueeze(1)], dim=1)
        delta_curio = self.curio_kernel(seg_maps_with_inverse).squeeze(1)
        delta_curio = self.curio_dropout(delta_curio)

        # Add changes and apply default layer normalization
        curio_maps = curio_maps + delta_curio
        curio_maps = F.layer_norm(curio_maps, curio_maps.shape[1:])

        return curio_maps

    def forward(self, features, pos, slots, batch_idx, seg_maps, curio_maps, max_mask_entries):
        """
        Forward method of SampleCrossAttention module.

        Args:
            features (FloatTensor): Features of shape [H*W, batch_size, feat_dim].
            pos (FloatTensor): Position encodings of shape [H*W, batch_size, feat_dim].
            slots (FloatTensor): Object slots of shape [1, num_slots_total, feat_dim].
            batch_idx (IntTensor): Batch indices corresponding to the slots of shape [num_slots_total].
            seg_maps (FloatTensor): Segmentation maps of shape [num_slots_total, H, W].
            curio_maps (FloatTensor): Curiosity maps of shape [num_slots_total, H, W].
            max_mask_entries (int): Maximum masked entries of mask from feature_masks.

        Returns:
            slots (FloatTensor): Updated object slots of shape [1, num_slots_total, feat_dim].
            seg_maps (FloatTensor): Updated segmentation maps of shape [num_slots_total, H, W].
            curio_maps (FloatTensor): Updated curiosity maps of shape [num_slots_total, H, W].
        """

        feat_idx = self.sample(features, pos, batch_idx, curio_maps, max_mask_entries)
        slots, affinities = self.update_slots(slots, features, pos, feat_idx, batch_idx)
        seg_maps = self.update_seg_maps(seg_maps, slots, affinities)
        curio_maps = self.update_curio_maps(curio_maps, feat_idx, seg_maps)

        return slots, seg_maps, curio_maps


class SampleSelfAttention(nn.Module):
    """
    Class implementing the SampleSelfAttention module.

    Attributes:
        mha (nn.MultiheadAttention): Multi-head attention (MHA) module used for self-attention.
        delta_dropout (nn.Dropout): Dropout module used after self-attention.
        layer_norm (nn.LayerNorm): Layernorm module used after skip connection.
    """

    def __init__(self, feat_dim, num_heads, dropout):
        """
        Initializes the SelfAttention module.

        Args:
            feat_dim (int): Feature dimension.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability used throughout the module.
        """

        super().__init__()
        self.mha = nn.MultiheadAttention(feat_dim, num_heads, dropout=dropout)
        self.delta_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feat_dim)

    def update_slots(self, slots, batch_idx):
        """
        Update slots through self-attention.

        Args:
            slots (FloatTensor): Object slots of shape [1, num_slots_total, feat_dim].
            batch_idx (IntTensor): Batch indices corresponding to the slots of shape [num_slots_total].

        Returns:
            slots (FloatTensor): Updated object slots of shape [1, num_slots_total, feat_dim].
        """

        queries = keys = values = slots.transpose(0, 1)
        attn_mask = (batch_idx[:, None]-batch_idx[None, :]) != 0
        delta_slots = self.mha(queries, keys, values, need_weights=False, attn_mask=attn_mask)[0]

        slots = slots + self.delta_dropout(delta_slots.transpose(0, 1))
        slots = self.layer_norm(slots)

        return slots

    def update_seg_maps(self, seg_maps, slots, batch_idx):
        """
        Update segmentation maps.

        Args:
            seg_maps (FloatTensor): Segmentation maps of shape [num_slots_total, H, W].
            slots (FloatTensor): Object slots of shape [1, num_slots_total, feat_dim].

        Returns:
            seg_maps (FloatTensor): Updated segmentation maps of shape [num_slots_total, H, W].
        """

        return seg_maps

    def forward(self, slots, batch_idx, seg_maps, curio_maps):
        """
        Forward method of SampleSelfAttention module.

        Args:
            slots (FloatTensor): Object slots of shape [1, num_slots_total, feat_dim].
            batch_idx (IntTensor): Batch indices corresponding to the slots of shape [num_slots_total].
            seg_maps (FloatTensor): Segmentation maps of shape [num_slots_total, H, W].
            curio_maps (FloatTensor): Curiosity maps of shape [num_slots_total, H, W].

        Returns:
            slots (FloatTensor): Updated object slots of shape [1, num_slots_total, feat_dim].
            seg_maps (FloatTensor): Updated segmentation maps of shape [num_slots_total, H, W].
            curio_maps (FloatTensor): Updated curiosity maps of shape [num_slots_total, H, W].
        """

        slots = self.update_slots(slots, batch_idx)
        # seg_maps = self.update_seg_maps(seg_maps, slots)
        # curio_maps = self.update_curio_maps()

        return slots, seg_maps, curio_maps


class FFN(nn.Module):
    """
    Class implementing the FFN module.

    Attributes:
        hidden_linear (nn.Linear): Linear layer going from feature space to hidden space.
        delta_linear (nn.Linear): Linear layer going from hidden space to delta features.
        hidden_dropout (nn.Dropout): Dropout layer used after the hidden layer.
        delta_dropout (nn.Dropout): Dropout layer used on the delta features.
        layer_norm (nn.LayerNorm): Layernorm module used after skip connection.
    """

    def __init__(self, feat_dim, hidden_dim, dropout):
        """
        Initializes the FFN module.

        Args:
            feat_dim (int): Feature dimension.
            hidden_dim (int): Hidden space dimension.
            dropout (float): Dropout probability used throughout the module.
        """

        super().__init__()
        self.hidden_linear = nn.Linear(feat_dim, hidden_dim)
        self.delta_linear = nn.Linear(hidden_dim, feat_dim)
        self.hidden_dropout = nn.Dropout(dropout)
        self.delta_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feat_dim)

    def forward(self, slots):
        """
        Forward method of FFN module.

        Args:
            slots (FloatTensor): Object slots of shape [1, num_slots_total, feat_dim].

        Returns:
            slots (FloatTensor): Updated object slots of shape [1, num_slots_total, feat_dim].
        """

        hidden_slots = F.relu(self.hidden_linear(slots))
        delta_slots = self.delta_linear(self.hidden_dropout(hidden_slots))

        slots = slots + self.delta_dropout(delta_slots)
        slots = self.layer_norm(slots)

        return slots


def build_decoder(args):
    """
    Build SampleDecoder module from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        decoder (SampleDecoder): The specified SampleDecoder module.
    """

    decoder_dict = {'num_layers': args.num_decoder_layers, 'return_all': args.aux_loss, 'train': args.lr_decoder > 0}
    mha_dict = {'num_heads': args.num_heads, 'dropout': args.mha_dropout}

    if args.decoder_type == 'sample':
        sample_dict = {'num_pos_samples': args.num_pos_samples, 'num_neg_samples': args.num_neg_samples}
        sample_dict['sample_type'] = args.sample_type
        curio_dict = {'kernel_size': args.curio_kernel_size, 'dropout': args.curio_dropout}

        cross_attention = SampleCrossAttention(args.feat_dim, sample_dict, mha_dict, curio_dict)
        self_attention = SampleSelfAttention(args.feat_dim, args.num_heads, args.mha_dropout)
        ffn = FFN(args.feat_dim, args.ffn_hidden_dim, args.ffn_dropout)

        iter_dict = {'num_iterations': args.num_decoder_iterations, 'type': args.iter_type}
        decoder_layer = SampleDecoderLayer(cross_attention, self_attention, ffn, iter_dict)

        decoder_dict['no_curio_sharing'] = args.no_curio_sharing
        decoder = SampleDecoder(decoder_layer, decoder_dict, args.feat_dim, args.num_init_slots)

    elif args.decoder_type == 'global':
        ffn_dict = {'hidden_dim': args.ffn_hidden_dim, 'dropout': args.ffn_dropout}
        decoder_layer = GlobalDecoderLayer(args.feat_dim, mha_dict, ffn_dict)
        decoder = GlobalDecoder(decoder_layer, decoder_dict, args.feat_dim, args.num_slots)

    return decoder
