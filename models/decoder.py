"""
Decoder modules and build function.
"""
from collections import OrderedDict
import copy
import random

import scipy.stats
import torch
from torch import nn
from torch.autograd.function import Function
import torch.nn.functional as F

from .utils import MLP


class GlobalDecoder(nn.Module):
    """
    Class implementing the GlobalDecoder module.

    Attributes:
        feat_dim (int): Feature dimension used in the decoder.
        layers (nn.ModulesList): List of decoder layers being concatenated.
        num_layers (int): Number of concatenated decoder layers.
    """

    def __init__(self, decoder_layer, feat_dim, num_slots, num_layers, train_decoder):
        """
        Initializes the GlobalDecoder module.

        Args:
            decoder_layer (nn.Module): Decoder layer module to be concatenated.
            feat_dim (int): Feature dimension used in the decoder.
            num_slots (int): Number of slots per image.
            num_layers (int): Number of concatenated decoder layers.
            train_decoder (bool): Whether decoder should be trained or not.
        """

        super().__init__()
        self.feat_dim = feat_dim
        self.num_slots = num_slots
        self.num_layers = num_layers

        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.requires_grad_(train_decoder)
        self.slot_embeds = nn.Embedding(num_slots, feat_dim)

    def load_from_original_detr(self, state_dict):
        """
        Load decoder from one of Facebook's original DETR model.

        state_dict (Dict): Dictionary containing model parameters and persistent buffers.
        """

        decoder_identifier = 'transformer.decoder.'
        identifier_length = len(decoder_identifier)
        decoder_state_dict = OrderedDict()

        for original_name, state in state_dict.items():
            if decoder_identifier in original_name:
                new_name = original_name[identifier_length:]
                decoder_state_dict[new_name] = state

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
            slots (FloatTensor): Object slots of shape [1, batch_size*num_slots, feat_dim].
            batch_idx (IntTensor): Batch indices corresponding to the slots of shape [batch_size*num_slots].
        """

        batch_size = feature_masks.shape[0]
        batch_idx = torch.arange(batch_size).repeat(self.num_slots)

        feature_masks = feature_masks.flatten(1)
        slot_embeds = self.slot_embeds.weight.unsqueeze(1).repeat(1, batch_size, 1)
        slots = torch.zeros_like(slot_embeds)

        for layer in self.layers:
            slots = layer(slots, slot_embeds, features, feature_masks, pos_encodings)

        slots = slots.view(-1, self.feat_dim)
        slots = slots.unsqueeze(0)

        return slots, batch_idx, None


class GlobalDecoderLayer(nn.Module):

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
        decoder_iterations (int): Number of decoder iterations per decoder layer.
        feat_dim (int): Feature dimension used in the decoder.
        layers (nn.ModuleList): List of decoder layers being concatenated.
        num_init_slots (int): Number of initial slots per image.
        num_layers (int): Number of concatenated decoder layers.
    """

    def __init__(self, decoder_layer, decoder_dict, feat_dim, num_init_slots):
        """
        Initializes the SampleDecoder module.

        Args:
            decoder_layer (nn.Module): Decoder layer module to be concatenated.
            decoder_dict (Dict): Dictionary containing the decoder parameters:
                - iterations (int): number of decoder iterations per decoder layer;
                - num_layers (int): number of concatenated decoder layers;
                - train (bool): whether decoder should be trained or not.
            feat_dim (int): Feature dimension used in the decoder.
            num_init_slots (int): Number of initial slots per image.
        """

        super().__init__()
        self.feat_dim = feat_dim
        self.iterations = decoder_dict['iterations']
        self.num_init_slots = num_init_slots
        self.num_layers = decoder_dict['num_layers']

        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(self.num_layers)])
        self.requires_grad_(decoder_dict['train'])

    def forward_init(self, feature_masks, pos_encodings, mask_fill=-1e6):
        """
        Initializes slots, segmentation/curiosity maps and other useful stuff.

        Args:
            feature_masks (BoolTensor): Boolean masks encoding inactive features of shape [batch_size, H, W].
            pos_encodings (FloatTensor): Position encodings of shape [H*W, batch_size, feat_dim].

        Returns:
            slots (FloatTensor): Initial slots of shape [1, num_slots_total, feat_dim].
            batch_idx (IntTensor): Batch indices corresponding to the slots of shape [num_slots_total].
            seg_maps (FloatTensor): Initial segmentation maps of shape [num_slots_total, 3, H*W].
            curio_maps (FloatTensor): Initial curiosity maps of shape [num_slots_total, H, W].

            def_xy_grid (IntTensor): Tensor containing the (H,W)-grid coordinates of shape [H, W, 2].
            gauss_grid (FloatTensor): Tensor containing the Gaussian grid of shape [2*H-1, 2*W-1].
            max_mask_entries (int): Maximum masked entries of mask from feature_masks.
        """

        batch_size, H, W = feature_masks.shape
        num_slots_total = self.num_init_slots * batch_size
        device = pos_encodings.device

        # Uniform sampling of initial slots within non-padded regions
        batch_idx = torch.repeat_interleave(torch.arange(batch_size), self.num_init_slots).to(device)
        modified_masks = ~feature_masks.view(batch_size, H*W) * torch.randint(9, size=(batch_size, H*W), device=device)
        _, sorted_idx = torch.sort(modified_masks, dim=1, descending=True)
        flat_pos_idx = sorted_idx[:, :self.num_init_slots].flatten()
        slots = pos_encodings[flat_pos_idx, batch_idx, :].unsqueeze(0)
        pos_idx = torch.stack([flat_pos_idx // W, flat_pos_idx % W], dim=-1)

        # Initialize segmentation maps
        masks = feature_masks[batch_idx]
        seg_maps = torch.zeros(num_slots_total, 3, H, W, device=device)
        seg_maps[masks[:, None, :, :].expand(-1, 3, -1, -1)] = mask_fill
        seg_maps = seg_maps.view(num_slots_total, 3, H*W)

        # Initialize curiosity maps
        xy_grid = torch.stack(torch.meshgrid(torch.arange(-H+1, H), torch.arange(-W+1, W)), dim=-1)
        gauss_pdf = scipy.stats.multivariate_normal([0, 0]).pdf
        gauss_grid = torch.from_numpy(gauss_pdf(xy_grid)).to(device=device, dtype=torch.float32)

        xy_grid = xy_grid[-H:, -W:][None, :].expand(num_slots_total, -1, -1, -1).to(device)
        xy_grid = (xy_grid - pos_idx[:, None, None, :]).permute(3, 0, 1, 2)
        curio_maps = gauss_grid[xy_grid[0], xy_grid[1]]
        curio_maps[masks] = mask_fill

        # Compute some useful items for later
        def_xy_grid = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)), dim=-1).to(device)
        max_mask_entries = torch.max(torch.sum(feature_masks.view(batch_size, H*W), dim=1)).item()

        return slots, batch_idx, seg_maps, curio_maps, def_xy_grid, gauss_grid, max_mask_entries

    def forward(self, features, feature_masks, pos):
        """
        Forward method of the SampleDecoder module.

        Args:
            features (FloatTensor): Features of shape [H*W, batch_size, feat_dim].
            feature_masks (BoolTensor): Boolean masks encoding inactive features of shape [batch_size, H, W].
            pos (FloatTensor): Position encodings of shape [H*W, batch_size, feat_dim].

        Returns:
            slots (FloatTensor): Object slots of shape [1, num_slots_total, feat_dim].
            batch_idx (IntTensor): Batch indices corresponding to the slots of shape [num_slots_total].
            seg_maps (FloatTensor): Segmentation maps of shape [num_slots_total, 3, H*W].
        """

        # Initialize slots, segmentation/curiosity maps and other useful stuff
        slots, batch_idx, seg_maps, curio_maps, *args = self.forward_init(feature_masks, pos)

        # Loop over different decoder layers
        for layer in self.layers:
            for _ in range(self.iterations):
                slots, seg_maps, curio_maps = layer(features, pos, slots, batch_idx, seg_maps, curio_maps, *args)

        return slots, batch_idx, seg_maps


class SampleDecoderLayer(nn.Module):
    """
    Class implementing the SampleDecoderLayer module.

    Attributes:
        cross_attention (nn.Module): The cross-attention module.
        self_attention (nn.Module): The self-attention module.
        ffn (nn.Module): The FFN (feedforward network) module.
    """

    def __init__(self, cross_attention, self_attention, ffn):
        """
        Initializes the SampleDecoderLayer module.

        Args:
            cross_attention (nn.Module): The cross-attention module.
            self_attention (nn.Module): The self-attention module.
            ffn (nn.Module): The FFN (feedforward network) module.
        """

        super().__init__()
        self.cross_attention = cross_attention
        self.self_attention = self_attention
        self.ffn = ffn

    def forward(self, features, pos, slots, batch_idx, seg_maps, curio_maps, *args):
        """
        Forward method of the SampleDecoderLayer module.

        Args:
            features (FloatTensor): Features of shape [H*W, batch_size, feat_dim].
            pos (FloatTensor): Position encodings of shape [H*W, batch_size, feat_dim].
            slots (FloatTensor): Object slots of shape [1, num_slots_total, feat_dim].
            batch_idx (IntTensor): Batch indices corresponding to the slots of shape [num_slots_total].
            seg_maps (FloatTensor): Segmentation maps of shape [num_slots_total, 3, H*W].
            curio_maps (FloatTensor): Curiosity maps of shape [num_slots_total, H, W].

            args (Tuple): Tuple containing following fixed items:
                - def_xy_grid (IntTensor): Tensor containing the (H,W)-grid coordinates of shape [H, W, 2];
                - gauss_grid (FloatTensor): Tensor containing the Gaussian grid of shape [2*H-1, 2*W-1];
                - max_mask_entries (int): Maximum number of masked entries in mask from feature_masks.

        Returns:
            slots (FloatTensor): Object slots of shape [1, num_slots_total, feat_dim].
            seg_maps (FloatTensor): Segmentation maps of shape [num_slots_total, 3, H*W].
            curio_maps (FloatTensor): Curiosity maps of shape [num_slots_total, H, W].
        """

        slots, seg_maps, curio_maps = self.cross_attention(features, pos, slots, batch_idx, seg_maps, curio_maps, *args)
        slots = self.self_attention(slots, batch_idx)
        slots = self.ffn(slots)

        return slots, seg_maps, curio_maps


class HardWeightGate(Function):
    """
    Class implementing the HardWeightGate autograd function.

    Forward method:
        It takes weights as input and transforms them into hard weights, all consisting of ones.

    Backward method:
        It returns the gradients unchanged as if no operation was performed in the forward method.
    """

    @staticmethod
    def forward(ctx, soft_weights):
        return torch.ones_like(soft_weights)

    @staticmethod
    def backward(ctx, grad_hard_weights):
        return grad_hard_weights


class WeightedCrossAttention(nn.Module):
    """
    Class implementing the WeightedCrossAttention module.

    Attributes:
        samples_per_slot (int): Number of features sampled per slot.
        coverage_ratio (float): Ratio of samples taken randomly (other samples are taken by importance).
        hard_weights (bool): If true, transform soft weights into hard weights (otherwise leave unchanged).

        mha (nn.MultiheadAttention): Multi-head attention (MHA) module used for cross-attention.
        delta_dropout (nn.Dropout): Dropout module used on multi-head attention output.
        layer_norm (nn.LayerNorm): Layernorm module used after skip connection.

        slot_proj (MLP): Multi-layer perceptron (MLP) projecting slots before segmentation classification.
        feat_proj (MLP): Multi-layer perceptron (MLP) projecting sampled features before segmentation classification.
        seg_class (MLP): Multi-layer perceptron (MLP) classifying each sampled feature as object/edge/no-object.

        curio_weights (FloatTensor): Tensor of curiosity weights corresponding to the object/edge/no-object classes.
        curio_memory (float): Determines ratio of current curiosity maintained during update in non-sampled positions.
    """

    def __init__(self, feat_dim, samples_dict, mha_dict, curio_dict):
        """
        Initializes the WeightedCrossAttention module.

        Args:
            feat_dim (int): Feature dimension.
            samples_dict (Dict): Dictionary containing sampling parameters:
                - samples_per_slot (int): number of features sampled per slot;
                - coverage_ratio (float): ratio of samples taken randomly (other samples are taken by importance).
            mha_dict (Dict): Dictionary containing parameters for the weighted multi-head attention:
                - hard_weights (bool): if true, transform soft weights into hard weights (otherwise leave unchanged);
                - num_heads (int): number of attention heads;
                - dropout (float): dropout probality used throughout the module.
            curio_dict (Dict): Dictionary containing parameters for the curiosity updates:
                - weights (List): list of curiosity weights corresponding to the object/edge/no-object classes;
                - memory (float): ratio of current curiosity maintained during update in non-sampled positions.
        """

        # Intialization of default nn.Module
        super().__init__()

        # Parameters defining the feature sampling procedure
        self.samples_per_slot = samples_dict['samples_per_slot']
        self.coverage_ratio = samples_dict['coverage_ratio']

        # Initializing the multi-head attention module
        self.hard_weights = mha_dict['hard_weights']
        num_heads = mha_dict['num_heads']
        dropout = mha_dict['dropout']
        self.mha = nn.MultiheadAttention(feat_dim, num_heads, dropout=dropout)

        # Initializing the dropout and layernorm modules
        self.delta_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feat_dim)

        # Initializing the MLP's used for updating the segmentation maps
        self.slot_proj = MLP(feat_dim, feat_dim//2, feat_dim//2, 2)
        self.feat_proj = MLP(feat_dim, feat_dim//2, feat_dim//2, 2)
        self.seg_class = MLP(feat_dim, feat_dim, 3, 2)

        # Parameters defining how the curiosities are learned and updated
        self.register_buffer('curio_weights', torch.tensor(curio_dict['weights']))
        self.curio_memory = curio_dict['memory']

    def sample(self, features, pos_encodings, batch_idx, curio_maps, max_mask_entries):
        """
        Sampling features based on the curiosity maps.

        Args:
            features (FloatTensor): Features of shape [H*W, batch_size, feat_dim].
            pos_encodings (FloatTensor): Position encodings of shape [H*W, batch_size, feat_dim].
            batch_idx (IntTensor): Batch indices corresponding to the slots of shape [num_slots_total].
            curio_maps (FloatTensor): Curiosity maps of shape [num_slots_total, H, W].
            max_mask_entries (int): Maximum number of masked entries in map from curio_maps.

        Returns:
            sample_dict (Dict): Dictionary of sampled items:
                - feat_idx (IntTensor): indices of sampled positions of shape [samples_per_slot, num_slots_total];
                - features (FloatTensor): sampled features of shape [samples_per_slot, num_slots_total, feat_dim];
                - pos_encodings (FloatTensor): sampled pos. enc. of shape [samples_per_slot, num_slots_total, feat_dim];
                - curiosities (FloatTensor): sampled curiosities of shape [samples_per_slot, num_slots_total].
        """

        # Get number of coverage and importance samples
        cov_samples = int(self.coverage_ratio*self.samples_per_slot)
        imp_samples = self.samples_per_slot - cov_samples

        # Get sample positions based on importance sampling
        num_slots_total, H, W = curio_maps.shape
        _, sorted_idx = torch.sort(curio_maps.view(num_slots_total, H*W), dim=1, descending=True)
        imp_idx = sorted_idx[:, :imp_samples]

        # Get sample positions based on coverage
        cov_idx = sorted_idx[:, imp_samples:-max_mask_entries] if max_mask_entries > 0 else sorted_idx[:, imp_samples:]
        cov_idx = cov_idx[:, random.sample(range(cov_idx.shape[1]), k=cov_samples)]

        # Concatenate both types of indices
        feat_idx = torch.cat([imp_idx, cov_idx], dim=1).t()

        # Get sampled features, position encodings and curiosities
        sample_dict = {'feat_idx': feat_idx}
        sample_dict['features'] = features[feat_idx, batch_idx, :]
        sample_dict['pos_encodings'] = pos_encodings[feat_idx, batch_idx, :]
        sample_dict['curiosities'] = curio_maps.view(num_slots_total, -1)[torch.arange(num_slots_total), feat_idx]

        return sample_dict

    def weighted_attention(self, slots, sample_dict):
        """
        Update slots through weighted cross-attention.

        Args:
            slots (FloatTensor): Object slots of shape [1, num_slots_total, feat_dim].
            sample_dict (Dict): Dictionary of sampled items:
                - feat_idx (IntTensor): indices of sampled positions of shape [samples_per_slot, num_slots_total];
                - features (FloatTensor): sampled features of shape [samples_per_slot, num_slots_total, feat_dim];
                - pos_encodings (FloatTensor): sampled pos. enc. of shape [samples_per_slot, num_slots_total, feat_dim];
                - curiosities (FloatTensor): sampled curiosities of shape [samples_per_slot, num_slots_total].

        Returns:
            slots (FloatTensor): Updated object slots of shape [1, num_slots_total, feat_dim].
        """

        # Get queries and keys
        queries = slots
        keys = sample_dict['features'] + sample_dict['pos_encodings']

        # Get weighted values
        value_weights = torch.softmax(sample_dict['curiosities'], dim=0)
        value_weights = HardWeightGate.apply(value_weights) if self.hard_weights else value_weights
        values = value_weights[:, :, None] * sample_dict['features']

        # Perform weighted cross-attention, dropout, skip connection and layernorm
        delta_slots = self.mha(queries, keys, values, need_weights=False)[0]
        slots = slots + self.delta_dropout(delta_slots)
        slots = self.layer_norm(slots)

        return slots

    def update_seg_maps(self, slots, seg_maps, sample_dict):
        """
        Update segmentation maps.

        Args:
            slots (FloatTensor): Object slots of shape [1, num_slots_total, feat_dim].
            seg_maps (FloatTensor): Segmentation maps of shape [num_slots_total, 3, H*W].
            sample_dict (Dict): Dictionary of sampled items:
                - feat_idx (IntTensor): indices of sampled positions of shape [samples_per_slot, num_slots_total];
                - features (FloatTensor): sampled features of shape [samples_per_slot, num_slots_total, feat_dim];
                - pos_encodings (FloatTensor): sampled pos. enc. of shape [samples_per_slot, num_slots_total, feat_dim];
                - curiosities (FloatTensor): sampled curiosities of shape [samples_per_slot, num_slots_total].

        Returns:
            seg_maps (FloatTensor): Updated segmentation maps of shape [num_slots_total, 3, H*W].
            seg_probs (FloatTensor): Segmentation probabilities of shape [samples_per_slot, num_slots_total, 3].
        """

        # Project slots and features, and concatenate them
        proj_slots = self.slot_proj(slots).expand(self.samples_per_slot, -1, -1)
        proj_feats = self.feat_proj(sample_dict['features'] + sample_dict['pos_encodings'])
        proj_slotfeats = torch.cat([proj_slots, proj_feats], dim=-1)

        # Get segmentation probabilities and update segmentation maps accordingly
        seg_probs = torch.softmax(self.seg_class(proj_slotfeats), dim=-1)
        seg_maps[torch.arange(seg_maps.shape[0]), :, sample_dict['feat_idx']] = seg_probs

        return seg_maps, seg_probs

    def update_curio_maps(self, curio_maps, feat_idx, seg_probs, def_xy_grid, gauss_grid):
        """
        Update curiosity maps.

        Args:
            curio_maps (FloatTensor): Curiosity maps of shape [num_slots_total, H, W].
            feat_idx (IntTensor): Indices of sampled positions of shape [samples_per_slot, num_slots_total].
            seg_probs (FloatTensor): Segmentation probabilities of shape [samples_per_slot, num_slots_total, 3].
            def_xy_grid (IntTensor): Tensor containing the (H,W)-grid coordinates of shape [H, W, 2].
            gauss_grid (FloatTensor): Tensor containing the Gaussian grid of shape [2*H-1, 2*W-1].

        Returns:
            curio_maps (FloatTensor): Updated curiosity maps of shape [num_slots_total, H, W].
        """

        samples_per_slot = feat_idx.shape[0]
        num_slots_total, H, W = curio_maps.shape

        # Get grids centered around sampled positions
        feat_idx = torch.stack([feat_idx // W, feat_idx % W], dim=-1)
        xy_grid = def_xy_grid[:, :, None, None, :].expand(H, W, samples_per_slot, num_slots_total, 2)
        xy_grid = (xy_grid - feat_idx).permute(4, 0, 1, 2, 3)

        # Update curiosity maps with gaussians placed at sampled positions
        gauss_weights = torch.sum(self.curio_weights*seg_probs, dim=-1)
        gauss_pdfs = gauss_weights*gauss_grid[xy_grid[0], xy_grid[1]]
        curio_delta, _ = torch.max(gauss_pdfs.permute(2, 3, 0, 1), dim=0)
        curio_maps = self.curio_memory*curio_maps + (1-self.curio_memory)*curio_delta

        # Overwrite curiosities at sampled positions according to segmentation probabilities
        feat_idx = feat_idx.permute(2, 0, 1)
        sampled_curiosities = 1-torch.max(seg_probs, dim=-1)[0]
        curio_maps[torch.arange(num_slots_total), feat_idx[0], feat_idx[1]] = sampled_curiosities

        return curio_maps

    def forward(self, features, pos, slots, batch_idx, seg_maps, curio_maps, def_xy_grid, gauss_grid, max_mask_entries):
        """
        Forward method of WeightedCrossAttention module.

        Args:
            features (FloatTensor): Features of shape [H*W, batch_size, feat_dim].
            pos (FloatTensor): Position encodings of shape [H*W, batch_size, feat_dim].
            slots (FloatTensor): Object slots of shape [1, num_slots_total, feat_dim].
            batch_idx (IntTensor): Batch indices corresponding to the slots of shape [num_slots_total].
            seg_maps (FloatTensor): Segmentation maps of shape [num_slots_total, 3, H*W].
            curio_maps (FloatTensor): Curiosity maps of shape [num_slots_total, H, W].
            def_xy_grid (IntTensor): Tensor containing the (H,W)-grid coordinates of shape [H, W, 2].
            gauss_grid (FloatTensor): Tensor containing the Gaussian grid of shape [2*H-1, 2*W-1].
            max_mask_entries (int): Maximum masked entries of mask from feature_masks.

        Returns:
            slots (FloatTensor): Updated object slots of shape [1, num_slots_total, feat_dim].
            seg_maps (FloatTensor): Updated segmentation maps of shape [num_slots_total, 3, H*W].
            curio_maps (FloatTensor): Updated curiosity maps of shape [num_slots_total, H, W].
        """

        sample_dict = self.sample(features, pos, batch_idx, curio_maps, max_mask_entries)
        slots = self.weighted_attention(slots, sample_dict)
        seg_maps, seg_probs = self.update_seg_maps(slots, seg_maps, sample_dict)
        curio_maps = self.update_curio_maps(curio_maps, sample_dict['feat_idx'], seg_probs, def_xy_grid, gauss_grid)

        return slots, seg_maps, curio_maps


class SelfAttention(nn.Module):
    """
    Class implementing the SelfAttention module.

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

    def forward(self, slots, batch_idx):
        """
        Forward method of SelfAttention module.

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

    mha_dict = {'num_heads': args.num_heads, 'dropout': args.mha_dropout}
    train = args.lr_decoder > 0

    if args.decoder_type == 'sample':
        sample_dict = {'samples_per_slot': args.samples_per_slot, 'coverage_ratio': args.coverage_ratio}
        mha_dict['hard_weights'] = args.hard_weights
        curio_weights = [args.curio_weight_obj, args.curio_weight_edge, args.curio_weight_nobj]
        curio_dict = {'weights': curio_weights, 'memory': args.curio_memory}

        cross_attention = WeightedCrossAttention(args.feat_dim, sample_dict, mha_dict, curio_dict)
        self_attention = SelfAttention(args.feat_dim, args.num_heads, args.mha_dropout)
        ffn = FFN(args.feat_dim, args.ffn_hidden_dim, args.ffn_dropout)

        decoder_layer = SampleDecoderLayer(cross_attention, self_attention, ffn)
        decoder_dict = {'iterations': args.decoder_iterations, 'num_layers': args.num_decoder_layers, 'train': train}
        decoder = SampleDecoder(decoder_layer, decoder_dict, args.feat_dim, args.num_init_slots)

    elif args.decoder_type == 'global':
        ffn_dict = {'hidden_dim': args.ffn_hidden_dim, 'dropout': args.ffn_dropout}
        decoder_layer = GlobalDecoderLayer(args.feat_dim, mha_dict, ffn_dict)
        decoder = GlobalDecoder(decoder_layer, args.feat_dim, args.num_slots, args.num_decoder_layers, train)

    return decoder
