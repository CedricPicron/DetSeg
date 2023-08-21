"""
Collection of custom autograd functions.
"""

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable


class IdAttn(Function):
    """
    Class implementing the IdAttn autograd function.

    This custom autograd function avoids keeping intermediate data structures in memory.
    """

    @staticmethod
    def forward(ctx, in_act_feats, pas_feats, feat_ids, feat_weights, weight, bias):
        """
        Forward method of the IdAttn autograd function.

        Args:
            ctx (FunctionCtx): Context object storing additional data.
            in_act_feats (FloatTensor): Input active features of shape [num_act_feats, feat_size].
            pas_feats (FloatTensor): Passive features of shape [num_pas_feats, feat_size].
            feat_ids (LongTensor): Feature indices of shape [num_act_feats, num_pts[, num_heads]].
            feat_weights (FloatTensor): Feature weights of shape [num_act_feats, num_pts, num_heads].
            weight (FloatTensor): Value projection weights of shape [feat_size, feat_size].
            bias (FloatTensor): Value projection biases of shape [feat_size] or None.

        Returns:
            out_act_feats (FloatTensor): Output active features of shape [num_act_feats, feat_size].
        """

        # Save input tensors for backward pass
        ctx.save_for_backward(in_act_feats, pas_feats, feat_ids, feat_weights, weight, bias)

        # Get tensor sizes of interest
        num_act_feats, num_pts, num_heads = feat_weights.size()
        feat_size = weight.size(dim=0)

        # Get value features
        cat_feats = torch.cat([in_act_feats, pas_feats], dim=0)
        sample_feats = cat_feats[feat_ids].flatten(0, 1)

        if feat_ids.dim() == 2:
            val_feats = torch.mm(sample_feats, weight.t())

        else:
            sample_feats = sample_feats.permute(1, 0, 2)
            weight = weight.view(num_heads, feat_size // num_heads, feat_size)

            val_feats = sample_feats @ weight.transpose(1, 2)
            val_feats = val_feats.transpose(0, 1).reshape(num_act_feats * num_pts, feat_size)

        if bias is not None:
            val_feats += bias

        val_feats = val_feats.view(num_act_feats, num_pts, num_heads, feat_size // num_heads)

        # Get output active features
        out_act_feats = feat_weights.unsqueeze(dim=3) * val_feats
        out_act_feats = out_act_feats.sum(dim=1).view(num_act_feats, feat_size)

        return out_act_feats

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out_act_feats):
        """
        Backward method of the IdAttn autograd function.

        Args:
            ctx (FunctionCtx): Context object storing additional data.
            grad_out_act_feats (FloatTensor): Output active features gradient of shape [num_act_feats, feat_size].

        Returns:
            grad_in_act_feats (FloatTensor): Input active features gradient of shape [num_act_feats, feat_size].
            grad_pas_feats (FloatTensor): Passive features gradient of shape [num_pas_feats, feat_size].
            grad_feat_ids (None): None.
            grad_feat_weights (FloatTensor): Feature weights gradient of shape [num_act_feats, num_pts, num_heads].
            grad_weight (FloatTensor): Value projection weights gradient of shape [feat_size, feat_size].
            grad_bias (FloatTensor): Value projection biases gradient of shape [feat_size] or None.
        """

        # Recover input tensors from forward method
        in_act_feats, pas_feats, feat_ids, feat_weights, weight, bias = ctx.saved_tensors

        # Get tensor sizes of interest
        num_act_feats, num_pts, num_heads = feat_weights.size()
        feat_size = weight.size(dim=0)

        # Recompute value features
        cat_feats = torch.cat([in_act_feats, pas_feats], dim=0)
        sample_feats = cat_feats[feat_ids].flatten(0, 1)

        if feat_ids.dim() == 2:
            val_feats = torch.mm(sample_feats, weight.t())

        else:
            sample_feats = sample_feats.transpose(0, 1)
            weight = weight.view(num_heads, feat_size // num_heads, feat_size)

            val_feats = sample_feats @ weight.transpose(1, 2)
            val_feats = val_feats.transpose(0, 1).reshape(num_act_feats * num_pts, feat_size)

        if bias is not None:
            val_feats += bias

        val_feats = val_feats.view(num_act_feats, num_pts, num_heads, feat_size // num_heads)

        # Get gradient tensors
        grad_out_act_feats = grad_out_act_feats.view(num_act_feats, num_heads, feat_size // num_heads)
        grad_out_act_feats = grad_out_act_feats[:, None, :, :].expand(-1, num_pts, -1, -1)

        grad_feat_weights = (grad_out_act_feats * val_feats).sum(dim=3)
        grad_val_feats = grad_out_act_feats * feat_weights.unsqueeze(dim=3)

        grad_val_feats = grad_val_feats.reshape(num_act_feats * num_pts, feat_size)
        grad_bias = grad_val_feats.sum(dim=0) if bias is not None else None

        if feat_ids.dim() == 2:
            grad_sample_feats = torch.mm(grad_val_feats, weight)
            grad_weight = torch.mm(grad_val_feats.t(), sample_feats)

        else:
            grad_val_feats = grad_val_feats.view(num_act_feats * num_pts, num_heads, feat_size // num_heads)
            grad_val_feats = grad_val_feats.transpose(0, 1)

            grad_sample_feats = (grad_val_feats @ weight).transpose(0, 1).flatten(0, 1)
            grad_weight = (grad_val_feats.transpose(1, 2) @ sample_feats).flatten(0, 1)

        grad_cat_feats = torch.zeros_like(cat_feats).index_add_(0, feat_ids.flatten(), grad_sample_feats)
        grad_in_act_feats = grad_cat_feats[:num_act_feats]
        grad_pas_feats = grad_cat_feats[num_act_feats:]
        grad_feat_ids = None

        return grad_in_act_feats, grad_pas_feats, grad_feat_ids, grad_feat_weights, grad_weight, grad_bias


class IdConv2d(Function):
    """
    Class implementing the IdConv2d autograd function.

    This custom autograd function avoids keeping intermediate data structures in memory.
    """

    @staticmethod
    def forward(ctx, in_core_feats, aux_feats, conv_ids, weight, bias):
        """
        Forward method of the IdConv2d autograd function.

        Args:
            ctx (FunctionCtx): Context object storing additional data.
            in_core_feats (FloatTensor): Input core features of shape [num_core_feats, in_channels].
            aux_feats (FloatTensor): Auxiliary features of shape [num_aux_feats, in_channels].
            conv_ids (LongTensor): Indices selecting convolution features of shape [num_core_feats, kH * kW].
            weight (FloatTensor): Tensor with convolution weights of shape [out_channels, kH * kW * in_channels].
            bias (FloatTensor): Tensor with the convolution biases of shape [out_channels].

        Returns:
            out_core_feats (FloatTensor): Output core features of shape [num_core_feats, out_channels].
        """

        # Compute intermediate data structures
        pad_feat = in_core_feats.new_zeros([1, in_core_feats.size()[1]])
        cat_feats = torch.cat([in_core_feats, aux_feats, pad_feat], dim=0)
        conv_feats = cat_feats[conv_ids].flatten(1)

        # Get output core features
        out_core_feats = torch.mm(conv_feats, weight.t())

        if bias is not None:
            out_core_feats += bias

        # Save input tensors for backward pass
        ctx.save_for_backward(in_core_feats, aux_feats, conv_ids, weight, bias)

        return out_core_feats

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out_core_feats):
        """
        Backward method of the IdConv2d autograd function.

        Args:
            ctx (FunctionCtx): Context object storing additional data.
            grad_out_core_feats (FloatTensor): Output core features gradient of shape [num_core_feats, out_channels].

        Returns:
            grad_in_core_feats (FloatTensor): Input core features gradient of shape [num_core_feats, in_channels].
            grad_aux_feats (FloatTensor): Auxiliary features gradient of shape [num_aux_feats, in_channels].
            grad_conv_ids (None): None.
            grad_weight (FloatTensor): Convolution weights gradient of shape [out_channels, kH * kW * in_channels].
            grad_bias (FloatTensor): Convolution biases gradient of shape [out_channels] or None.
        """

        # Recover input tensors from forward method
        in_core_feats, aux_feats, conv_ids, weight, bias = ctx.saved_tensors

        # Recompute intermediate data structures
        pad_feat = aux_feats.new_zeros([1, aux_feats.size()[1]])
        cat_feats = torch.cat([in_core_feats, aux_feats, pad_feat], dim=0)
        conv_feats = cat_feats[conv_ids].flatten(1)

        # Get gradient tensors
        grad_conv_feats = torch.mm(grad_out_core_feats, weight)
        grad_weight = torch.mm(grad_out_core_feats.t(), conv_feats)
        grad_bias = grad_out_core_feats.sum(dim=0) if bias is not None else None

        num_core_feats, in_channels = in_core_feats.size()
        grad_conv_feats = grad_conv_feats.view(-1, in_channels)
        grad_cat_feats = torch.zeros_like(cat_feats).index_add_(0, conv_ids.flatten(), grad_conv_feats)

        grad_in_core_feats = grad_cat_feats[:num_core_feats]
        grad_aux_feats = grad_cat_feats[num_core_feats:-1]
        grad_conv_ids = None

        return grad_in_core_feats, grad_aux_feats, grad_conv_ids, grad_weight, grad_bias
