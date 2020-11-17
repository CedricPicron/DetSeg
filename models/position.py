"""
Position encoder modules and build function.
"""
import math
import torch
from torch import nn, Tensor


class SinePositionEncoder(nn.Module):
    """
    Class implementing the SinePositionEncoder module.

    It is a two-dimensional position encoder with sines and cosines, relative to padding mask.

    Attributes:
        temperature (int): Sample points will cover between 1/temperature and 1 of the (co)sine period.
        normalize (bool): Normalize sample points to given scale.
        scale (float): Scale used during normalization (ignored when normalize is False).
    """

    def __init__(self, temperature=10000, normalize=True, scale=2*math.pi):
        """
        Initializes the SinePositionEncoder module.

        Args:
            temperature (int): Sample points will cover between 1/temperature and 1 of the (co)sine period.
            normalize (bool): Normalize sample points to given scale.
            scale (float): Scale used during normalization (ignored when normalize is False).

        Raises:
            ValueError: Raised when normalize is False and scale was passed.
        """

        super().__init__()

        if scale is not None and normalize is False:
            raise ValueError("Normalize should be True if scale is passed.")

        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    def forward(self, features: Tensor, feature_masks: Tensor) -> Tensor:
        """
        Forward method of the SinePositionEncoder module.

        Args:
             features (FloatTensor): Features of shape [batch_size, feat_dim, fH, fW].
             feature_masks (BoolTensor): Boolean masks encoding inactive pixels of shape [batch_size, fH, fW].

        Returns:
            pos (Tensor): Position encodings of shape [batch_size, feat_dim, fH, fW].
        """

        not_mask = ~feature_masks
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        feature_size_per_dim = features.shape[1] // 2
        dim_t = torch.arange(feature_size_per_dim, dtype=torch.float32, device=features.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / feature_size_per_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.to(features.dtype)

        return pos


def build_position_encoder(args):
    """
    Build position encoder from command-line arguments.

    Args:
        args (argsparse.Namespace): Command-line arguments.

    Returns:
        position_encoder (nn.Module): The specified position encoder module.

    Raises:
        ValueError: Raised when unknown args.position_encoding is provided.
    """

    if args.position_encoding == 'sine':
        position_encoder = SinePositionEncoder()
    else:
        raise ValueError(f'Unknown position encoding {args.position_encoding} was provided.')

    return position_encoder
