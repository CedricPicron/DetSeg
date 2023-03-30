"""
Collection of anchor-related modules.
"""

from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from torch import nn

from models.build import MODELS
from structures.boxes import Boxes


@MODELS.register_module()
class AnchorGenerator(nn.Module):
    """
    Class implementing the AnchorGenerator module.

    Attributes:
        generator (DefaultAnchorGenerator): Default anchor generator module from Detectron2.
        num_cell_anchors (int): Integer containing the number of cell anchors.
    """

    def __init__(self, map_ids, num_sizes=1, scale_factor=4.0, aspect_ratios=None):
        """
        Initializes the AnchorGenerator module.

        Args:
            map_ids (Tuple): Tuple of size [num_maps] containing the map ids (i.e. downsampling exponents) of each map.
            num_sizes (int): Integer containing the number of different anchor sizes per aspect ratio (default=1).
            scale_factor (float): Factor scaling the anchors w.r.t. non-overlapping tiling anchors (default=4.0).
            aspect_ratios (Tuple): Tuple [num_aspect_ratios] with different anchor aspect ratios (default=None).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Get and set generator attribute
        sizes = [[scale_factor * 2**(i+j/num_sizes) for j in range(num_sizes)] for i in map_ids]
        aspect_ratios = aspect_ratios if aspect_ratios is not None else (1.0,)
        strides = [2**i for i in map_ids]
        self.generator = DefaultAnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios, strides=strides)

        # Set attribute containing the number of cell anchors
        self.num_cell_anchors = num_sizes * len(aspect_ratios)

    def forward(self, feat_maps, **kwargs):
        """
        Forward method of the AnchorGenerator module.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            anchors (Boxes): Structure containing axis-aligned anchor boxes of size [num_feats * num_cell_anchors].
        """

        # Get anchors
        anchors = self.generator(feat_maps)
        anchors = [Boxes(map_anchors.tensor, format='xyxy') for map_anchors in anchors]
        anchors = Boxes.cat(anchors).to(feat_maps[0].device)

        return anchors
