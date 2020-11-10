"""
BiViNet modules and build function.
"""

from torch import nn

from .backbone import build_backbone
from .bicore import build_bicore
from .heads.objectness import build_obj_head


class BiViNet(nn.Module):
    """
    Class implementing the BiViNet module.

    Attributes:
        backbone (nn.Module): Module implementing the BiViNet backbone.
        projs (nn.ModuleList): List of size [num_core_maps] implementing backbone to core projection modules.
        core (nn.Module): Module implementing the BiViNet core.
        heads (nn.ModuleList): List of size [num_heads] with BiViNet head modules.
    """

    def __init__(self, backbone, core_feat_sizes, core, heads):
        """
        Initializes the BiViNet module.

        Args:
            backbone (nn.Module): Module implementing the BiViNet backbone.
            core_feat_sizes (List): List of size [num_core_maps] containing the feature size of each core feature map.
            core (nn.Module): Module implementing the BiViNet core.
            heads (List): List of size [num_heads] with BiViNet head modules.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set backbone, core and heads attributes
        self.backbone = backbone
        self.core = core
        self.heads = nn.ModuleList(heads)

        # Build backbone to core projection modules
        f0s = backbone.feat_sizes
        f1s = core_feat_sizes[:len(f0s)]
        self.projs = nn.ModuleList(nn.Conv2d(f0, f1, kernel_size=1) for f0, f1 in zip(f0s, f1s))
        self.projs.append(nn.Conv2d(f0s[-1], core_feat_sizes[-1], kernel_size=3, stride=2, padding=1))

    def forward(self, images=None, core_feat_maps=None, tgt_dict=None):
        """
        Forward method of the BiViNet module.

        Args:
            images (NestedTensor): If provided, a NestedTensor consisting of:
               - images.tensor (FloatTensor): padded images of shape [batch_size, 3, H, W];
               - images.mask (BoolTensor): boolean masks encoding inactive pixels of shape [batch_size, H, W].
            core_feat_maps (List): If provided, a list of size [num_core_maps] with core feature maps to be updated.
            tgt_dict (Dict): Dictionary containing targets used by the heads during trainval (None during testing).

        Returns:
            * If tgt_dict is not None (i.e. during training and validation):
                core_feat_maps (List): List of size [num_core_maps] of initialized or updated core feature maps.
                loss_dict (Dict): Dictionary with loss terms from different heads (used for backpropagtion).
                analysis_dict (Dict): Dictionary with analysis terms from different heads (not for backpropagation).

            * If tgt_dict is None (i.e. during testing):
                pred_dict (Dict): Dictionary with predictions from different heads.
        """

        # Check inputs
        check = (images is not None and core_feat_maps is None) or (images is None and core_feat_maps is not None)
        assert_msg = "exactly one of the inputs 'images' and 'core_feat_maps' must be provided"
        assert check, assert_msg

        # Update core feature maps (if core feature maps are provided as input)
        if core_feat_maps is not None:
            core_feat_maps = self.core(core_feat_maps)

        # Initialize core feature maps (if images are provided as input)
        elif images is not None:
            backbone_feat_maps, _ = self.backbone(images)
            map_ids = [min(i, len(backbone_feat_maps)-1) for i in range(len(self.projs))]
            core_feat_maps = [proj(backbone_feat_maps[i]) for i, proj in zip(map_ids, self.projs)]
            core_feat_maps = [feat_map.permute(0, 2, 3, 1) for feat_map in core_feat_maps]

        # Get prediction dictionary from head predictions and return (testing only)
        if tgt_dict is None:
            pred_dict = {k: v for head in self.heads for k, v in head(core_feat_maps).items()}
            return pred_dict

        # Initialize loss and analysis dictionaries (training/validation only)
        loss_dict = {}
        analysis_dict = {}

        # Populate loss and analysis dictionaries from head outputs (training/validation only)
        for head in self.heads:
            head_loss_dict, head_analysis_dict = head(core_feat_maps, tgt_dict)
            loss_dict.update(head_loss_dict)
            analysis_dict.update(head_analysis_dict)

        return core_feat_maps, loss_dict, analysis_dict


def build_bivinet(args):
    """
    Build BiViNet module from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        bivinet (BiViNet): The specified BiViNet module.
    """

    # Check command-line arguments
    check = args.max_resolution_id > args.min_resolution_id
    assert_msg = "'--max_resolution_id' should be larger than '--min_resolution_id'"
    assert check, assert_msg

    # Get core feature sizes
    map_ids = range(args.min_resolution_id, args.max_resolution_id+1)
    core_feat_sizes = [min((args.base_feat_size * 2**i, args.max_feat_size)) for i in map_ids]

    # Build backbone, core and desired heads
    backbone = build_backbone(args)
    core = build_bicore(args)
    heads = [build_obj_head(args)]

    # Build BiViNet module
    bivinet = BiViNet(backbone, core_feat_sizes, core, heads)

    return bivinet
