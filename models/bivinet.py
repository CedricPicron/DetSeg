"""
BiViNet modules and build function.
"""
import copy

from torch import nn
from torch.nn.utils import clip_grad_norm_

from .backbone import build_backbone
from .bicore import build_bicore
from .heads.objectness import build_obj_head


class BiViNet(nn.Module):
    """
    Class implementing the BiViNet module.

    Attributes:
        backbone (nn.Module): Module implementing the BiViNet backbone.
        projs (nn.ModuleList): List of size [num_core_maps] implementing backbone to core projection modules.
        cores (nn.ModuleList): List of size [num_core_layers] with concatenated core modules.
        heads (nn.ModuleList): List of size [num_heads] with BiViNet head modules.
    """

    def __init__(self, backbone, core_feat_sizes, core, num_core_layers, heads):
        """
        Initializes the BiViNet module.

        Args:
            backbone (nn.Module): Module implementing the BiViNet backbone.
            core_feat_sizes (List): List of size [num_core_maps] containing the feature size of each core feature map.
            core (nn.Module): Module implementing the BiViNet core.
            num_core_layers (int): Number of concatenated core layers.
            heads (List): List of size [num_heads] with BiViNet head modules.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set backbone, core and heads attributes
        self.backbone = backbone
        self.cores = nn.ModuleList([copy.deepcopy(core) for _ in range(num_core_layers)])
        self.heads = nn.ModuleList(heads)

        # Build backbone to core projection modules
        f0s = backbone.feat_sizes
        f1s = core_feat_sizes[:len(f0s)]
        self.projs = nn.ModuleList(nn.Conv2d(f0, f1, kernel_size=1) for f0, f1 in zip(f0s, f1s))
        self.projs.append(nn.Conv2d(f0s[-1], core_feat_sizes[-1], kernel_size=3, stride=2, padding=1))

    @staticmethod
    def get_param_families():
        """
        Method returning the BiViNet parameter families.

        Returns:
            List of strings containing the BiViNet parameter families.
        """

        return ['backbone', 'projs', 'core', 'heads']

    def evaluate_feat_maps(self, feat_maps, tgt_dict, optimizer, **kwargs):
        """
        Method evaluating core feature maps.

        Loss and analysis dictionaries are computed from the input feature maps using the module's heads.
        The model parameters are updated during training by backpropagating the loss terms from the loss dictionary.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, H, W, feat_size].
            tgt_dict (Dict): Target dictionary containing following keys:
                - labels (IntTensor): tensor of shape [num_targets_total] containing the class indices;
                - boxes (FloatTensor): boxes [num_targets_total, 4] in (center_x, center_y, width, height) format;
                - sizes (IntTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries;
                - masks (List): List of size [num_maps] with padded masks of shape [num_targets_total, height, width].

            optimizer (torch.optim.Optimizer): Optional optimizer updating the BiViNet parameters during training.
            kwargs(Dict): Dictionary of keyword arguments, potentially containing following keys:
                - max_grad_norm (float): maximum norm of optimizer update during training (clipped if larger).

        Returns:
            loss_dict (Dict): Dictionary of different weighted loss terms used for backpropagation during training.
            analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.
        """

        # Initialize loss and analysis dictionaries
        loss_dict = {}
        analysis_dict = {}

        # Populate loss and analysis dictionaries from head outputs
        for head in self.heads:
            head_loss_dict, head_analysis_dict = head(feat_maps, tgt_dict)
            loss_dict.update(head_loss_dict)
            analysis_dict.update(head_analysis_dict)

        # Return loss and analysis dictionaries (validation only)
        if optimizer is None:
            return loss_dict, analysis_dict

        # Update model parameters (training only)
        optimizer.zero_grad()
        loss = sum(loss_dict.values())
        loss.backward()
        clip_grad_norm_(self.parameters(), kwargs['max_grad_norm']) if 'max_grad_norm' in kwargs else None
        optimizer.step()

        return loss_dict, analysis_dict

    def forward(self, images, tgt_dict=None, optimizer=None, **kwargs):
        """
        Forward method of the BiViNet module.

        Args:
            images (NestedTensor): NestedTensor consisting of:
                - images.tensor (FloatTensor): padded images of shape [batch_size, 3, H, W];
                - images.mask (BoolTensor): boolean masks encoding inactive pixels of shape [batch_size, H, W].

            tgt_dict (Dict): Optional target dictionary used during training and validation containing following keys:
                - labels (IntTensor): tensor of shape [num_targets_total] containing the class indices;
                - boxes (FloatTensor): boxes [num_targets_total, 4] in (center_x, center_y, width, height) format;
                - sizes (IntTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries;
                - masks (ByteTensor): padded segmentation masks of shape [num_targets_total, height, width].

            optimizer (torch.optim.Optimizer): Optional optimizer updating the BiViNet parameters during training.
            kwargs(Dict): Dictionary of keyword arguments, potentially containing following keys:
                - max_grad_norm (float): maximum norm of optimizer update during training (clipped if larger).

       Returns:
            * If tgt_dict is not None and optimizer is not None (i.e. during training):
                loss_dict (Dict): Dictionary of different weighted loss terms used for backpropagation during training.
                analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

            * If tgt_dict is not None and optimizer is None (i.e. during validation):
                pred_dict (Dict): Dictionary containing different predictions from the last core layer.
                loss_dict (Dict): Dictionary of different weighted loss terms used for backpropagation during training.
                analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

            * If tgt_dict is None (i.e. during testing):
                pred_dict (Dict): Dictionary containing different predictions from the last core layer.
        """

        # Get backbone feature maps
        feat_maps, _ = self.backbone(images)

        # Get initial core feature maps by projecting backbone feature maps
        map_ids = [min(i, len(feat_maps)-1) for i in range(len(self.projs))]
        feat_maps = [proj(feat_maps[i]).permute(0, 2, 3, 1) for i, proj in zip(map_ids, self.projs)]

        # Evaluate initial core feature maps (training/validation only)
        if tgt_dict is not None:
            loss_dict, analysis_dict = self.evaluate_feat_maps(feat_maps, tgt_dict, optimizer, **kwargs)

        # Iteratively update and evaluate core feature maps
        for i, core in enumerate(self.cores, 1):
            feat_maps = core(feat_maps)

            # Evaluate updated core feature maps (training/validation only)
            if tgt_dict is not None:
                layer_loss_dict, layer_analysis_dict = self.evaluate_feat_maps(feat_maps, tgt_dict, optimizer, **kwargs)
                loss_dict.update({f'{k}_{i}': v for k, v in layer_loss_dict.items()})
                analysis_dict.update({f'{k}_{i}': v for k, v in layer_analysis_dict.items()})

            # Detach feature maps (training only)
            if optimizer is not None:
                feat_maps = [feat_map.detach() for feat_map in feat_maps]

        # Get prediction dictionary (validation/testing only)
        if tgt_dict is None or optimizer is None:
            pred_dict = {k: v for head in self.heads for k, v in head(feat_maps).items()}

        # Return prediction dictionary (testing only)
        if tgt_dict is None:
            return pred_dict

        # Return prediction, loss and analysis dictionaries (validation only)
        if optimizer is None:
            return pred_dict, loss_dict, analysis_dict

        return loss_dict, analysis_dict


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
    bivinet = BiViNet(backbone, core_feat_sizes, core, args.num_core_layers, heads)

    return bivinet
