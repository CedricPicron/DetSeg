"""
BiViNet modules and build function.
"""
import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from .backbone import build_backbone
from .bicore import build_bicore
from .heads.segmentation import build_seg_heads


class BiViNet(nn.Module):
    """
    Class implementing the BiViNet module.

    Attributes:
        backbone (nn.Module): Module implementing the BiViNet backbone.
        projs (nn.ModuleList): List of size [num_core_maps] implementing backbone to core projection modules.
        cores (nn.ModuleList): List of size [num_core_layers] with concatenated core modules.
        heads (nn.ModuleList): List of size [num_heads] with BiViNet head modules.
        mask_types (Set): Set of strings containing the names of the required mask types.
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

        # Get mask types required by the heads
        self.mask_types = {mask_type for head in heads for mask_type in head.required_mask_types()}

    @staticmethod
    def get_param_families():
        """
        Method returning the BiViNet parameter families.

        Returns:
            List of strings containing the BiViNet parameter families.
        """

        return ['backbone', 'projs', 'core', 'heads']

    @staticmethod
    def get_binary_masks(tgt_sizes, tgt_masks):
        """
        Method obtaining the binary mask (object + background) corresponding to each batch entry.

        Args:
            tgt_sizes (IntTensor): Tensor of shape [batch_size+1] with the cumulative target sizes of batch entries.
            tgt_masks (ByteTensor): Padded segmentation masks of shape [num_targets_total, max_iH, max_iW].

        Returns:
            binary_masks (ByteTensor): Binary (object + background) segmentation masks of shape [batch_size, iH, iW].
        """

        binary_masks = torch.cat([tgt_masks[i0:i1].any(dim=0) for i0, i1 in zip(tgt_sizes[:-1], tgt_sizes[1:])], dim=0)

        return binary_masks

    @staticmethod
    def downsample_masks(in_masks, feat_maps):
        """
        Method downsampling the full-resolution 'in_masks' to the same resolutions as found in 'feat_maps'.

        Args:
            in_masks (ByteTensor): Padded full-resolution segmentation masks of shape [*, max_iH, max_iW].
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, fH, fW, feat_size].

        Returns:
            out_masks (List): List of size [num_maps] with downsampled FloatTensor masks of shape [*, fH, fW].
        """

        # Initialize list of downsampled output masks
        device = in_masks.device
        num_targets_total = in_masks.shape[0]
        map_sizes = [feat_map.shape[1:-1] for feat_map in feat_maps]
        out_masks = [torch.zeros(num_targets_total, *map_size, device=device) for map_size in map_sizes]

        # Prepare masks for convolution and get convolution kernel
        masks = in_masks[None, None].float()
        average_kernel = torch.ones(1, 1, 3, 3, device=device)/9

        # Compue list of downsampled output masks
        for i in range(len(out_masks)):
            while masks.shape[-2:] != map_sizes[i]:
                masks = F.conv2d(masks, average_kernel, stride=2, padding=1)

            out_masks[i] = masks

        return out_masks

    def prepare_masks(self, tgt_dict, feat_maps):
        """
        Method preparing the segmentation masks.

        For each mask type, the preparation consists of 2 steps:
            1) Get the desired full-resolution masks.
            2) Downsample the full-resolution masks to the same resolutions as found in 'feat_maps'.

        Args:
            tgt_dict (Dict): Target dictionary containing at least following keys:
                - sizes (IntTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries;
                - masks (ByteTensor): padded segmentation masks of shape [num_targets_total, max_iH, max_iW].

            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, fH, fW, feat_size].

        Returns:
            tgt_dict (List): Updated target dictionary with 'masks' key removed and potential additional keys:
                - binary_masks (List): binary (object + background) segmentation masks of shape [batch_size, fH, fW].

        Raises:
            ValueError: Raised when an unknown mask type is found in the 'self.mask_types' set.
        """

        # Get the required masks
        for mask_type in self.mask_types:
            if mask_type == 'binary_masks':
                binary_masks = self.get_binary_masks(tgt_dict['sizes'], tgt_dict['masks'])
                tgt_dict['binary_masks'] = self.downsample_masks(binary_masks, feat_maps)

            else:
                raise ValueError(f"Unknown mask type '{mask_type}' in 'self.mask_types'.")

        # Remove 'masks' key from target dictionary
        del tgt_dict['masks']

        return tgt_dict

    def evaluate_feat_maps(self, feat_maps, tgt_dict, optimizer, **kwargs):
        """
        Method evaluating core feature maps.

        Loss and analysis dictionaries are computed from the input feature maps using the module's heads.
        The model parameters are updated during training by backpropagating the loss terms from the loss dictionary.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, fH, fW, feat_size].
            tgt_dict (Dict): Target dictionary containing following keys:
                - labels (IntTensor): tensor of shape [num_targets_total] containing the class indices;
                - boxes (FloatTensor): boxes [num_targets_total, 4] in (center_x, center_y, width, height) format;
                - sizes (IntTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries;
                - masks (List): List of size [num_maps] with target masks of shape [num_targets_total, fH, fW].

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
                - images.tensor (FloatTensor): padded images of shape [batch_size, 3, max_iH, max_iW];
                - images.mask (BoolTensor): masks encoding inactive pixels of shape [batch_size, max_iH, max_iW].

            tgt_dict (Dict): Optional target dictionary used during training and validation containing following keys:
                - labels (IntTensor): tensor of shape [num_targets_total] containing the class indices;
                - boxes (FloatTensor): boxes [num_targets_total, 4] in (center_x, center_y, width, height) format;
                - sizes (IntTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries;
                - masks (ByteTensor): padded segmentation masks of shape [num_targets_total, iH, iW].

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

        # Prepare masks and evaluate initial core feature maps (training/validation only)
        if tgt_dict is not None:
            tgt_dict = self.prepare_masks(tgt_dict, feat_maps)
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
    heads = [*build_seg_heads(args)]

    # Build BiViNet module
    bivinet = BiViNet(backbone, core_feat_sizes, core, args.num_core_layers, heads)

    return bivinet
