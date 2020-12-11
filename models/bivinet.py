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
        heads (nn.ModuleList): List of size [num_core_layers+1] with copies of BiViNet head modules for each layer.
        map_types (Set): Set of strings containing the names of the required map types.
        num_classes (int, optional): Integer containing the number of object classes (without background).
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

        Raises:
            ValueError: Error when incompatible heads designed for a different number of classes are provided.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set backbone, core and heads attributes
        self.backbone = backbone
        self.cores = nn.ModuleList([copy.deepcopy(core) for _ in range(num_core_layers)])
        self.heads = nn.ModuleList([copy.deepcopy(nn.ModuleList(heads)) for _ in range(num_core_layers+1)])

        # Build backbone to core projection modules
        f0s = backbone.feat_sizes
        f1s = core_feat_sizes[:len(f0s)]
        self.projs = nn.ModuleList(nn.Conv2d(f0, f1, kernel_size=1) for f0, f1 in zip(f0s, f1s))
        self.projs.append(nn.Conv2d(f0s[-1], core_feat_sizes[-1], kernel_size=3, stride=2, padding=1))

        # Get map types required by the heads
        self.map_types = {map_type for head in heads for map_type in head.required_map_types()}

        # Set number of classes attribute if requested by one of the heads
        num_classes_set = {head.num_classes for head in heads if hasattr(head, 'num_classes')}

        if len(num_classes_set) == 1:
            self.num_classes = num_classes_set.pop()

        elif len(num_classes_set) > 1:
            raise ValueError("Incompatible heads designed for a different number of classes are provided.")

    @staticmethod
    def get_param_families():
        """
        Method returning the BiViNet parameter families.

        Returns:
            List of strings containing the BiViNet parameter families.
        """

        return ['backbone', 'projs', 'core', 'heads']

    @staticmethod
    def get_feat_masks(img_masks, map_sizes):
        """
        Method obtaining feature masks corresponding to each feature map encoding features from padded image regions.

        Args:
            img_masks (BoolTensor): masks encoding padded pixels of shape [batch_size, max_iH, max_iW].
            map_sizes (List): List of size [num_core_maps] with tuples of requested map sizes (fH, fW).

        Returns:
            feat_masks (List): List of size [num_core_maps] with padding feature masks of shape [batch_size, fH, fW].
        """

        padding_maps = BiViNet.downsample_masks(img_masks, map_sizes)
        feat_masks = [padding_map > 0.5 for padding_map in padding_maps]

        return feat_masks

    @staticmethod
    def get_binary_masks(tgt_sizes, tgt_masks):
        """
        Method obtaining the binary mask (object + background) corresponding to each batch entry.

        Args:
            tgt_sizes (LongTensor): Tensor of shape [batch_size+1] with the cumulative target sizes of batch entries.
            tgt_masks (ByteTensor): Padded segmentation masks of shape [num_targets_total, max_iH, max_iW].

        Returns:
            binary_masks (ByteTensor): Binary segmentation masks of shape [batch_size, max_iH, max_iW].
        """

        binary_masks = torch.stack([tgt_masks[i0:i1].any(dim=0) for i0, i1 in zip(tgt_sizes[:-1], tgt_sizes[1:])])

        return binary_masks

    def get_semantic_maps(self, tgt_labels, tgt_sizes, tgt_masks):
        """
        Method obtaining the full-resolution semantic segmentation map corresponding to each batch entry.

        Args:
            tgt_labels (LongTensor): Tensor of shape [num_targets_total] containing the class indices.
            tgt_sizes (LongTensor): Tensor of shape [batch_size+1] with the cumulative target sizes of batch entries.
            tgt_masks (ByteTensor): Padded segmentation masks of shape [num_targets_total, max_iH, max_iW].

        Returns:
            semantic_maps (LongTensor): Semantic maps with class indices of shape [batch_size, max_iH, max_iW].
        """

        # Initialize semantic maps
        batch_size = len(tgt_sizes) - 1
        tensor_kwargs = {'dtype': torch.long, 'device': tgt_masks.device}
        semantic_maps = torch.full((batch_size, *tgt_masks.shape[-2:]), self.num_classes, **tensor_kwargs)

        # Compute full-resolution semantic masks for each batch entry
        for i, i0, i1 in zip(range(batch_size), tgt_sizes[:-1], tgt_sizes[1:]):
            for tgt_label, tgt_mask in zip(tgt_labels[i0:i1], tgt_masks[i0:i1]):
                semantic_maps[i].masked_fill_(tgt_mask, tgt_label)

        return semantic_maps

    @staticmethod
    def downsample_masks(masks, map_sizes):
        """
        Method downsampling the full-resolution masks to maps with the given map sizes.

        Args:
            masks (ByteTensor): Padded full-resolution masks of shape [*, max_iH, max_iW].
            map_sizes (List): List of size [num_core_maps] with tuples of requested map sizes (fH, fW).

        Returns:
            maps_list (List): List of size [num_core_maps] with downsampled FloatTensor maps of shape [*, fH, fW].
        """

        # Save original size of masks and get initial full-resolution maps
        original_size = masks.shape
        maps = masks.float().view(-1, 1, *original_size[-2:])

        # Get averaging convolutional kernel
        device = masks.device
        average_kernel = torch.ones(1, 1, 3, 3, device=device)/9

        # Initialize list of downsampled output maps
        maps_list = [torch.zeros(*original_size[:-2], *map_size, device=device) for map_size in map_sizes]

        # Compute list of downsampled output maps
        for i in range(len(maps_list)):
            while maps.shape[-2:] != map_sizes[i]:
                maps = F.pad(maps, (1, 1, 1, 1), mode='replicate')
                maps = F.conv2d(maps, average_kernel, stride=2, padding=0)

            maps_list[i] = maps.view(*original_size[:-2], *maps.shape[-2:])

        return maps_list

    @staticmethod
    def downsample_index_maps(index_maps, map_sizes):
        """
        Method downsampling the full-resolution index maps to maps with the given map sizes.

        Args:
            index_maps (LongTensor): Padded full-resolution index maps of shape [*, max_iH, max_iW].
            map_sizes (List): List of size [num_core_maps] with tuples of requested map sizes (fH, fW).

        Returns:
            maps_list (List): List of size [num_core_maps] with downsampled index maps of shape [*, fH, fW].
        """

        # Save original size of index maps and convert index maps into desired format
        original_size = index_maps.shape
        maps = index_maps.view(-1, *original_size[-2:])

        # Initialize list of downsampled index maps
        maps_list = [torch.zeros(*original_size[:-2], *map_size).to(index_maps) for map_size in map_sizes]

        # Compute list of downsampled index maps
        for i in range(len(maps_list)):
            while maps.shape[-2:] != map_sizes[i]:
                maps = maps[:, ::2, ::2]

            maps_list[i] = maps.view(*original_size[:-2], *maps.shape[-2:]).contiguous()

        return maps_list

    def prepare_tgt_maps(self, tgt_dict, map_sizes):
        """
        Method preparing the target segmentation maps and adding them to the target dictionary.

        For each map type, the preparation consists of 2 steps:
            1) Get the desired full-resolution masks.
            2) Downsample the full-resolution masks to maps with the same resolutions as found in 'feat_maps'.

        Args:
            tgt_dict (Dict): Target dictionary containing at least following keys:
                - labels (LongTensor): tensor of shape [num_targets_total] containing the class indices;
                - sizes (LongTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries;
                - masks (ByteTensor): padded segmentation masks of shape [num_targets_total, max_iH, max_iW].

            map_sizes (List): List of size [num_core_maps] with tuples of requested map sizes (fH, fW).

        Returns:
            tgt_dict (List): Updated target dictionary with 'masks' key removed and potential additional keys:
                - binary_maps (List): binary (object + background) segmentation maps of shape [batch_size, fH, fW];
                - semantic_maps (List): semantic segmentation maps with class indices of shape [batch_size, fH, fW].

        Raises:
            ValueError: Raised when an unknown map type is found in the 'self.map_types' set.
        """

        # Get the required maps
        for map_type in self.map_types:
            if map_type == 'binary_maps':
                binary_masks = BiViNet.get_binary_masks(tgt_dict['sizes'], tgt_dict['masks'])
                tgt_dict[map_type] = BiViNet.downsample_masks(binary_masks, map_sizes)

            elif map_type == 'semantic_maps':
                semantic_maps = self.get_semantic_maps(tgt_dict['labels'], tgt_dict['sizes'], tgt_dict['masks'])
                tgt_dict[map_type] = BiViNet.downsample_index_maps(semantic_maps, map_sizes)

            else:
                raise ValueError(f"Unknown map type '{map_type}' in 'self.map_types'.")

        # Remove 'masks' key from target dictionary
        del tgt_dict['masks']

        return tgt_dict

    def train_evaluate(self, feat_maps, feat_masks, tgt_dict, optimizer, layer_id, **kwargs):
        """
        Method performing the training/evaluation step for the given feature maps.

        Loss and analysis dictionaries are computed from the input feature maps using the module's heads.
        The model parameters are updated during training by backpropagating the loss terms from the loss dictionary.

        Args:
            feat_maps (List): List of size [num_core_maps] with feature maps of shape [batch_size, fH, fW, feat_size].
            feat_masks (List): List of size [num_core_maps] with padding feature masks of shape [batch_size, fH, fW].
            tgt_dict (Dict): Target dictionary containing following keys:
                - labels (LongTensor): tensor of shape [num_targets_total] containing the class indices;
                - boxes (FloatTensor): boxes [num_targets_total, 4] in (center_x, center_y, width, height) format;
                - sizes (LongTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries;
                - binary_maps (List, optional): binary segmentation maps of shape [batch_size, fH, fW];
                - semantic_maps (List): semantic segmentation maps of shape [batch_size, fH, fW].

            optimizer (torch.optim.Optimizer): Optional optimizer updating the BiViNet parameters during training.
            layer_id (int): Layer index, with 0 for the backbone layer and higher integers for the core layers.
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
        for head in self.heads[layer_id]:
            head_loss_dict, head_analysis_dict = head(feat_maps, feat_masks, tgt_dict, **kwargs)
            loss_dict.update({f'{k}_{layer_id}': v for k, v in head_loss_dict.items()})
            analysis_dict.update({f'{k}_{layer_id}': v for k, v in head_analysis_dict.items()})

        # Add total layer loss to analysis dictionary
        with torch.no_grad():
            analysis_dict[f'loss_{layer_id}'] = sum(loss_dict.values())

        # Return loss and analysis dictionaries (validation only)
        if optimizer is None:
            return loss_dict, analysis_dict

        # Backpropagate loss (training only)
        loss = sum(loss_dict.values())
        loss.backward()

        return loss_dict, analysis_dict

    def forward(self, images, tgt_dict=None, optimizer=None, **kwargs):
        """
        Forward method of the BiViNet module.

        Args:
            images (NestedTensor): NestedTensor consisting of:
                - images.tensor (FloatTensor): padded images of shape [batch_size, 3, max_iH, max_iW];
                - images.mask (BoolTensor): masks encoding padded pixels of shape [batch_size, max_iH, max_iW].

            tgt_dict (Dict): Optional target dictionary used during training and validation containing following keys:
                - labels (LongTensor): tensor of shape [num_targets_total] containing the class indices;
                - boxes (FloatTensor): boxes [num_targets_total, 4] in (center_x, center_y, width, height) format;
                - sizes (LongTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries;
                - masks (ByteTensor): padded segmentation masks of shape [num_targets_total, max_iH, max_iW].

            optimizer (torch.optim.Optimizer): Optional optimizer updating the BiViNet parameters during training.
            kwargs(Dict): Dictionary of keyword arguments, potentially containing following keys:
                - max_grad_norm (float): maximum norm of optimizer update during training (clipped if larger);
                - visualize (bool): boolean indicating whether in visualization mode or not.

       Returns:
            * If tgt_dict is not None and optimizer is not None (i.e. during training):
                loss_dict (Dict): Dictionary of different weighted loss terms used for backpropagation during training.
                analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

            * If tgt_dict is not None and optimizer is None (i.e. during validation):
                pred_dict (Dict): Dictionary containing different predictions from the last core layer.
                loss_dict (Dict): Dictionary of different weighted loss terms used for backpropagation during training.
                analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

            * If in validation mode (see above) with the keyword argument 'visualize=True' (i.e. during visualization):
                images_dict (Dict): Dictionary of different annotated images based on predictions and targets.
                loss_dict (Dict): Dictionary of different weighted loss terms used for backpropagation during training.
                analysis_dict (Dict): Dictionary of different analyses used for logging purposes only.

            * If tgt_dict is None (i.e. during testing):
                pred_dict (Dict): Dictionary containing different predictions from the last core layer.
        """

        # Reset gradients of model parameters (training only)
        if optimizer is not None:
            optimizer.zero_grad()

        # Get backbone feature maps
        feat_maps, _ = self.backbone(images)

        # Get initial core feature maps by projecting backbone feature maps
        map_ids = [min(i, len(feat_maps)-1) for i in range(len(self.projs))]
        feat_maps = [proj(feat_maps[i]).permute(0, 2, 3, 1) for i, proj in zip(map_ids, self.projs)]

        # Do some preparation and train/evaluate initial core feature maps (training/validation only)
        if tgt_dict is not None:
            map_sizes = [tuple(feat_map.shape[1:-1]) for feat_map in feat_maps]
            feat_masks = BiViNet.get_feat_masks(images.mask, map_sizes)
            tgt_dict = self.prepare_tgt_maps(tgt_dict, map_sizes)

            train_eval_args = (feat_masks, tgt_dict, optimizer)
            loss_dict, analysis_dict = self.train_evaluate(feat_maps, *train_eval_args, layer_id=0, **kwargs)

        # Iteratively detach, update and train from and/or evaluate core feature maps
        for i, core in enumerate(self.cores, 1):

            # Set correct requires_grad attributes and detach core feature maps (training only)
            if optimizer is not None:
                feat_maps = [feat_map.detach() for feat_map in feat_maps]

            # Update core feature maps (training/validation only)
            feat_maps = core(feat_maps)

            # Train/evaluate updated core feature maps (training/validation only)
            if tgt_dict is not None:
                layer_loss_dict, layer_analysis_dict = self.train_evaluate(feat_maps, *train_eval_args, i, **kwargs)
                loss_dict.update(layer_loss_dict)
                analysis_dict.update(layer_analysis_dict)

        # Get prediction dictionary (validation/testing only)
        if optimizer is None:
            pred_dict = {k: v for head in self.heads[0] for k, v in head(feat_maps).items()}

        # Return prediction dictionary (testing only)
        if tgt_dict is None:
            return pred_dict

        # Return desired dictionaries (validation/visualization only)
        if optimizer is None:

            # Return prediction, loss and analysis dictionaries (validation only)
            if not kwargs.setdefault('visualize', False):
                return pred_dict, loss_dict, analysis_dict

            # Get and return annotated images, loss and analysis dictionaries (visualization only)
            images_dict = {}
            for head in self.heads[0]:
                images_dict = {**images_dict, **head.visualize(images, pred_dict, tgt_dict)}

            return images_dict, loss_dict, analysis_dict

        # Clip gradient when positive maximum norm is provided (training only)
        if kwargs.setdefault('max_grad_norm', -1.0) > 0.0:
            clip_grad_norm_(self.parameters(), kwargs['max_grad_norm'])

        # Update model parameters (training only)
        optimizer.step()

        # Compute average heads state dictionary (training only)
        avg_heads_state_dict = {}
        for key_values in zip(*[heads.state_dict().items() for heads in self.heads]):
            keys, values = list(zip(*key_values))
            avg_heads_state_dict[keys[0]] = sum(values) / len(values)

        # Load average heads state dictionary into different heads (training only)
        [heads.load_state_dict(avg_heads_state_dict) for heads in self.heads]

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
