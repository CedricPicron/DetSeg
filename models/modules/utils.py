"""
Collection of utility modules.
"""

import torch
from torch import nn

from models.build import build_model, MODELS
from structures.boxes import Boxes


@MODELS.register_module()
class ApplyAll(nn.Module):
    """
    Class implementing the ApplyAll module.

    The ApplyAll module applies its underlying module to all inputs from the given input list.

    Attributes:
        module (Sequential): Underlying module applied to all inputs from the input list.
    """

    def __init__(self, module_cfg, **kwargs):
        """
        Initializes the ApplyAll module.

        Args:
            module_cfg (Dict): Configuration dictionary specifying the underlying module.
            kwargs (Dict): Dictionary of unused keyword arguments.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build underlying module
        self.module = build_model(module_cfg, sequential=True)

    def forward(self, in_list, **kwargs):
        """
        Forward method of the ApplyAll module.

        Args:
            in_list (List): List with inputs to be processed by the underlying module.
            kwargs (Dict): Dictionary of keyword arguments passed to the underlying module.

        Returns:
            out_list (List): List with resulting outputs of the underlying module.
        """

        # Apply underlying module to all inputs from input list
        out_list = [self.module(input, **kwargs) for input in in_list]

        return out_list


@MODELS.register_module()
class ApplyToSelected(nn.Module):
    """
    Class implementing the ApplyToSelected module.

    The ApplyToSelected module applies the underlying module to the selected input from the given input list.

    Attributes:
        select_id (int): Integer containing the index of the selected input.
        module (nn.Module): Underlying module applied to the selected input.
    """

    def __init__(self, select_id, module_cfg):
        """
        Initializes the ApplyToSelected module.

        Args:
            select_id (int): Integer containing the index of the selected input.
            module_cfg (Dict): Configuration dictionary specifying the underlying module.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set select_id attribute
        self.select_id = select_id

        # Build underlying module
        self.module = build_model(module_cfg, sequential=True)

    def forward(self, in_list, **kwargs):
        """
        Forward method of the ApplyToSelected module.

        Args:
            in_list (List): Input list containing the selected input to be processed by the underlying module.
            kwargs (Dict): Dictionary of keyword arguments passed to the underlying module.

        Returns:
            out_list (List): Output list replacing the selected input by its processed output.
        """

        # Get output list
        out_list = in_list.copy()
        out_list[self.select_id] = self.module(in_list[self.select_id], **kwargs)

        return out_list


@MODELS.register_module()
class ApplyOneToOne(nn.Module):
    """
    Class implementing the ApplyOneToOne module.

    The ApplyOneToOne module applies one specific sub-module to each input from the input list.

    Attributes:
        sub_modules (nn.ModuleList): List [num_sub_modules] of sub-modules applied to inputs from input list.
    """

    def __init__(self, sub_module_cfgs):
        """
        Initializes the ApplyOneToOne module.

        Args:
            sub_module_cfgs (List): List [num_sub_modules] of configuration dictionaries specifying the sub-modules.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build sub-modules
        self.sub_modules = nn.ModuleList([build_model(cfg_i, sequential=True) for cfg_i in sub_module_cfgs])

    def forward(self, in_list, **kwargs):
        """
        Forward method of the ApplyOneToOne module.

        Args:
            in_list (List): List with inputs to be processed by the sub-module.
            kwargs (Dict): Dictionary of keyword arguments passed to each sub-module.

        Returns:
            out_list (List): List with resulting outputs of the sub-modules.
        """

        # Apply sub-modules to inputs from input list
        out_list = [module_i(in_i) for in_i, module_i in zip(in_list, self.sub_modules)]

        return out_list


@MODELS.register_module()
class BottomUp(nn.Module):
    """
    Class implementing the BottomUp module.

    Attributes:
        bu (nn.Module): Module computing the residual bottum-up features.
    """

    def __init__(self, bu_cfg):
        """
        Initializes the BottomUp module.

        Args:
            bu_cfg (Dict): Configuration dictionary specifying the residual bottom-up module.
        """

        # Iniialization of default nn.Module
        super().__init__()

        # Build residual bottom-up module
        self.bu = build_model(bu_cfg)

    def forward(self, in_feat_maps, **kwargs):
        """
        Forward method of the BottomUp module.

        Args:
            in_feat_maps (List): List of size [num_maps] with input feature maps.
            kwargs (Dict): Dictionary of keyword arguments passed to the residual bottom-up module.

        Returns:
            out_feat_maps (List): List of size [num_maps] with output feature maps.
        """

        # Get list with output feature maps
        num_maps = len(in_feat_maps)
        out_feat_list = [in_feat_maps[i+1] + self.bu(in_feat_maps[i], **kwargs) for i in range(num_maps-1)]

        return out_feat_list


@MODELS.register_module()
class GetApplyInsert(nn.Module):
    """
    Class implementing the GetApplyInsert module.

    The GetApplyInsert module applies its underlying module to the selected input from the given input list, and
    inserts the resulting output back into the given input list.

    Attributes:
        get_id (int): Index selecting the input for the underlying module from a given input list.
        module (nn.Module): Underlying module applied on the selected input from the given input list.
        insert_id (int): Index inserting the output of the underlying module into the given input list.
    """

    def __init__(self, module_cfg, get_id, insert_id):
        """
        Initializes the GetApplyInsert module.

        Args:
            module_cfg (Dict): Configuration dictionary specifying the underlying module.
            get_id (int): Index selecting the input for the underlying module from a given input list.
            insert_id (int): Index inserting the output of the underlying module into the given input list.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build underlying module
        self.module = build_model(module_cfg)

        # Set index attributes
        self.get_id = get_id
        self.insert_id = insert_id

    def forward(self, in_list, **kwargs):
        """
        Forward method of the GetApplyInsert module.

        Args:
            in_list (List): Input list of size [in_size] containing the input for the underlying module.
            kwargs (Dict): Dictionary of keyword arguments passed to the underlying module.

        Returns:
            in_list (List): Updated input list of size [in_size+1] additionally containing output of underlying module.
        """

        # Get input for underlying module from input list
        input = in_list[self.get_id]

        # Apply underlying module
        output = self.module(input, **kwargs)

        # Insert output from underying module into input list
        in_list.insert(self.insert_id, output)

        return in_list


@MODELS.register_module()
class GetBoxesTensor(nn.Module):
    """
    Class implementing the GetBoxesTensor module.

    Attributes:
        boxes_key (str): String with key to retrieve Boxes structure from storage dictionary (or None).
        clone (bool): Boolean indicating whether to clone Boxes structure.
        detach (bool): Boolean indicating whether to detach Boxes structure.
        format (str): String containing the desired bounding box format (or None).
        normalize (str): String containing the desired bounding box normalization (or None).
    """

    def __init__(self, boxes_key=None, clone=False, detach=False, format=None, normalize=None):
        """
        Initializes the GetBoxesTensor module.

        Args:
            boxes_key (str): String with key to retrieve Boxes structure from storage dictionary (default=None).
            clone (bool): Boolean indicating whether to clone Boxes structure (default=False).
            detach (bool): Boolean indicating whether to detach Boxes structure (default=False).
            format (str): String containing the desired bounding box format (default=None).
            normalize (str): String containing the desired bounding box normalization (default=None).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attributes
        self.boxes_key = boxes_key
        self.clone = clone
        self.detach = detach
        self.format = format
        self.normalize = normalize

    def forward(self, *args, storage_dict=None, **kwargs):
        """
        Forward method of the GetBoxesTensor module.

        Args:
            args (Tuple): Tuple with positional arguments (possibly) containing following items:
                - 0: Boxes structure with axis-aligned bounding boxes of size [num_boxes];
                - 1: Images structure containing the batched images of size [batch_size].

            storage_dict (Dict): Optional storage dictionary (possibly) containing following keys:
                - {self.boxes_key} (Boxes): Boxes structure with axis-aligned bounding boxes of size [num_boxes];
                - images (Images): Images structure containing the batched images of size [batch_size].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            boxes_tensor (FloatTensor): Tensor containing the bounding boxes of shape [num_boxes, 4].

        Raises:
            ValueError: Error when an invalid normalization string is provided.
        """

        # Get Boxes structure
        boxes = args[0] if self.boxes_key is None else storage_dict[self.boxes_key]

        # Transform Boxes structure if needed
        if self.clone:
            boxes = boxes.clone()

        if self.detach:
            boxes = boxes.detach()

        if self.format is not None:
            boxes = boxes.to_format(self.format)

        if self.normalize is not None:
            images = args[1] if self.boxes_key is None else storage_dict['images']

            if self.normalize == 'false':
                boxes = boxes.to_img_scale(images)

            elif self.normalize == 'with_padding':
                boxes = boxes.normalize(images, with_padding=True)

            elif self.normalize == 'without_padding':
                boxes = boxes.normalize(images, with_padding=False)

            else:
                error_msg = f"Invalid normalization string in GetBoxesTensor (got {self.normalize})."
                raise ValueError(error_msg)

        # Get tensor with bounding boxes
        boxes_tensor = boxes.boxes

        return boxes_tensor


@MODELS.register_module()
class IdAvg2d(nn.Module):
    """
    Class implementing the IdAvg2d module.
    """

    def __init__(self):
        """
        Initializes the IdAvg2d module.
        """

        # Initialization of default nn.Module
        super().__init__()

    def forward(self, core_feats, aux_feats, id_map, **kwargs):
        """
        Forward method of the IdAvg2d module.

        Args:
            core_feats (FloatTensor): Core features of shape [num_core_feats, feat_size].
            aux_feats (FloatTensor): Auxiliary features of shape [num_aux_feats, feat_size].
            id_map (LongTensor): Index map with feature indices of shape [num_rois, rH, rW].
            kwargs (Dict): Dictionary with unused keyword arguments.

        Returns:
            avg_feat (FloatTensor): Average feature of shape [1, feat_size].
        """

        # Get average feature
        num_core_feats = len(core_feats)
        num_aux_feats = len(aux_feats)
        num_feats = num_core_feats + num_aux_feats

        counts = torch.bincount(id_map.flatten(), minlength=num_feats)
        counts = counts[:, None]

        avg_feat = (counts[:num_core_feats] * core_feats).sum(dim=0, keepdim=True)
        avg_feat += (counts[num_core_feats:] * aux_feats).sum(dim=0, keepdim=True)
        avg_feat *= 1/id_map.numel()

        return avg_feat


@MODELS.register_module()
class QryInit(nn.Module):
    """
    Class implementing the QryInit module.

    Attributes:
        qry_init (nn.Module): Module implementing the query initialization module.
        rename_dict (Dict): Dictionary used to rename the keys from the original query dictionary.
        qry_ids_key (str): String containing the storage dictionary key to store the query indices (or None).
    """

    def __init__(self, qry_init_cfg, rename_dict=None, qry_ids_key=None):
        """
        Initializes the QryInit module.

        Args:
            qry_init_cfg (Dict): Configuration dictionary specifying the query initialization module.
            rename_dict (Dict): Dictionary used to rename the keys from the original query dictionary (default=None).
            qry_ids_key (str): String containing the storage dictionary key to store the query indices (default=None).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build query initialization module
        self.qry_init = build_model(qry_init_cfg)

        # Set remaining attributes
        self.rename_dict = rename_dict if rename_dict is not None else {}
        self.qry_ids_key = qry_ids_key

    def forward(self, _, storage_dict, **kwargs):
        """
        Forward method of the QryInit module.

        Args:
            storage_dict (Dict): Dictionary storing various items of interest.
            kwargs (Dict): Dictionary of keyword arguments passed to the query initialization module.

        Returns:
            out_qry_feats (FloatTensor): Output query features of shape [num_out_qrys, qry_feat_size].

        Raises:
            ValueError: Error when two objects of an unknown type need to be concatenated.
        """

        # Apply query initialization module
        qry_dict, storage_dict = self.qry_init(storage_dict=storage_dict, **kwargs)

        # Rename specific keys from query dictionary
        for old_key, new_key in self.rename_dict.items():
            qry_dict[new_key] = qry_dict.pop(old_key, None)

        # Add query indices to storage dictionary if needed
        if self.qry_ids_key is not None:
            num_qrys_before = len(storage_dict.get('qry_feats', []))
            num_qrys_after = num_qrys_before + len(qry_dict['qry_feats'])

            device = qry_dict['qry_feats'].device
            qry_ids = torch.arange(num_qrys_before, num_qrys_after, device=device)
            storage_dict[self.qry_ids_key] = qry_ids

        # Add elements from query dictionary to storage dictionary
        for k, v in qry_dict.items():
            if k in storage_dict.keys():

                if torch.is_tensor(v):
                    storage_dict[k] = torch.cat([storage_dict[k], v], dim=0)
                elif isinstance(v, Boxes):
                    storage_dict[k] = Boxes.cat([storage_dict[k], v])
                else:
                    error_msg = f"Cannot concatenate objects of unknown type '{type(v)}' for key '{k}'."
                    raise ValueError(error_msg)

            else:
                storage_dict[k] = v

        # Get output query features
        out_qry_feats = storage_dict['qry_feats']

        return out_qry_feats


@MODELS.register_module()
class SkipConnection(nn.Module):
    """
    Class implementing the SkipConnection module.

    Attributes:
        res (nn.Module): Module computing the residual features from the input features.
    """

    def __init__(self, res_cfg):
        """
        Initializes the SkipConnection module.

        Args:
            res_cfg (Dict): Configuration dictionary specifying the residual module.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build residual module
        self.res = build_model(res_cfg)

    def forward(self, in_feats, **kwargs):
        """
        Forward method of the SkipConnection module.

        Args:
            in_feats (FloatTensor): Input features of shape [num_feats, feat_size].
            kwargs (Dict): Dictionary of keyword arguments passed to the residual module.

        Returns:
            out_feats (FloatTensor): Output features of shape [num_feats, feat_size].
        """

        # Get output features
        out_feats = in_feats + self.res(in_feats, **kwargs)

        return out_feats


@MODELS.register_module()
class StorageCat(nn.Module):
    """
    Class implementing the StorageCat module.

    Attributes:
        keys_to_cat (str): List with keys of storage dictionary entries to concatenate.
        module (nn.Module): Underlying module computing the storage dictionary entries to be concatenated.
        cat_dim (int): Integer containing the dimension along which to concatenate.
    """

    def __init__(self, keys_to_cat, module_cfg, cat_dim=0):
        """
        Initializes the StorageCat module.

        Args:
            keys_to_cat (List): List with keys of storage dictionary entries to concatenate.
            module_cfg (Dict): Configuration dictionary specifying the underlying module.
            cat_dim (int): Integer containing the dimension along which to concatenate (default=0).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build underlying module
        self.module = build_model(module_cfg)

        # Set remaining attributes
        self.keys_to_cat = keys_to_cat
        self.cat_dim = cat_dim

    def forward(self, storage_dict, **kwargs):
        """
        Forward method of the StorageCat module.

        Args:
            storage_dict (Dict): Dictionary storing various items of interest.
            kwargs (Dict): Dictionary of keyword arguments passed to the underlying module.

        Returns:
            storage_dict (Dict): Storage dictionary with some entries updated by concatenation.
        """

        # Swap desired entries from dictionary
        cat_dict = {}

        for key in self.keys_to_cat:
            cat_dict[key] = storage_dict.pop(key)

        # Apply underlying module
        self.module(storage_dict=storage_dict, **kwargs)

        # Concatenate desired entries
        for key in self.keys_to_cat:
            storage_dict[key] = torch.cat([cat_dict[key], storage_dict[key]], dim=self.cat_dim)

        return storage_dict


@MODELS.register_module()
class StorageMasking(nn.Module):
    """
    Class implementing the StorageMasking module.

    Attributes:
        with_in_tensor (bool): Boolean indicating whether an input tensor is provided as positional argument.
        mask_key (str): String with key to retrieve mask from storage dictionary.
        mask_in_tensor (bool): Boolean indicating whether the input tensor should be masked.
        keys_to_mask (List): List with keys of storage dictionary entries to mask.

        ids_mask_dicts (List): List of dictionaries used to mask index tensors, each of them containing following keys:
            - ids_key (str): string with key of storage dictionary entry to build index-based mask from;
            - apply_keys (List): list with keys of storage dictionary entries to apply index-based mask on.

        module (nn.Module): Underlying module applied on the (potentially) masked inputs.
        keys_to_update (List): List with masked keys of storage dictionary entries to update.
    """

    def __init__(self, with_in_tensor, mask_key, module_cfg, mask_in_tensor=True, keys_to_mask=None,
                 ids_mask_dicts=None, keys_to_update=None, **kwargs):
        """
        Initializes the StorageMasking module.

        Args:
            with_in_tensor (bool): Boolean indicating whether an input tensor is provided as positional argument.
            mask_key (str): String with key to retrieve mask from storage dictionary.
            module_cfg (Dict): Configuration dictionary specifying the underlying module.
            mask_in_tensor (bool): Boolean indicating whether the input tensor should be masked (default=True).
            keys_to_mask (List): List with keys of storage dictionary entries to mask (default=None).
            ids_mask_dicts (List): List of dictionaries used to mask storage dictionary index tensors (default=None).
            keys_to_update (List): List with masked keys of storage dictionary entries to update (default=None).
            kwargs (Dict): Dictionary of keyword arguments passed to the build function of the underlying module.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build underlying module
        self.module = build_model(module_cfg, **kwargs)

        # Set remaining attributes
        self.with_in_tensor = with_in_tensor
        self.mask_key = mask_key
        self.mask_in_tensor = mask_in_tensor
        self.keys_to_mask = keys_to_mask if keys_to_mask is not None else []
        self.ids_mask_dicts = ids_mask_dicts if ids_mask_dicts is not None else []
        self.keys_to_update = keys_to_update if keys_to_update is not None else []

    def forward_with(self, in_tensor, storage_dict, **kwargs):
        """
        Forward method of the StorageMasking module with an input tensor provided as positional argument.

        Args:
            in_tensor (Tensor): Input tensor of arbitrary shape.
            storage_dict (Dict): Dictionary storing various items of interest.
            kwargs (Dict): Dictionary of keyword arguments passed to the underlying module.

        Returns:
            out_tensor (Tensor): Output tensor of arbitrary shape.
        """

        # Retrieve mask from storage dictionary
        mask = storage_dict[self.mask_key]

        # Get masked input tensor
        mask_in_tensor = in_tensor[mask] if self.mask_in_tensor else in_tensor

        # Mask desired entries from storage dictionary
        unmask_dict = {}
        ids_mask_list = [None for _ in range(len(self.ids_mask_dicts))]

        for key in self.keys_to_mask:
            if key in storage_dict:
                unmask_dict[key] = storage_dict[key]
                storage_dict[key] = storage_dict[key][mask]

        if len(self.ids_mask_dicts) > 0:
            if mask.dtype == torch.bool:
                mask_ids = mask.nonzero(as_tuple=False)[:, 0]
            else:
                mask_ids = mask

        for i, ids_mask_dict in enumerate(self.ids_mask_dicts):
            ids_key = ids_mask_dict['ids_key']

            if ids_key in storage_dict:
                ids_tensor = storage_dict[ids_key]

                ids_mask = (ids_tensor[:, None] == mask_ids[None, :]).any(dim=1)
                ids_mask_list[i] = ids_mask

                for key in ids_mask_dict['apply_keys']:
                    if key in storage_dict:
                        unmask_dict[key] = storage_dict[key]
                        storage_dict[key] = storage_dict[key][ids_mask]

        # Apply underlying module to get masked output tensor
        mask_out_tensor = self.module(mask_in_tensor, storage_dict=storage_dict, **kwargs)

        # Unmask desired entries from storage dictionary
        for key in self.keys_to_mask:
            if key in unmask_dict:
                unmask_value = unmask_dict[key]

                if key in self.keys_to_update:
                    unmask_value = unmask_value.clone()
                    unmask_value[mask] = storage_dict[key]

                storage_dict[key] = unmask_value

        for i, ids_mask_dict in enumerate(self.ids_mask_dicts):
            ids_mask = ids_mask_list[i]

            if ids_mask is not None:
                for key in ids_mask_dict['apply_keys']:
                    if key in unmask_dict:
                        unmask_value = unmask_dict[key]

                        if key in self.keys_to_update:
                            unmask_value = unmask_value.clone()
                            unmask_value[ids_mask] = storage_dict[key]

                        storage_dict[key] = unmask_value

        # Get output tensor
        if self.mask_in_tensor:
            out_tensor = in_tensor.clone()
            out_tensor[mask] = mask_out_tensor

        else:
            out_tensor = mask_out_tensor

        return out_tensor

    def forward_without(self, storage_dict, **kwargs):
        """
        Forward method of the StorageMasking module without an input tensor provided as positional argument.

        Args:
            storage_dict (Dict): Dictionary storing various items of interest.
            kwargs (Dict): Dictionary of keyword arguments passed to the underlying module.

        Returns:
            storage_dict (Dict): Storage dictionary with (potentially) some new and updated entries.
        """

        # Retrieve mask from storage dictionary
        mask = storage_dict[self.mask_key]

        # Mask desired entries from storage dictionary
        unmask_dict = {}
        ids_mask_list = [None for _ in range(len(self.ids_mask_dicts))]

        for key in self.keys_to_mask:
            if key in storage_dict:
                unmask_dict[key] = storage_dict[key]
                storage_dict[key] = storage_dict[key][mask]

        if len(self.ids_mask_dicts) > 0:
            if mask.dtype == torch.bool:
                mask_ids = mask.nonzero(as_tuple=False)[:, 0]
            else:
                mask_ids = mask

        for i, ids_mask_dict in enumerate(self.ids_mask_dicts):
            ids_key = ids_mask_dict['ids_key']

            if ids_key in storage_dict:
                ids_tensor = storage_dict[ids_key]

                ids_mask = (ids_tensor[:, None] == mask_ids[None, :]).any(dim=1)
                ids_mask_list[i] = ids_mask

                for key in ids_mask_dict['apply_keys']:
                    if key in storage_dict:
                        unmask_dict[key] = storage_dict[key]
                        storage_dict[key] = storage_dict[key][ids_mask]

        # Apply underlying module
        self.module(storage_dict=storage_dict, **kwargs)

        # Unmask desired entries from storage dictionary
        for key in self.keys_to_mask:
            if key in unmask_dict:
                unmask_value = unmask_dict[key]

                if key in self.keys_to_update:
                    unmask_value = unmask_value.clone()
                    unmask_value[mask] = storage_dict[key]

                storage_dict[key] = unmask_value

        for i, ids_mask_dict in enumerate(self.ids_mask_dicts):
            ids_mask = ids_mask_list[i]

            if ids_mask is not None:
                for key in ids_mask_dict['apply_keys']:
                    if key in unmask_dict:
                        unmask_value = unmask_dict[key]

                        if key in self.keys_to_update:
                            unmask_value = unmask_value.clone()
                            unmask_value[ids_mask] = storage_dict[key]

                        storage_dict[key] = unmask_value

        return storage_dict

    def forward(self, *args, **kwargs):
        """
        Forward method of the StorageMasking module.

        Args:
            args (Tuple): Tuple of positional arguments passed to the underlying forward method.
            kwargs (Dict): Dictionary of keyword arguments passed to the underlying forward method.

        Returns:
            * If 'self.with_in_tensor' is True:
                out_tensor (Tensor): Output tensor of arbitrary shape.

            * If 'self.with_in_tensor' is False:
                storage_dict (Dict): Storage dictionary with (potentially) some new and updated entries.
        """

        # Get and apply underlying forward method
        forward_method = self.forward_with if self.with_in_tensor else self.forward_without
        output = forward_method(*args, **kwargs)

        return output


@MODELS.register_module()
class TopDown(nn.Module):
    """
    Class implementing the TopDown module.

    Attributes:
        td (nn.Module): Module computing the residual top-down features.
    """

    def __init__(self, td_cfg):
        """
        Initializes the TopDown module.

        Args:
            td_cfg (Dict): Configuration dictionary specifying the residual top-down module.
        """

        # Iniialization of default nn.Module
        super().__init__()

        # Build residual top-down module
        self.td = build_model(td_cfg)

    def forward(self, in_feat_maps, **kwargs):
        """
        Forward method of the TopDown module.

        Args:
            in_feat_maps (List): List of size [num_maps] with input feature maps.
            kwargs (Dict): Dictionary of keyword arguments passed to the residual top-down module.

        Returns:
            out_feat_maps (List): List of size [num_maps] with output feature maps.
        """

        # Get list with output feature maps
        num_maps = len(in_feat_maps)
        out_feat_list = [in_feat_maps[i] + self.td(in_feat_maps[i+1], **kwargs) for i in range(num_maps-1)]

        return out_feat_list


@MODELS.register_module()
class View(nn.Module):
    """
    Class implementing the View module.

    Attributes:
        out_shape (Tuple): Tuple containing the output shape.
    """

    def __init__(self, out_shape):
        """
        Initializes the View module.

        Args:
            out_shape (Tuple): Tuple containing the output shape.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set output shape attribute
        self.out_shape = out_shape

    def forward(self, in_tensor, **kwargs):
        """
        Forward method of the View module.

        Args:
            in_tensor (Tensor): Input tensor of shape [*in_shape].
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            out_tensor (Tensor): Output tensor of shape [*out_shape].
        """

        # Get output tensor
        out_tensor = in_tensor.view(*self.out_shape)

        return out_tensor
