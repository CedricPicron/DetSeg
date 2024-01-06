"""
Collection of utility modules.
"""

import torch
from torch import nn
from torch.nn.parameter import Parameter

from models.build import build_model, MODELS
from models.functional.utils import maps_to_seq, seq_to_maps
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
class GetItemStorage(nn.Module):
    """
    Class implementing the GetItemStorage module.

    Attributes:
        in_key (str): String with key to retrieve input from storage dictionary.
        index_key (str): String with key to retrieve index from storage dictionary.
        out_key (str): String with key to store output in storage dictionary.
    """

    def __init__(self, in_key, index_key, out_key):
        """
        Initializes the GetItemStorage module.

        Args:
            in_key (str): String with key to retrieve input from storage dictionary.
            index_key (str): String with key to retrieve index from storage dictionary.
            out_key (str): String with key to store output in storage dictionary.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set index attribute
        self.in_key = in_key
        self.index_key = index_key
        self.out_key = out_key

    def forward(self, storage_dict, **kwargs):
        """
        Forward method of the GetItemStorage module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - {self.in_key} (Any): input to select items from;
                - {self.index_key} (Any): index indicating items to be selected.

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional key:
                - {self.out_key} (Any): output with selected items from input.
        """

        # Retrieve desired items from storage dictionary
        input = storage_dict[self.in_key]
        index = storage_dict[self.index_key]

        # Get output with selected items
        output = input[index]

        # Store output in storage dictionary
        storage_dict[self.out_key] = output

        return storage_dict


@MODELS.register_module()
class GetItemTensor(nn.Module):
    """
    Class implementing the GetItemTensor module.

    Attributes:
        index (Any): Object selecting the desired items from the input.
    """

    def __init__(self, index):
        """
        Initializes the GetItemTensor module.

        Args:
            index (any): Object selecting the desired items from the input.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set index attribute
        self.index = index

    def forward(self, input, **kwargs):
        """
        Forward method of the GetItemTensor module.

        Args:
            input (Any): Input to select the desired items from.
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            output (Any): Output containing the selected items.
        """

        # Get output with selected items
        output = input[self.index]

        return output


@MODELS.register_module()
class GetPosFromBoxes(nn.Module):
    """
    Class implementing the GetPosFromBoxes module.

    Attributes:
        boxes_key (str): String with key to retrieve boxes from storage dictionary.
        pos_module_key (str): String with key to retrieve position module from storage dictionary.
        box_mask_key (str): String with key to retrieve box mask from storage dictionary (or None).
        non_box_pos_feats (Parameter): Optional parameter with non-box position features of shape [pos_feat_size].
        pos_feats_key (str): String with key to save position features in storage dictionary.
    """

    def __init__(self, boxes_key, pos_module_key, pos_feats_key, box_mask_key=None, pos_feat_size=None):
        """
        Initializes the GetPosFromBoxes module.

        Args:
            boxes_key (str): String with key to retrieve boxes from storage dictionary.
            pos_module_key (str): String with key to retrieve position module from storage dictionary.
            pos_feats_key (str): String with key to save position features in storage dictionary.
            box_mask_key (str): String with key to retrieve box mask from storage dictionary (default=None).
            pos_feat_size (int): Integer containing the position feature size (default=None).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Initialize non-box postion features if needed
        if box_mask_key is not None:
            self.non_box_pos_feats = Parameter(torch.empty(pos_feat_size), requires_grad=True)
            nn.init.normal_(self.non_box_pos_feats)

        # Set remaining attributes
        self.boxes_key = boxes_key
        self.pos_module_key = pos_module_key
        self.box_mask_key = box_mask_key
        self.pos_feats_key = pos_feats_key

    def forward(self, feats, storage_dict, **kwargs):
        """
        Forward method of the GetPosFromBoxes module.

        Args:
            feats (FloatTensor): Input features of shape [num_feats, feat_size].

            storage_dict (Dict): Storage dictionary containing at least following keys:
                - images (Images): Images structure containing the batched images of size [batch_size];
                - {self.boxes_key} (Boxes): boxes from which to infer position coordinates of size [num_feats];
                - {self.pos_module} (nn.Module): module computing position features from position coordinates.

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            feats (FloatTensor): Unchanged input features of shape [num_feats, feat_size].
        """

        # Get position coordinates
        boxes = storage_dict[self.boxes_key]
        boxes = boxes.clone().normalize(storage_dict['images']).to_format('cxcywh')

        pos_xy = boxes.boxes[:, :2]
        pos_wh = boxes.boxes[:, 2:]

        # Get box position features
        pos_module = storage_dict[self.pos_module_key]
        pos_feats = pos_module(pos_xy, pos_wh)

        # Add non-box position features if needed
        if self.box_mask_key is not None:
            box_pos_feats = pos_feats
            pos_feats = self.non_box_pos_feats.repeat(len(feats), 1)

            box_mask = storage_dict[self.box_mask_key]
            pos_feats[box_mask] = box_pos_feats

        # Save position features
        storage_dict[self.pos_feats_key] = pos_feats

        return feats


@MODELS.register_module()
class GetPosFromMaps(nn.Module):
    """
    Class implementing the GetPosFromMaps module.

    Attributes:
        pos_module_key (str): String with key to retrieve position module from storage dictionary.
        pos_feats_key (str): String with key to save position features in storage dictionary.
        pos_maps_key (str): String with key to save position feature maps in storage dictionary.
    """

    def __init__(self, pos_module_key, pos_feats_key=None, pos_maps_key=None):
        """
        Initializes the GetPosFromMaps module.

        Args:
            pos_module_key (str): String with key to retrieve position module from storage dictionary.
            pos_feats_key (str): String with key to save position features in storage dictionary (default=None).
            pos_maps_key (str): String with key to save position feature maps in storage dictionary (default=None).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set remaining attributes
        self.pos_module_key = pos_module_key
        self.pos_feats_key = pos_feats_key
        self.pos_maps_key = pos_maps_key

    def forward(self, feat_maps, storage_dict, **kwargs):
        """
        Forward method of the GetPosFromMaps module.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

            storage_dict (Dict): Storage_dictionary (possibly) containing following keys:
                - map_shapes (LongTensor): feature map shapes in (H, W) format of shape [num_maps, 2];
                - {self.pos_module} (nn.Module): module computing position features from position coordinates.

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].
        """

        # Get batch size and device
        batch_size = len(feat_maps[0])
        device = feat_maps[0].device

        # Add map shapes to storage dictionary if needed
        if 'map_shapes' not in storage_dict:
            map_shapes = [feat_map.shape[-2:] for feat_map in feat_maps]
            storage_dict['map_shapes'] = torch.tensor(map_shapes, device=device)

        # Get position coordinates
        pos_xy_maps = []
        pos_wh_maps = []

        for feat_map in feat_maps:
            fH, fW = feat_map.size()[-2:]

            pos_x = 0.5 + torch.arange(fW, device=device)[None, :].expand(fH, -1)
            pos_y = 0.5 + torch.arange(fH, device=device)[:, None].expand(-1, fW)

            pos_x = pos_x / fW
            pos_y = pos_y / fH

            pos_xy_map = torch.stack([pos_x, pos_y], dim=0)
            pos_xy_map = pos_xy_map[None, :, :, :].expand(batch_size, -1, -1, -1)
            pos_xy_maps.append(pos_xy_map)

            pos_wh_map = torch.tensor([1/fW, 1/fH], device=device)
            pos_wh_map = pos_wh_map[None, :, None, None].expand(batch_size, -1, fH, fW)
            pos_wh_maps.append(pos_wh_map)

        pos_xy = maps_to_seq(pos_xy_maps).view(-1, 2)
        pos_wh = maps_to_seq(pos_wh_maps).view(-1, 2)

        # Get position features
        pos_module = storage_dict[self.pos_module_key]
        pos_feats = pos_module(pos_xy, pos_wh)

        feat_size = pos_feats.size(dim=1)
        pos_feats = pos_feats.view(batch_size, -1, feat_size)

        # Save position features if needed
        if self.pos_feats_key is not None:
            storage_dict[self.pos_feats_key] = pos_feats

        # Get and save position feature maps if needed
        if self.pos_maps_key is not None:
            pos_feat_maps = seq_to_maps(pos_feats, storage_dict['map_shapes'])
            storage_dict[self.pos_module_key] = pos_feat_maps

        return feat_maps


@MODELS.register_module()
class IdsToMask(nn.Module):
    """
    Class implementing the IdsToMask module.

    Attributes:
        in_key (str): String with key to retrieve input indices from storage dictionary.
        size_key (str): String with key to retrieve tensor with size information from storage dictionary.
        out_key (str): String with key to store output mask in storage dictionary.
    """

    def __init__(self, in_key, size_key, out_key):
        """
        Initializes the IdsToMask module.

        Args:
            in_key (str): String with key to retrieve input indices from storage dictionary.
            size_key (str): String with key to retrieve tensor with size information from storage dictionary.
            out_key (str): String with key to store output mask in storage dictionary.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set additional attributes
        self.in_key = in_key
        self.size_key = size_key
        self.out_key = out_key

    def forward(self, storage_dict, **kwargs):
        """
        Forward method of the IdsToMask module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - {self.in_key} (LongTensor): input tensor containing the indices of True elements;
                - {self.size_key} (Tensor): tensor from which to infer the mask size.

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional key:
                - {self.out_key} (Tensor): output mask with True elements at locations of input indices.
        """

        # Retrieve desired items from storage dictionary
        in_ids = storage_dict[self.in_key]
        size_tensor = storage_dict[self.size_key]

        # Get output mask
        mask_size = size_tensor.size(dim=0)
        out_mask = torch.zeros(mask_size, dtype=torch.bool, device=in_ids.device)
        out_mask[in_ids] = True

        # Store output mask in storage dictionary
        storage_dict[self.out_key] = out_mask

        return storage_dict


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
class SelectClass(nn.Module):
    """
    Class implementing the SelectClass module.

    Attributes:
        in_key (str): String with key to retrieve input map from storage dictionary.
        labels_key (str): String with key to retrieve class labels from storage dictionary.
        out_key (str): String with key to store output map in storage dictionary.
    """

    def __init__(self, in_key, labels_key, out_key):
        """
        Initializes the SelectClass module.

        Args:
            in_key (str): String with key to retrieve input map from storage dictionary.
            labels_key (str): String with key to retrieve class labels from storage dictionary.
            out_key (str): String with key to store output map in storage dictionary.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set additional attributes
        self.in_key = in_key
        self.labels_key = labels_key
        self.out_key = out_key

    def forward(self, storage_dict, **kwargs):
        """
        Forward method of the SelectClass module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - {self.in_key} (FloatTensor): input feature map of shape [batch_size, num_classes, fH, fW];
                - {self.labels_key} (LongTensor): class labels of shape [batch_size].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional key:
                - {self.out_key} (FloatTensor): output feature map of shape [batch_size, fH, fW].
        """

        # Retrieve desired items from storage dictionary
        in_feat_map = storage_dict[self.in_key]
        labels = storage_dict[self.labels_key]

        # Get output feature map
        out_feat_map = in_feat_map[range(len(labels)), labels]

        # Store output feature map in storage dictionary
        storage_dict[self.out_key] = out_feat_map

        return storage_dict


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
class Uncertainty(nn.Module):
    """
    Class implementing the Uncertainty module.

    Attributes:
        in_key (str): String with key to retrieve input logits from storage dictionary.
        out_key (str): String with key to store output uncertainty values in storage dictionary.
    """

    def __init__(self, in_key, out_key):
        """
        Initializes the Uncertainty module.

        Args:
            in_key (str): String with key to retrieve input logits from storage dictionary.
            out_key (str): String with key to store output uncertainty values in storage dictionary.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set additional attributes
        self.in_key = in_key
        self.out_key = out_key

    def forward(self, storage_dict, **kwargs):
        """
        Forward method of the Uncertainty module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following key:
                - {self.in_key} (FloatTensor): input logits of shape [*].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional key:
                - {self.out_key} (FloatTensor): output uncertainty values of shape [*].
        """

        # Retrieve input logits from storage dictionary
        logits = storage_dict[self.in_key]

        # Get uncertainty values
        unc_vals = -logits.abs()

        # Store uncertainty values in storage dictionary
        storage_dict[self.out_key] = unc_vals

        return storage_dict
