"""
Collection of utility modules.
"""

from torch import nn

from models.build import build_model, MODELS


@MODELS.register_module()
class ApplyAll(nn.Module):
    """
    Class implementing the ApplyAll module.

    The ApplyAll module applies its underlying module to all inputs from the given input list.

    Attributes:
        module (Sequential): Underlying module applied to all inputs from the input list.
    """

    def __init__(self, module_cfg):
        """
        Initializes the ApplyAll module.

        Args:
            module_cfg (Dict): Configuration dictionary specifying the underlying module.
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

    The ApplyAll module applies its underlying module to the selected input from the given input list, and inserts the
    resulting output back into the given input list.

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
