"""
Collection of utility modules.
"""

from torch import nn

from models.build import build_model, MODELS


@MODELS.register_module()
class GetApplyInsert(nn.Module):
    """
    Class implementing the GetApplyInsert module.

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