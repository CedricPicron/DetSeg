"""
GC module and build function.
"""

from pathlib import Path

from torch import nn
from yaml import safe_load

from .operations import initialize_operation


class GC(nn.Module):
    """
    Class implementing the GC (Generalized Core) module.

    Attributes:
        modules_list (nn.ModuleList): List with modules used by the GC operations.
        operations (List): List with operations transforming input feature maps into output feature maps.
        out_map_indices (List): List of size [num_out_maps] with indices selecting the output maps.
        feat_sizes (List): List of size [num_out_maps] containing the feature size of each output map.
    """

    def __init__(self, yaml_file):
        """
        Initializes the GC module.

        Args:
            yaml_file (str): String containing the path of the yaml-file with the GC specification.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Convert yaml-file string to Path object
        yaml_file = Path(yaml_file)

        # Get GC specification dictionary from yaml-file
        with open(yaml_file, 'r') as stream:
            gc_dict = safe_load(stream)

        # Get expected feature sizes from input maps and get layers
        feat_sizes = gc_dict['feat_sizes']
        layers = gc_dict.setdefault('layers', [])

        # Initialize modules and operations attributes
        self.modules_list = nn.ModuleList([])
        self.operations = []

        # Register every operation with their corresponding modules
        for operation in gc_dict['operations']:
            modules_offset = len(self.modules_list)
            modules, sub_operations, feat_sizes = initialize_operation(operation, feat_sizes, layers, modules_offset)
            self.modules_list.extend(modules)
            self.operations.extend(sub_operations)

        # Set output map indices and feature sizes attributes
        self.out_map_ids = gc_dict['outputs']
        self.feat_sizes = [feat_sizes[map_id] for map_id in self.out_map_ids]

    def forward(self, in_feat_maps, **kwargs):
        """
        Forward method of the GC module.

        Args:
            in_feat_maps (List): Input feature maps [num_in_maps] of shape [batch_size, feat_size, fH, fW].

        Returns:
            out_feat_maps (List): Output feature maps [num_out_maps] of shape [batch_size, feat_size, fH, fW].

        Raises:
            ValueError: Error for operation dictionary without key from {'filter_ids', 'module_id', 'function'}.
        """

        # Initialize list of feature maps
        feat_maps = in_feat_maps.copy()

        # Perform every operation
        for operation in self.operations:
            if 'filter_ids' in operation:
                feat_maps = [feat_maps[i] for i in operation['filter_ids']]
                continue

            elif 'module_id' in operation:
                function = self.modules_list[operation['module_id']]

            elif 'function' in operation:
                function = operation['function']

            else:
                required_keys = {'filter_ids', 'module_id', 'function'}
                raise ValueError(f"Operation dictonary should contain a key from {required_keys}.")

            inputs = [feat_maps[i] for i in operation['in_map_ids']]
            kwargs = {k: v for k, v in operation.items() if k not in ['module_id', 'function', 'in_map_ids']}
            outputs = function(*inputs, **kwargs)

            if isinstance(outputs, tuple):
                feat_maps.extend(outputs)
            else:
                feat_maps.append(outputs)

        # Get output feature maps
        out_feat_maps = [feat_maps[map_id] for map_id in self.out_map_ids]

        return out_feat_maps
