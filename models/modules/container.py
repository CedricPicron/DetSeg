"""
Collection of container-type modules.
"""
from copy import deepcopy
from inspect import signature

from torch import nn

from models.build import build_model, MODELS


@MODELS.register_module()
class Sequential(nn.Sequential):
    """
    Class implementing the enhanced Sequential module.

    It adds following two features to the forward method:
        1) It allows the use of keyword arguments to be passed to the sub-modules depending on their forward signature.
        2) It adds the possibility to return all intermediate outputs from each of the sub-modules.
    """

    def forward(self, input, return_intermediate=False, **kwargs):
        """
        Forward method of the enhanced Sequential module.

        Args:
            input (Any): Input of the forward method.
            return_intermediate (bool): Whether intermediate outputs should be returned (default=False).
            kwargs (Dict): Dictionary of keyword arguments to be passed to each of the sub-modules.

        Returns:
            * If return_intermediate is False:
                output (Any): Output from the final sub-module.

            * If return_intermediate is True:
                output (List): List of size [num_sub_modules] containing the outputs from each of the sub-modules.
        """

        # Initialize empty list of outputs
        output = []

        # Iterate over all sub-modules
        for module in self:
            module_kwargs = {name: kwargs[name] for name in signature(module.forward).parameters if name in kwargs}
            input = module(input, **module_kwargs)
            output.append(input)

        # Select output from final sub-module if requested
        if not return_intermediate:
            output = output[-1]

        return output


@MODELS.register_module()
class Stage(nn.Module):
    """
    Class implementing the Stage module.

    This module implements a sequence of block modules, with the possibility to return intermediate block outputs and
    with support for input and output projection modules.

    Attributes:
        in_proj (nn.Module): Optional module implementing the input projection module (None when missing).
        blocks (Sequential): Sequence of block modules.
        out_proj (nn.Module): Optional module implementing the output projection module (None when missing).
        return_inter_blocks (bool): Boolean indicating whether intermediate block outputs should be returned.
    """

    def __init__(self, block_cfg, num_blocks=1, return_inter_blocks=False, in_proj_cfg=None, out_proj_cfg=None):
        """
        Initializes the Stage module.

        Args:
            block_cfg (Dict): Configuration dictionary specifying a single block module.
            num_blocks (int): Integer containing the number of consecutive block modules (default=1).
            return_inter_blocks (bool): Whether intermediate block outputs should be returned (default=False).
            in_proj_cfg (Dict): Configuration dictionary specifying an input projection module (default=None).
            out_proj_cfg (Dict): Configuration dictionary specifying an output projection module (default=None).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build sub-modules
        self.in_proj = build_model(in_proj_cfg) if in_proj_cfg is not None else None
        self.blocks = Sequential(*[build_model(block_cfg) for _ in range(num_blocks)])
        self.out_proj = build_model(out_proj_cfg) if out_proj_cfg is not None else None

        # Set return_inter_blocks attribute
        self.return_inter_blocks = return_inter_blocks

    def forward(self, input):
        """
        Forward method of the Stage module.

        Args:
            input (Any): Input of the forward method.

        Returns:
            output (Any) Output of the forward method.
        """

        # Apply input projection if needed
        if self.in_proj is not None:
            input = self.in_proj(input)

        # Apply sequence of block modules
        output = self.blocks(input, return_intermediate=self.return_inter_blocks)

        # Apply output projection if needed
        if self.out_proj is not None:
            if self.return_inter_blocks:
                output[-1] = self.out_proj(output[-1])
            else:
                output = self.out_proj(output)

        return output


@MODELS.register_module()
class Net(nn.Module):
    """
    Class implementing the Net module.

    This module implements a sequence of stage modules, which itself consists of a sequence of block modules.
    The module allows each stage to have a different amount of blocks, and different feature sizes (i.e. a multiple of
    the base feature sizes). The module provides the possibility to return intermediate stage (and block) outputs.

    Attributes:
        stages (nn.ModuleList): List [num_stages] of stage modules.
        return_inter_stages (bool): Boolean indicating whether intermediate stage outputs should be returned.
        return_inter_blocks (bool): Boolean indicating whether intermediate block outputs should be returned.
    """

    def __init__(self, base_stage_cfg, blocks_per_stage, scale_factors, scale_tags, scale_overwrites=None,
                 return_inter_stages=False):
        """
        Initializes the Net module.

        Args:
            base_stage_cfg (Dict): Configuration dictionary specifying the base stage module.
            blocks_per_stage (List): List [num_stages] containing the number of blocks per stage.
            scale_factors (List): List [num_stages] containing the factors scaling the base stage feature sizes.
            scale_tags (List): List [num_tags] of strings tagging keys from base stage configuration to be scaled.
            scale_overwrites (List): List [num_overwrites] of lists with stage-specific overwrites (default=None).
            return_inter_stages (bool): Whether intermediate stage outputs should be returned (default=False).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build stage modules
        num_stages = len(blocks_per_stage)
        return_inter_blocks = base_stage_cfg.pop('return_inter_blocks', False)
        scale_overwrites = [] if scale_overwrites is None else scale_overwrites
        stages = []

        for stage_id in range(num_stages):
            num_blocks = blocks_per_stage[stage_id]
            scale_factor = scale_factors[stage_id]

            def cfg_scaling(cfg, scale, tags):
                if isinstance(cfg, dict):
                    for k, v in cfg.items():
                        if isinstance(v, (dict, list)):
                            cfg_scaling(v, scale, tags)
                            continue

                        for tag in tags:
                            if tag in k:
                                cfg[k] = scale * v

                elif isinstance(cfg, list):
                    for cfg_i in cfg:
                        cfg_scaling(cfg_i, scale, tags)

            stage_cfg = deepcopy(base_stage_cfg)
            cfg_scaling(stage_cfg, scale_factor, scale_tags)

            for overwrite in scale_overwrites:
                if overwrite[0] == stage_id:
                    item = stage_cfg

                    for key in overwrite[1:-2]:
                        item = item[key]

                    key, value = overwrite[-2:]
                    item[key] = value

            block_cfg = stage_cfg.pop('block_cfg')
            in_proj_cfg = stage_cfg.pop('in_proj_cfg', None)
            out_proj_cfg = stage_cfg.pop('out_proj_cfg', None)

            stage = Stage(block_cfg, num_blocks, return_inter_blocks, in_proj_cfg, out_proj_cfg)
            stages.append(stage)

        self.stages = nn.ModuleList(stages)

        # Set additional attributes
        self.return_inter_stages = return_inter_stages
        self.return_inter_blocks = return_inter_blocks

    def forward(self, input):
        """
        Forward method of the Net module.

        Args:
            input (Any): Input of the forward method.

        Returns:
            output (Any) Output of the forward method.
        """

        # Apply sequence of stage modules
        if self.return_inter_stages:
            output = []

        for stage in self.stages:
            stage_output = stage(input)

            if self.return_inter_blocks:
                input = stage_output[-1]
            else:
                input = stage_output

            if self.return_inter_stages:
                if self.return_inter_blocks:
                    output.extend(stage_output)
                else:
                    output.append(stage_output)

        if not self.return_inter_stages:
            output = stage_output

        return output
