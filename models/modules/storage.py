"""
Collection of storage-related modules.
"""

import torch
from torch import nn

from models.build import build_model, MODELS


@MODELS.register_module()
class StorageApply(nn.Module):
    """
    Class implementing the StorageApply module.

    Attributes:
        in_key (str): String with key to retrieve input from storage dictionary.
        module (nn.Module): Underlying module applied on the retrieved input.
        out_key (str): String with key to store output in storage dictionary.
        storage_kwargs (Dict): Dictionary selecting keyword arguments from storage dictionary.
        filter_kwargs (List): List of filterd keyword argument keys passed to underlying module (or None).
    """

    def __init__(self, in_key, module_cfg, out_key, storage_kwargs=None, filter_kwargs=None):
        """
        Initializes the StorageApply module.

        Args:
            in_key (str): String with key to retrieve input from storage dictionary.
            module_cfg (Dict): Configuration dictionary specifying the underlying module.
            out_key (str): String with key to store output in storage dictionary.
            storage_kwargs (Dict): Dictionary selecting keyword arguments from storage dictionary (default=None).
            filter_kwargs (List): List of filterd keyword argument keys passed to underlying module (default=None).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build underlying module
        self.module = build_model(module_cfg, sequential=True)

        # Set attributes
        self.in_key = in_key
        self.out_key = out_key
        self.storage_kwargs = storage_kwargs if storage_kwargs is not None else {}
        self.filter_kwargs = filter_kwargs

    def forward(self, storage_dict, **kwargs):
        """
        Forward method of the StorageApply module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following key:
                - {self.in_key} (Any): input on which to apply the underlying module.

            kwargs (Dict): Dictionary of keyword arguments passed to the underlying module.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional key:
                - {self.out_key} (Any): output from the underlying module.
        """

        # Retrieve input from storage dictionary
        input = storage_dict[self.in_key]

        # Get additional keyword arguments from storage dictionary
        storage_kwargs = {v: storage_dict[k] for k, v in self.storage_kwargs.items()}

        # Filter keyword arguments if needed
        if self.filter_kwargs is not None:
            kwargs = {k: kwargs[k] for k in self.filter_kwargs}

        # Apply underlying module
        output = self.module(input, **storage_kwargs, **kwargs)

        # Store output in storage dictionary
        storage_dict[self.out_key] = output

        return storage_dict


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
class StorageCondition(nn.Module):
    """
    Class implementing the StorageCondition module.

    Attributes:
        cond_key (str): String with key to retrieve condition boolean from storage dictionary.
        module (nn.Module): Underlying module only applied if the retrieved condition is true.
    """

    def __init__(self, cond_key, module_cfg):
        """
        Initializes the StorageCondition module.

        Args:
            cond_key (str): String with key to retrieve condition boolean from storage dictionary.
            module_cfg (Dict): Configuration dictionary specifying the underlying module.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build underlying module
        self.module = build_model(module_cfg, sequential=True)

        # Set additional attribute
        self.cond_key = cond_key

    def forward(self, storage_dict, **kwargs):
        """
        Forward method of the StorageCondition module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following key:
                {self.cond_key} (bool): boolean condition determining whether underlying module will be applied.

            kwargs (Dict): Dictionary of keyword arguments passed to the underlying module.

        Returns:
            storage_dict (Dict): Storage dictionary possibly containing additional or updated keys.
        """

        # Retrieve condition from storage dictionary
        condition = storage_dict.get(self.cond_key, False)

        # Apply underlying module if condition is true
        if condition:
            storage_dict = self.module(storage_dict, **kwargs)

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
