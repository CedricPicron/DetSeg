"""
Collection of point-related modules.
"""

from mmdet.models.utils.point_sample import get_uncertain_point_coords_with_randomness
import torch
from torch import nn

from models.build import MODELS


@MODELS.register_module()
class BoxToImgPts(nn.Module):
    """
    Class implementing the BoxToImgPts module.

    Attributes:
        in_key (str): String with key to retrieve box-normalized points from storage dictionary.
        boxes_key (str): String with key to retrieve corresponding boxes from storage dictionary.
        out_key (str): String with key to store image-normalized points in storage dictionary.
    """

    def __init__(self, in_key, boxes_key, out_key):
        """
        Initializes the BoxToImgPts module.

        Args:
            in_key (str): String with key to retrieve box-normalized points from storage dictionary.
            boxes_key (str): String with key to retrieve corresponding boxes from storage dictionary.
            out_key (str): String with key to store image-normalized points in storage dictionary.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set additional attributes
        self.in_key = in_key
        self.boxes_key = boxes_key
        self.out_key = out_key

    def forward(self, storage_dict, **kwargs):
        """
        Forward method of the BoxToImgPts module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - images (Images): Images structure containing the batched images of size [batch_size];
                - {self.in_key} (FloatTensor): box-normalized points of shape [num_boxes, num_pts, 2];
                - {self.boxes_key} (Boxes): 2D bounding boxes of size [num_boxes].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            storage_dict (dict): Storage dictionary containing following additional key:
                - {self.out_key}: image-normalized points of shape [num_boxes, num_pts, 2].
        """

        # Retrieve desired items from storage dictionary
        images = storage_dict['images']
        box_pts = storage_dict[self.in_key]
        boxes = storage_dict[self.boxes_key].clone()

        # Get image-normalized points
        boxes = boxes.to_format('xywh').normalize(images).boxes.detach()

        num_pts = box_pts.size()[1]
        boxes = boxes[:, None, :].expand(-1, num_pts, -1)

        box_xy = boxes[:, :, :2]
        box_wh = boxes[:, :, 2:]
        img_pts = box_xy + box_pts * box_wh

        # Store image-normalized points in storage dictionary
        storage_dict[self.out_key] = img_pts

        return storage_dict


@MODELS.register_module()
class GridPts2d(nn.Module):
    """
    Class implementing the GridPts2d module.

    Attributes:
        in_key (str): String with key to retrieve input map from storage dictionary.
        num_pts (int): Integer containing the number of output points.
        out_key (str): String with key to store normalized points in storage dictionary.
        out_ids_key (str): String with key to store grid indices in storage dictionary (or None).
    """

    def __init__(self, in_key, num_pts, out_key, out_ids_key=None):
        """
        Initializes the GridPts2d module.

        Args:
            in_key (str): String with key to retrieve input map from storage dictionary.
            num_pts (int): Integer containing the number of output points.
            out_key (str): String with key to store normalized points in storage dictionary.
            out_ids_key (str): String with key to store grid indices in storage dictionary (default=None).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set additional attributes
        self.in_key = in_key
        self.num_pts = num_pts
        self.out_key = out_key
        self.out_ids_key = out_ids_key

    def forward(self, storage_dict, **kwargs):
        """
        Forward method of the GridPts2d module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following key:
                - {self.in_key} (FloatTensor): input map to sample from of shape [num_groups, {1}, mH, mW].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            storage_dict (Dict): Storage dictionary possibly containing following additional keys:
                - {self.out_ids_key} (LongTensor): 2D-flattened grid indices of shape [num_groups, num_pts];
                - {self.out_key} (FloatTensor): normalized points of shape [num_groups, num_pts, 2].
        """

        # Retrieve input map from storage dictionary
        in_map = storage_dict[self.in_key]

        # Reshape input map
        mH, mW = in_map.size()[-2:]
        in_map = in_map.view(-1, mH*mW)

        # Get grid indices
        grid_ids = in_map.topk(self.num_pts, dim=1)[1]

        # Store grid indices in storage dictionary if needed
        if self.out_ids_key is not None:
            storage_dict[self.out_ids_key] = grid_ids

        # Get normalized points
        pts_xy = torch.stack([grid_ids % mW, grid_ids // mW], dim=2)
        pts_xy = pts_xy / torch.tensor([mW, mH], device=in_map.device)[None, None, :]

        # Store points in storage dictionary
        storage_dict[self.out_key] = pts_xy

        return storage_dict


@MODELS.register_module()
class IdsToPts2d(nn.Module):
    """
    Class implementing the IdsToPts2d module.

    Attributes:
        in_key (str): String with key to retrieve input grid indices from storage dictionary.
        size_key (str): String with key to retrieve grid size tensor from storage dictionary.
        out_key (str): String with key to store output points in storage dictionary.
    """

    def __init__(self, in_key, size_key, out_key):
        """
        Initializes the IdsToPts2d module.

        Args:
            in_key (str): String with key to retrieve input grid indices from storage dictionary.
            size_key (str): String with key to retrieve grid size tensor from storage dictionary.
            out_key (str): String with key to store output points in storage dictionary.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set additional attributes
        self.in_key = in_key
        self.size_key = size_key
        self.out_key = out_key

    def forward(self, storage_dict, **kwargs):
        """
        Forward method of the IdsToPts2d module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - {self.in_key} (LongTensor): input grid indices of shape [*, 2];
                - {self.size_key} (Tensor): tensor from which to infer grid size of shape [*, mH, mW].

            kwargs (Dict): Dictionary of keyword arguments.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional key:
                - {self.out_key} (FloatTensor): normalized output points of shape [*, 2].
        """

        # Retrieve desired items from storage dictionary
        in_ids = storage_dict[self.in_key]
        size_tensor = storage_dict[self.size_key]

        # Get normalized output points
        mH, mW = size_tensor.size()[-2:]
        out_pts = in_ids / torch.tensor([mH, mW], device=in_ids.device)

        # Store output points in storage dictionary
        storage_dict[self.out_key] = out_pts

        return storage_dict


@MODELS.register_module()
class PointRendTrainPts2d(nn.Module):
    """
    Class implementing the PointRendTrainPts2d module.

    Attributes:
        in_key (str): String with key to retrieve input map from storage dictionary.
        labels_key (str): String with key to retrieve group labels from storage dictionary.
        out_key (str): String with key to store normalized points in storage dictionary.
        num_pts (int): Integer containing the number of output points.
        oversample_ratio (float): Value containing the oversample ratio.
        importance_ratio (float): Value containing the importance sampling ratio.
    """

    def __init__(self, in_key, labels_key, out_key, num_pts, oversample_ratio=3.0, importance_ratio=0.75):
        """
        Initializes the PointRendTrainPts2d module.

        Args:
            in_key (str): String with key to retrieve input map from storage dictionary.
            labels_key (str): String with key to retrieve group labels from storage dictionary.
            out_key (str): String with key to store normalized points in storage dictionary.
            num_pts (int): Integer containing the number of output points.
            oversample_ratio (float): Value containing the oversample ratio (default=3.0).
            importance_ratio (float): Value containing the importance sampling ratio (default=0.75).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set additional attributes
        self.in_key = in_key
        self.labels_key = labels_key
        self.num_pts = num_pts
        self.out_key = out_key
        self.oversample_ratio = oversample_ratio
        self.importance_ratio = importance_ratio

    def forward(self, storage_dict, **kwargs):
        """
        Forward method of the PointRendTrainPts2d module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - {self.in_key} (FloatTensor): input map to sample from of shape [num_groups, *, mH, mW];
                - {self.labels_key} (LongTensor): tensor containing the group labels of shape [num_groups].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional key:
                - {self.out_key} (FloatTensor): normalized points of shape [num_groups, num_pts, 2].
        """

        # Retrieve desired items from storage dictionary
        in_map = storage_dict[self.in_key]
        labels = storage_dict[self.labels_key]

        # Get input map in desired format
        if in_map.dim() == 3:
            in_map = in_map.unsqueeze(dim=1)

        # Get normalized points
        pts_kwargs = {'num_points': self.num_pts, 'oversample_ratio': self.oversample_ratio}
        pts_kwargs = {**pts_kwargs, 'importance_sample_ratio': self.importance_ratio}
        pts_xy = get_uncertain_point_coords_with_randomness(in_map, labels, **pts_kwargs)

        # Store points in storage dictionary
        storage_dict[self.out_key] = pts_xy

        return storage_dict
