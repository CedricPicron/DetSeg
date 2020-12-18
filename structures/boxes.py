"""
Boxes structure and bounding box utilities.
"""

import torch


class Boxes(object):
    """
    Class implementing the Boxes structure.

    Attributes:
        boxes (FloatTensor): Tensor of axis-aligned bounding boxes of shape [num_boxes, 4].
        format (str): String containing the format in which the bounding boxes are expressed.
        normalized (bool): Boolean indicating whether boxes are normalized or not.
    """

    def __init__(self, boxes, format, normalized=False):
        """
        Initializes the Boxes structure.

        Args:
            boxes (FloatTensor): Tensor of axis-aligned bounding boxes of shape [num_boxes, 4].
            format (str): String containing the format in which the bounding boxes are expressed.
            normalized (bool): Boolean indicating whether boxes are normalized or not (defaults to False).
        """

        # Check whether input boxes tensor has valid shape
        check = boxes.dim() == 2 and boxes.shape[-1] == 4
        assert_msg = f"Tensor has incorrect shape {boxes.shape} to become Boxes structure."
        assert check, assert_msg

        # Set attributes as specified by input arguments
        self.boxes = boxes
        self.format = format
        self.normalized = normalized

    def __getitem__(self, item):
        """
        Implements the __getitem__ method of the Boxes structure.

        Args:
            item: We support three possibilities:
                1) item (int): integer containing the index of the bounding box to be returned;
                2) item (slice): one-dimensional slice slicing a subset of bounding boxes;
                3) item (BoolTensor): tensor of shape [num_boxes] containing boolean values of boxes to be selected.

        Returns:
            selected_boxes (Boxes): New Boxes structure containing the selected bounding boxes.
        """

        boxes_tensor = self.boxes[item].view(1, -1) if isinstance(item, int) else self.boxes[item]
        selected_boxes = Boxes(boxes_tensor, self.format, self.normalized)

        return selected_boxes

    def __len__(self):
        """
        Implements the __len__ method of the Boxes structure.

        It is measured as the number of boxes within the structure.

        Returns:
            num_boxes (int): Number of boxes within the structure.
        """

        num_boxes = len(self.boxes)

        return num_boxes

    def __repr__(self):
        """
        Implements the __repr__ method of the Boxes structure.

        Returns:
            boxes_string (str): String containing information about the Boxes structure.
        """

        boxes_string = "Boxes structure:\n"
        boxes_string += f"   Size: {len(self)}\n"
        boxes_string += f"   Format: {self.format}\n"
        boxes_string += f"   Normalized: {self.normalized}\n"
        boxes_string += f"   Content: {self.boxes}"

        return boxes_string

    def area(self):
        """
        Computes the area of each of the bounding boxes within the Boxes structure.

        Returns:
            areas (FloatTensor): Tensor of shape [num_boxes] containing the area of each of the boxes.
        """

        if self.format in ['cxcywh', 'xywh']:
            areas = self.boxes[:, 2] * self.boxes[:, 3]

        elif self.format == 'xyxy':
            areas = (self.boxes[:, 2] - self.boxes[:, 0]) * (self.boxes[:, 3] - self.boxes[:, 1])

        return areas

    @staticmethod
    def cat(boxes_list):
        """
        Concatenate list of Boxes structures into single Boxes structure.

        Args:
            boxes_list (List): List of size [num_structures] with Boxes structures to be concatenated.

        Returns:
            cat_boxes (Boxes): Boxes structure containing the concatenated input Boxes structures.
        """

        # Check whehter all Boxes structures have the same format
        format_set = {boxes.format for boxes in boxes_list}
        assert_msg = f"All Boxes structures must have the same format (got {format_set})."
        assert len(format_set) == 1, assert_msg

        # Check whehter all Boxes structures have the same normalized attribute
        normalized_set = {boxes.normalized for boxes in boxes_list}
        assert_msg = f"All Boxes structures must have the same normalized attribute (got {normalized_set})."
        assert len(normalized_set) == 1, assert_msg

        # Concatenate Boxes structures into single Boxes structure
        boxes_tensor = torch.cat([structure.boxes for structure in boxes_list])
        cat_boxes = Boxes(boxes_tensor, format_set[0], normalized_set[0])

        return cat_boxes

    def clip(self, clip_region):
        """
        Clips bounding boxes to be within the given boundaries.

        Args:
            clip_region (Tuple): Following two input formats are supported:
                1) tuple of size [2] containing the (right, bottom) clip boundaries;
                2) tuple of size [4] containing the (left, top, right, bottom) clip boundaries.

        Returns:
            well_defined (BoolTensor): Tensor of shape [num_boxes] indicating well-defined boxes after clipping.
        """

        # Set left and top boundaries to zero if not specified
        if len(clip_region) == 2:
            clip_region = (0, 0, *clip_region)

        # Convert boxes if needed
        in_format = self.format
        if in_format != 'xyxy':
            self.to_format('xyxy')

        # Clip boxes according to its boundaries
        self.boxes[:, 0].clamp_(min=clip_region[0], max=clip_region[2])
        self.boxes[:, 1].clamp_(min=clip_region[1], max=clip_region[3])
        self.boxes[:, 2].clamp_(min=clip_region[0], max=clip_region[2])
        self.boxes[:, 3].clamp_(min=clip_region[1], max=clip_region[3])

        # Convert boxes back if needed
        if in_format != 'xyxy':
            self.to_format(in_format)

        # Find out which boxes are well-defined after clipping
        well_defined = self.well_defined()

        return well_defined

    def clone(self):
        """
        Clones the Boxes structure into a new Boxes structure.

        Returns:
            cloned_boxes (Boxes): Cloned Boxes structure.
        """

        cloned_boxes = Boxes(self.boxes.clone(), self.format, self.normalized)

        return cloned_boxes

    def crop(self, crop_region):
        """
        Updates the bounding boxes w.r.t. to the given crop region.

        Args:
            crop_region (Tuple): Tuple delineating the cropped region in (left, top, width, height) format.

        Returns:
            well_defined (BoolTensor): Tensor of shape [num_boxes] indicating well-defined boxes after cropping.
        """

        # Update bounding boxes w.r.t. the crop
        left, top, width, height = crop_region

        if self.format in ['cxcywh', 'xywh']:
            self.boxes[:, :2] = self.boxes[:, :2] - torch.tensor([left, top]).to(self.boxes)

        elif self.format == 'xyxy':
            self.boxes = self.boxes - torch.tensor([left, top, left, top]).to(self.boxes)

        # Clip boxes and find out which ones are well-defined
        well_defined = self.clip((width, height))

        return well_defined

    def hflip(self, image_width):
        """
        Flips the bounding boxes horizontally.

        Args:
            image_width (int): Integer containing the image width.
        """

        if self.format == 'cxcywh':
            self.boxes[:, 0] = image_width - self.boxes[:, 0]

        elif self.format == 'xywh':
            self.boxes[:, 0] = image_width - (self.boxes[:, 0] + self.boxes[:, 2])

        elif self.format == 'xyxy':
            self.boxes[:, [0, 2]] = image_width - self.boxes[:, [2, 0]]

    def resize(self, resize_ratio):
        """
        Resizes the bounding boxes by the given resize ratio.

        Args:
            resize_ratio (Tuple): Tuple of size [2] containing the resize ratio as (width_ratio, height_ratio).
        """

        resize_ratio = torch.tensor([*resize_ratio, *resize_ratio]).to(self.boxes)
        self.boxes = resize_ratio * self.boxes

    def to(self, *args, **kwargs):
        """
        Performs type and/or device conversion for the tensors within the Boxes structure.

        Returns:
            Updated Boxes structure with converted tensors.
        """

        self.boxes.to(*args, **kwargs)

        return self

    def to_format(self, format):
        """
        Changes the bounding boxes of the Boxes structure to the given format.

        Args:
            format (str): String containing the bounding box format to convert to.

        Raises:
            ValueError: Raised when given format results in unknown format conversion.
        """

        if self.format != format:
            method_name = f'fmt_{self.format}_to_{format}'

            if hasattr(self, method_name):
                getattr(self, method_name)()
            else:
                raise ValueError(f"Unknown format conversion {method_name}.")

    def fmt_cxcywh_to_xywh(self):
        """
        Function transforming boxes from (center_x, center_y, width, height) to (left, top, width, height) format.
        """

        cx, cy, w, h = self.boxes.unbind(dim=-1)
        self.boxes = torch.stack([cx-0.5*w, cy-0.5*h, w, h], dim=-1)
        self.format = 'xywh'

    def fmt_cxcywh_to_xyxy(self):
        """
        Function transforming boxes from (center_x, center_y, width, height) to (left, top, right, bottom) format.
        """

        cx, cy, w, h = self.boxes.unbind(dim=-1)
        self.boxes = torch.stack([cx-0.5*w, cy-0.5*h, cx+0.5*w, cy+0.5*h], dim=-1)
        self.format = 'xyxy'

    def fmt_xywh_to_cxcywh(self):
        """
        Function transforming boxes from (left, top, width, height) to (center_x, center_y, width, height) format.
        """

        x0, y0, w, h = self.boxes.unbind(dim=-1)
        self.boxes = torch.stack([x0+w/2, y0+h/2, w, h], dim=-1)
        self.format = 'cxcywh'

    def fmt_xywh_to_xyxy(self):
        """
        Function transforming boxes from (left, top, width, height) to (left, top, right, bottom) format.
        """

        x0, y0, w, h = self.boxes.unbind(dim=-1)
        self.boxes = torch.stack([x0, y0, x0+w, y0+h], dim=-1)
        self.format = 'xyxy'

    def fmt_xyxy_to_cxcywh(self):
        """
        Function transforming boxes from (left, top, right, bottom) to (center_x, center_y, width, height) format.
        """

        x0, y0, x1, y1 = self.boxes.unbind(dim=-1)
        self.boxes = torch.stack([(x0+x1)/2, (y0+y1)/2, x1-x0, y1-y0], dim=-1)
        self.format = 'cxcywh'

    def fmt_xyxy_to_xywh(self):
        """
        Function transforming boxes from (left, top, right, bottom) to (left, top, width, height) format.
        """

        x0, y0, x1, y1 = self.boxes.unbind(dim=-1)
        self.boxes = torch.stack([x0, y0, x1-x0, y1-y0], dim=-1)
        self.format = 'xywh'

    def well_defined(self):
        """
        Finds well-defined boxes, i.e. boxes with (stricly) positive width and height.

        Returns:
            well_defined (BoolTensor): Boolean tensor of shape [num_boxes] indicating which boxes are well-defined.
        """

        if self.format in ['cxcywh', 'xywh']:
            well_defined = (self.boxes[:, 2] > 0) & (self.boxes[:, 3] > 0)

        elif self.format == 'xyxy':
            well_defined = (self.boxes[:, 2] > self.boxes[:, 0]) & (self.boxes[:, 3] > self.boxes[:, 1])

        return well_defined


def box_giou(boxes1, boxes2):
    """
    Function computing the 2D GIoU's between every pair of boxes from two Boxes structures.

    Args:
        boxes1 (Boxes): First Boxes structure of axis-aligned bounding boxes of size [N].
        boxes2 (Boxes): Second Boxes structure of axis-aligned bounding boxes of size [M]

    Returns:
        gious (FloatTensor): The 2D GIoU's between every pair of boxes of shape [N, M].
    """

    # Check for degenerate boxes (i.e. boxes with negative width or height)
    assert boxes1.well_defined().all(), "boxes1 input contains degenerate boxes"
    assert boxes1.well_defined().all(), "boxes2 input contains degenerate boxes"

    # Convert bounding boxes to (left, top, right, bottom) format
    boxes1 = boxes1.clone().to_format('xyxy')
    boxes2 = boxes2.clone().to_format('xyxy')

    # Get bounding box tensors
    tensors1 = boxes1.boxes
    tensors2 = boxes2.boxes

    # Compute areas of smallest axis-aligned box containing the union
    lt = torch.min(tensors1[:, None, :2], tensors2[:, :2])  # [N,M,2]
    rb = torch.max(tensors1[:, None, 2:], tensors2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    areas = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    # Compute GIoU's based on 2D IoU's, union area's and axis-aligned union areas
    ious, unions = box_iou(boxes1, boxes2)
    gious = ious - (areas - unions) / areas

    return gious


def box_iou(boxes1, boxes2):
    """
    Function computing the 2D IoU's and union areas between every pair of boxes from two Boxes structures.

    Args:
        boxes1 (Boxes): First Boxes structure of axis-aligned bounding boxes of size [N].
        boxes2 (Boxes): Second Boxes structure of axis-aligned bounding boxes of size [M].

    Returns:
        ious (FloatTensor): The 2D IoU's between every pair of boxes of shape [N, M].
        unions (FloatTensor): The areas of the box unions between every pair of boxes of shape [N, M].
    """

    # Compute bounding box areas
    areas1 = boxes1.area()
    areas2 = boxes2.area()

    # Convert bounding boxes to (left, top, right, bottom) format
    boxes1 = boxes1.clone().to_format('xyxy')
    boxes2 = boxes2.clone().to_format('xyxy')

    # Get bounding box tensors
    tensors1 = boxes1.boxes
    tensors2 = boxes2.boxes

    # Get intersection areas
    lt = torch.max(tensors1[:, None, :2], tensors2[:, :2])  # [N,M,2]
    rb = torch.min(tensors1[:, None, 2:], tensors2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inters = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    # Get union areas and 2D IoU's
    unions = areas1[:, None] + areas2 - inters
    ious = inters / unions

    return ious, unions
