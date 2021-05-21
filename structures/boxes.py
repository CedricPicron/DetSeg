"""
Boxes structure and bounding box utilities.
"""
import math

import torch

from models.functional.position import sine_pos_encodings
from utils.distributed import get_world_size, is_dist_avail_and_initialized


class Boxes(object):
    """
    Class implementing the Boxes structure.

    Attributes:
        boxes (FloatTensor): Tensor of axis-aligned bounding boxes of shape [num_boxes, 4].
        format (str): String containing the format in which the bounding boxes are expressed.
        normalized (str): String indicating whether and w.r.t. what boxes are normalized.
        boxes_per_img (LongTensor): Tensor of shape [num_images] containing the number of boxes per batched image.
    """

    def __init__(self, boxes, format, normalized='false', boxes_per_img=None):
        """
        Initializes the Boxes structure.

        Args:
            boxes (FloatTensor): Tensor of axis-aligned bounding boxes of shape [num_boxes, 4].
            format (str): String containing the format in which the bounding boxes are expressed.
            normalized (str): String indicating whether and w.r.t. what boxes are normalized (default='false').
            boxes_per_img (LongTensor): Number of boxes per batched image of shape [num_images] (default=None).
        """

        # Check whether input boxes tensor has valid shape
        check = boxes.dim() == 2 and boxes.shape[-1] == 4
        assert_msg = f"Tensor has incorrect shape {boxes.shape} to become Boxes structure."
        assert check, assert_msg

        # Set boxes, format and normalized attributes
        self.boxes = boxes
        self.format = format
        self.normalized = normalized

        # Set boxes per image attribute
        if boxes_per_img is None:
            self.boxes_per_img = torch.tensor([len(boxes)], dtype=torch.int64, device=boxes.device)
        else:
            self.boxes_per_img = torch.as_tensor(boxes_per_img, dtype=torch.int64, device=boxes.device)

    def __getitem__(self, key):
        """
        Implements the __getitem__ method of the Boxes structure.

        Args:
            key: Object of any type also supported by Tensor determining the selected bounding boxes.

        Returns:
            item (Boxes): New Boxes structure containing the selected bounding boxes.
        """

        boxes_tensor = self.boxes[key].view(1, -1) if isinstance(key, int) else self.boxes[key]
        item = Boxes(boxes_tensor, self.format, self.normalized)

        return item

    def __len__(self):
        """
        Implements the __len__ method of the Boxes structure.

        Returns:
            num_boxes (int): Number of boxes within the Boxes structure.
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
        boxes_string += f"   Boxes per image: {self.boxes_per_img}\n"
        boxes_string += f"   Content: {self.boxes}"

        return boxes_string

    def __setitem__(self, key, item):
        """
        Implements the __setitem__ method of the Boxes structure.

        Args:
            key: Object of any type also supported by Tensor determining the bounding boxes to be set.
            item (Boxes): Boxes structure containing the bounding boxes to be set.
        """

        # Check whether formats are consistent
        check = self.format == item.format
        assert_msg = f"Inconsistent formats between self '{self.format}' and item '{item.format}'."
        assert check, assert_msg

        # Check whether normalized attributes are consistent
        check = self.normalized == item.normalized
        assert_msg = f"Inconsistent normalizations between self '{self.normalized}' and item '{item.normalized}'."
        assert check, assert_msg

        # Set boxes attribute
        self.boxes[key] = item.boxes

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
    def cat(boxes_list, same_image=False):
        """
        Concatenate list of Boxes structures into single Boxes structure.

        Args:
            boxes_list (List): List of size [num_structures] with Boxes structures to be concatenated.
            same_image (boolean): Boolean indicating whether Boxes structures belong to the same image or not.

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

        # Check whether all Boxes structures reside on the same device
        device_set = {s.boxes.device for s in boxes_list}
        assert len(device_set) == 1, f"All boxes should reside on the same device (got {device_set})."

        # Concatenate Boxes structures into single Boxes structure
        boxes_tensor = torch.cat([structure.boxes for structure in boxes_list])
        boxes_per_img = torch.cat([structure.boxes_per_img for structure in boxes_list]) if not same_image else None
        cat_boxes = Boxes(boxes_tensor, format_set.pop(), normalized_set.pop(), boxes_per_img)

        return cat_boxes

    def clip(self, clip_region):
        """
        Clips bounding boxes to be within the given clip region.

        Args:
            clip_region (Tuple): Following two input formats are supported:
                1) tuple of size [2] containing the (right, bottom) clip boundaries;
                2) tuple of size [4] containing the (left, top, right, bottom) clip boundaries.

        Returns:
            self (Boxes): Updated Boxes structure with clipped bounding boxes.
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

        return self, well_defined

    def clone(self):
        """
        Clones the Boxes structure into a new Boxes structure.

        Returns:
            cloned_boxes (Boxes): Cloned Boxes structure.
        """

        cloned_boxes = Boxes(self.boxes.clone(), self.format, self.normalized, self.boxes_per_img.clone())

        return cloned_boxes

    def crop(self, crop_region):
        """
        Updates the bounding boxes w.r.t. the given crop operation.

        Args:
            crop_region (Tuple): Tuple delineating the cropped region in (left, top, right, bottom) format.

        Returns:
            self (Boxes): Boxes structure updated w.r.t. the crop operation.
            well_defined (BoolTensor): Tensor of shape [num_boxes] indicating well-defined boxes after cropping.
        """

        # Update bounding boxes w.r.t. the crop operation
        left, top, right, bottom = crop_region

        if self.format in ['cxcywh', 'xywh']:
            self.boxes[:, :2] = self.boxes[:, :2] - torch.tensor([left, top]).to(self.boxes)

        elif self.format == 'xyxy':
            self.boxes = self.boxes - torch.tensor([left, top, left, top]).to(self.boxes)

        # Clip boxes and find out which ones are well-defined
        self, well_defined = self.clip((right-left, bottom-top))

        return self, well_defined

    def dist_len(self):
        """
        Gets the average number of boxes across Boxes structures from different processes.

        Returns:
            num_boxes (float): Average number of boxes across Boxes structures from different processes.
        """

        # Get number of boxes for structure from this process
        num_boxes = len(self)

        # Return if not in distributed mode
        if not is_dist_avail_and_initialized():
            return num_boxes

        # Get average number of boxes across structures from different processes
        num_boxes = torch.tensor([num_boxes], dtype=torch.float, device=self.boxes.device)
        torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes/get_world_size(), min=1).item()

        return num_boxes

    def fmt_cxcywh_to_xywh(self):
        """
        Function transforming boxes from (center_x, center_y, width, height) to (left, top, width, height) format.

        Returns:
            self (Boxes): Updated Boxes structure in the (left, top, width, height) format.
        """

        cx, cy, w, h = self.boxes.unbind(dim=-1)
        self.boxes = torch.stack([cx-0.5*w, cy-0.5*h, w, h], dim=-1)
        self.format = 'xywh'

        return self

    def fmt_cxcywh_to_xyxy(self):
        """
        Function transforming boxes from (center_x, center_y, width, height) to (left, top, right, bottom) format.

        Returns:
            self (Boxes): Updated Boxes structure in the (left, top, right, bottom) format.
        """

        cx, cy, w, h = self.boxes.unbind(dim=-1)
        self.boxes = torch.stack([cx-0.5*w, cy-0.5*h, cx+0.5*w, cy+0.5*h], dim=-1)
        self.format = 'xyxy'

        return self

    def fmt_xywh_to_cxcywh(self):
        """
        Function transforming boxes from (left, top, width, height) to (center_x, center_y, width, height) format.

        Returns:
            self (Boxes): Updated Boxes structure in the (center_x, center_y, width, height) format.
        """

        x0, y0, w, h = self.boxes.unbind(dim=-1)
        self.boxes = torch.stack([x0+w/2, y0+h/2, w, h], dim=-1)
        self.format = 'cxcywh'

        return self

    def fmt_xywh_to_xyxy(self):
        """
        Function transforming boxes from (left, top, width, height) to (left, top, right, bottom) format.

        Returns:
            self (Boxes): Updated Boxes structure in the (left, top, right, bottom) format.
        """

        x0, y0, w, h = self.boxes.unbind(dim=-1)
        self.boxes = torch.stack([x0, y0, x0+w, y0+h], dim=-1)
        self.format = 'xyxy'

        return self

    def fmt_xyxy_to_cxcywh(self):
        """
        Function transforming boxes from (left, top, right, bottom) to (center_x, center_y, width, height) format.

        Returns:
            self (Boxes): Updated Boxes structure in the (center_x, center_y, width, height) format.
        """

        x0, y0, x1, y1 = self.boxes.unbind(dim=-1)
        self.boxes = torch.stack([(x0+x1)/2, (y0+y1)/2, x1-x0, y1-y0], dim=-1)
        self.format = 'cxcywh'

        return self

    def fmt_xyxy_to_xywh(self):
        """
        Function transforming boxes from (left, top, right, bottom) to (left, top, width, height) format.

        Returns:
            self (Boxes): Updated Boxes structure in the (left, top, width, height) format.
        """

        x0, y0, x1, y1 = self.boxes.unbind(dim=-1)
        self.boxes = torch.stack([x0, y0, x1-x0, y1-y0], dim=-1)
        self.format = 'xywh'

        return self

    def hflip(self, image_width):
        """
        Flips the bounding boxes horizontally.

        Args:
            image_width (int): Integer containing the image width.

        Returns:
            self (Boxes): Updated Boxes structure with horizontally flipped bounding boxes.
        """

        if self.format == 'cxcywh':
            self.boxes[:, 0] = image_width - self.boxes[:, 0]

        elif self.format == 'xywh':
            self.boxes[:, 0] = image_width - (self.boxes[:, 0] + self.boxes[:, 2])

        elif self.format == 'xyxy':
            self.boxes[:, [0, 2]] = image_width - self.boxes[:, [2, 0]]

        return self

    def normalize(self, images, with_padding=True):
        """
        Normalizes bounding boxes w.r.t. the image sizes from the given Images structure.

        It is the inverse operation of 'to_img_scale'.

        Args:
            images (Images): Images structure containing batched images with their entire transform history.
            with_padding (bool): Whether to normalize w.r.t the image sizes with padding or not (default=True).

        Returns:
            self (Boxes): Updated Boxes structure with normalized bounding boxes.
        """

        # Normalize bounding box coordinates if not yet normalized
        if self.normalized == 'false':

            # Get image sizes in (width, height) format
            img_sizes = images.size(with_padding=with_padding)
            img_sizes = [img_sizes for _ in range(len(images))] if with_padding else img_sizes

            # Normalize bounding box coordinates w.r.t. the image sizes
            scales = torch.tensor([[*img_size, *img_size] for img_size in img_sizes]).to(self.boxes)
            self.boxes = self.boxes / scales.repeat_interleave(self.boxes_per_img, dim=0)
            self.normalized = 'img_with_padding' if with_padding else 'img_without_padding'

        return self

    def pad(self, padding):
        """
        Updates the bounding boxes w.r.t. the given padding operation.

        Args:
            padding (Tuple): Padding vector of size [4] with padding values in (left, top, right, bottom) format.

        Returns:
            self (Boxes): Boxes structure updated w.r.t. the padding operation.
        """

        # Update bounding boxes w.r.t. the padding operation
        left, top, width, height = padding

        if self.format in ['cxcywh', 'xywh']:
            self.boxes[:, :2] = self.boxes[:, :2] + torch.tensor([left, top]).to(self.boxes)

        elif self.format == 'xyxy':
            self.boxes = self.boxes + torch.tensor([left, top, left, top]).to(self.boxes)

        return self

    def resize(self, resize_ratio):
        """
        Resizes the bounding boxes by the given resize ratio.

        Args:
            resize_ratio (Tuple): Tuple of size [2] containing the resize ratio as (width_ratio, height_ratio).

        Returns:
            self (Boxes): Boxes structure updated w.r.t. the resize operation.
        """

        resize_ratio = torch.tensor([*resize_ratio, *resize_ratio]).to(self.boxes)
        self.boxes = resize_ratio * self.boxes

        return self

    def to(self, *args, **kwargs):
        """
        Performs type and/or device conversion for the tensors within the Boxes structure.

        Returns:
            self (Boxes): Updated Boxes structure with converted tensors.
        """

        self.boxes = self.boxes.to(*args, **kwargs)
        self.boxes_per_img = self.boxes_per_img.to(*args, **kwargs)

        return self

    def to_format(self, format):
        """
        Changes the bounding boxes of the Boxes structure to the given format.

        Args:
            format (str): String containing the bounding box format to convert to.

        Returns:
            self (Boxes): Boxes structure with bounding boxes in the specified format.

        Raises:
            ValueError: Raised when given format results in unknown format conversion.
        """

        if self.format != format:
            method_name = f'fmt_{self.format}_to_{format}'

            if hasattr(self, method_name):
                self = getattr(self, method_name)()
            else:
                raise ValueError(f"Unknown format conversion {method_name}.")

        return self

    def to_img_scale(self, images):
        """
        Scales normalized bounding boxes w.r.t. the image sizes from the given Images structure.

        It is the inverse operation of 'normalize'.

        Args:
            images (Images): Images structure containing batched images with their entire transform history.

        Returns:
            self (Boxes): Updated Boxes structure with bounding boxes resized to image scale.
        """

        # Scale if boxes are normalized w.r.t. image sizes
        if self.normalized in ['img_with_padding', 'img_without_padding']:

            # Get image sizes in (width, height) format
            with_padding = self.normalized == 'img_with_padding'
            img_sizes = images.size(with_padding=with_padding)
            img_sizes = [img_sizes for _ in range(len(images))] if with_padding else img_sizes

            # Scale bounding box coordinates w.r.t. the image sizes
            scales = torch.tensor([[*img_size, *img_size] for img_size in img_sizes]).to(self.boxes)
            self.boxes = self.boxes * scales.repeat_interleave(self.boxes_per_img, dim=0)
            self.normalized = 'false'

        return self

    def transform(self, images, inverse=False):
        """
        Applies transforms (recorded by the given Images structure) to each of the bounding boxes.

        Args:
            images (Images): Images structure containing batched images with their entire transform history.
            inverse (bool): Boolean indicating whether inverse transformation should be applied or not.

        Returns:
            self (Boxes): Boxes structure updated by the transforms from the Images structure.
            well_defined (BoolTensor): Tensor of shape [num_boxes] indicating well-defined boxes after transformation.
        """

        # Scale if boxes are normalized
        self = self.to_img_scale(images)

        # Get image splits
        img_splits = torch.cumsum(self.boxes_per_img, dim=0)
        img_splits = [0] + img_splits.tolist()

        # Apply transforms
        for i0, i1, transforms in zip(img_splits[:-1], img_splits[1:], images.transforms):
            boxes = self[i0:i1]

            if inverse:
                range_obj = range(len(transforms)-1, -1, -1)
            else:
                range_obj = range(len(transforms))

            for j in range_obj:
                transform = transforms[j]

                if transform[0] == 'crop':
                    if not inverse:
                        boxes, _ = boxes.crop(transform[1])
                    else:
                        boxes = boxes.pad(transform[2])

                elif transform[0] == 'hflip':
                    boxes = boxes.hflip(transform[1])

                elif transform[0] == 'pad':
                    if not inverse:
                        boxes = boxes.pad(transform[1])
                    else:
                        boxes, _ = boxes.crop(transform[2])

                elif transform[0] == 'resize':
                    resize_ratio = tuple(1/x for x in transform[1]) if inverse else transform[1]
                    boxes = boxes.resize(resize_ratio)

            self[i0:i1] = boxes

        # Get well-defined boxes after transformation
        well_defined = self.well_defined()

        return self, well_defined

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


def apply_box_deltas(box_deltas, in_boxes, scale_clamp=math.log(1000.0/16)):
    """
    Function applying box deltas to the given Boxes structure.

    Args:
        box_deltas (FloatTensor): Tensor of shape [num_boxes, 4] encoding the box transformation to be applied.
        in_boxes (Boxes): Boxes structure of axis-aligned bounding boxes of size [num_boxes] to be transformed.
        scale_clamp (float): Optional threshold indicating the maximum allowed relative change in width or height.

    Returns:
        out_boxes (Boxes): Boxes structure of transformed axis-aligned bounding boxes of size [num_boxes].
    """

    # Check whether both inputs have same length
    check = len(box_deltas) == len(in_boxes)
    assert_msg = f"Both deltas and boxes inputs should have same length (got {len(box_deltas)} and {len(in_boxes)})."
    assert check, assert_msg

    # Check for degenerate boxes (i.e. boxes with non-positive width or height)
    assert in_boxes.well_defined().all(), "in_boxes input contains degenerate boxes"

    # Get transformed bounding boxes
    out_boxes = in_boxes.clone().to_format('cxcywh')
    out_boxes.boxes[:, :2] += box_deltas[:, :2] * out_boxes.boxes[:, 2:]
    out_boxes.boxes[:, 2:] *= torch.exp(box_deltas[:, 2:].clamp(max=scale_clamp))

    return out_boxes


def apply_edge_dists(edge_dists, pts, scales=None, normalized='false', alter_degenerate=True):
    """
    Function applying normalized point to box edge distances to each corresponding point to obtain a bounding box.

    Args:
        edge_dists (FloatTensor): Normalized point to edge distances of shape [N, 4].
        pts (FloatTensor): Two-dimensional points of shape [N, 2].
        scales (FloatTensor): Tensor of shape [N, 2] with which the edge distances were normalized (default=None).
        normalized (str): String indicating whether and w.r.t. what output boxes will be normalized (default='false').
        alter_degenerate (bool): Whether to alter boxes such that no degenerate boxes are outputted (default=True).

    Returns:
        out_boxes: Boxes structure of output boxes constructed from the given inputs of size [N].
    """

    # Get default scales variable if not specified
    if scales is None:
        scales = torch.ones_like(pts)

    # Check whether inputs have same length
    check = len(edge_dists) == len(pts)
    assert_msg = f"Both distances and points inputs should have same length (got {len(edge_dists)} and {len(pts)})."
    assert check, assert_msg

    check = len(pts) == len(scales)
    assert_msg = f"Both points and scales inputs should have same length (got {len(pts)} and {len(scales)})."
    assert check, assert_msg

    # Construct output boxes
    left_top = pts - scales*edge_dists[:, :2]
    right_bottom = pts + scales*edge_dists[:, 2:]

    if alter_degenerate:
        degenerate = right_bottom <= left_top
        right_bottom[degenerate] = left_top[degenerate] + 1e-6

    out_boxes = torch.cat([left_top, right_bottom], dim=1)
    out_boxes = Boxes(out_boxes, format='xyxy', normalized=normalized)

    return out_boxes


def box_giou(boxes1, boxes2):
    """
    Function computing the 2D GIoU's between every pair of boxes from two Boxes structures.

    Args:
        boxes1 (Boxes): First Boxes structure of axis-aligned bounding boxes of size [N].
        boxes2 (Boxes): Second Boxes structure of axis-aligned bounding boxes of size [M].

    Returns:
        gious (FloatTensor): The 2D GIoU's between every pair of boxes of shape [N, M].
    """

    # Check for degenerate boxes (i.e. boxes with non-positive width or height)
    assert boxes1.well_defined().all(), "boxes1 input contains degenerate boxes"
    assert boxes2.well_defined().all(), "boxes2 input contains degenerate boxes"

    # Compute 2D IoU's and union areas
    ious, unions = box_iou(boxes1, boxes2, return_unions=True)

    # Convert bounding boxes to (left, top, right, bottom) format and get box tensors
    boxes1 = boxes1.clone().to_format('xyxy').boxes
    boxes2 = boxes2.clone().to_format('xyxy').boxes

    # Compute areas of smallest axis-aligned box containing the union
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    areas = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    # Compute GIoU's based on 2D IoU's, union area's and axis-aligned union areas
    gious = ious - (areas - unions) / areas

    return gious


def box_intersection(boxes1, boxes2):
    """
    Function computing the intersection areas between every pair of boxes from two Boxes structures.

    Args:
        boxes1 (Boxes): First Boxes structure of axis-aligned bounding boxes of size [N].
        boxes2 (Boxes): Second Boxes structure of axis-aligned bounding boxes of size [M].

    Returns:
        inters (FloatTensor): The intersection areas between every pair of boxes of shape [N, M].
    """

    # Check for degenerate boxes (i.e. boxes with non-positive width or height)
    assert boxes1.well_defined().all(), "boxes1 input contains degenerate boxes"
    assert boxes2.well_defined().all(), "boxes2 input contains degenerate boxes"

    # Convert bounding boxes to (left, top, right, bottom) format and get box tensors
    boxes1 = boxes1.clone().to_format('xyxy').boxes
    boxes2 = boxes2.clone().to_format('xyxy').boxes

    # Get intersection areas
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inters = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return inters


def box_iou(boxes1, boxes2, return_inters=False, return_unions=False):
    """
    Function computing the 2D IoU's and union areas between every pair of boxes from two Boxes structures.

    Args:
        boxes1 (Boxes): First Boxes structure of axis-aligned bounding boxes of size [N].
        boxes2 (Boxes): Second Boxes structure of axis-aligned bounding boxes of size [M].
        return_inters (bool): Whether to return areas of the box intersections or not (default=False).
        return_unions (bool): Whether to return areas of the box unions or not (default=False).

    Returns:
        ious (FloatTensor): The 2D IoU's between every pair of boxes of shape [N, M].

        If 'return_inters' is True:
            inters (FloatTensor): The areas of the box intersections between every pair of boxes of shape [N, M].

        If 'return_unions' is True:
            unions (FloatTensor): The areas of the box unions between every pair of boxes of shape [N, M].
    """

    # Check for degenerate boxes (i.e. boxes with non-positive width or height)
    assert boxes1.well_defined().all(), "boxes1 input contains degenerate boxes"
    assert boxes2.well_defined().all(), "boxes2 input contains degenerate boxes"

    # Compute bounding box areas
    areas1 = boxes1.area()
    areas2 = boxes2.area()

    # Get intersection areas
    inters = box_intersection(boxes1, boxes2)

    # Get union areas and 2D IoU's
    unions = areas1[:, None] + areas2 - inters
    ious = inters / unions

    # Return IoU's and intersection/union areas if requested
    if not return_inters and not return_unions:
        return ious
    elif return_inters and not return_unions:
        return ious, inters
    elif not return_inters and return_unions:
        return ious, unions
    else:
        return ious, inters, unions


def get_box_deltas(boxes1, boxes2):
    """
    Function computing box deltas encoding the transformation from one Boxes structure to a second one.

    Args:
        boxes1 (Boxes): First Boxes structure of axis-aligned bounding boxes of size [num_boxes] to transform from.
        boxes2 (Boxes): Second Boxes structure of axis-aligned bounding boxes of size [num_boxes] to transform to.

    Returns:
        box_deltas (FloatTensor): Tensor of shape [num_boxes, 4] encoding the transformation between both Boxes inputs.
    """

    # Check whether both inputs contain the same amount of boxes
    check = len(boxes1) == len(boxes2)
    assert_msg = f"Both Boxes inputs should contain the same amount of boxes (got {len(boxes1)} and {len(boxes2)})."
    assert check, assert_msg

    # Check whether normalized attributes are consistent
    check = boxes1.normalized == boxes2.normalized
    assert_msg = f"Inconsistent normalizations between Boxes inputs (got {boxes1.normalized} and {boxes2.normalized})."
    assert check, assert_msg

    # Check for degenerate boxes (i.e. boxes with non-positive width or height)
    assert boxes1.well_defined().all(), "boxes1 input contains degenerate boxes"
    assert boxes2.well_defined().all(), "boxes2 input contains degenerate boxes"

    # Convert boxes to (center_x, center_y, width, height) format and get box tensors
    boxes1 = boxes1.clone().to_format('cxcywh').boxes
    boxes2 = boxes2.clone().to_format('cxcywh').boxes

    # Get box deltas
    box_deltas = torch.zeros_like(boxes1)
    box_deltas[:, :2] = (boxes2[:, :2] - boxes1[:, :2]) / boxes1[:, 2:]
    box_deltas[:, 2:] = torch.log(boxes2[:, 2:] / boxes1[:, 2:])

    return box_deltas


def get_edge_dists(pts, boxes, scales=None):
    """
    Function computing the normalized distances of each point to its corresponding box edges.

    Args:
        pts (FloatTensor): Two-dimensional points of shape [N, 2].
        boxes (Boxes): Boxes structure of axis-aligned bounding boxes of size [N].
        scales (FloatTensor): Tensor of shape [N, 2] normalizing the point to edge distances (default=None).

    Returns:
        edge_dists (FloatTensor): Normalized point to edge distances of shape [N, 4].
    """

    # Get default scales variable if not specified
    if scales is None:
        scales = torch.ones_like(pts)

    # Check whether inputs have same length
    check = len(pts) == len(boxes)
    assert_msg = f"Both points and boxes inputs should have same length (got {len(pts)} and {len(boxes)})."
    assert check, assert_msg

    check = len(pts) == len(scales)
    assert_msg = f"Both points and scales inputs should have same length (got {len(pts)} and {len(scales)})."
    assert check, assert_msg

    # Check for degenerate boxes (i.e. boxes with non-positive width or height)
    assert boxes.well_defined().all(), "boxes input contains degenerate boxes"

    # Convert bounding boxes to (left, top, right, bottom) format and get box tensors
    boxes = boxes.clone().to_format('xyxy').boxes

    # Get normalized point to edge distances
    left_top_dists = (pts - boxes[:, :2]) / scales
    right_bottom_dists = (boxes[:, 2:] - pts) / scales
    edge_dists = torch.cat([left_top_dists, right_bottom_dists], dim=1)

    return edge_dists


def get_feat_boxes(feat_maps):
    """
    Function computing feature boxes based on map sizes of given feature maps.

    Args:
        feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

    Returns:
        feat_boxes (Boxes): Axis-aligned bounding boxes related to features of given feature maps of size [num_feats].
    """

    # Get feature centers
    _, feat_cts_maps = sine_pos_encodings(feat_maps, normalize=True)
    feat_cts = torch.cat([feat_cts_map.flatten(1).t() for feat_cts_map in feat_cts_maps], dim=0)

    # Get feature widths and heights
    map_numel = torch.tensor([feat_map.flatten(2).shape[-1] for feat_map in feat_maps]).to(feat_cts.device)
    feat_wh = torch.tensor([[1/s for s in feat_map.shape[:1:-1]] for feat_map in feat_maps]).to(feat_cts)
    feat_wh = torch.repeat_interleave(feat_wh, map_numel, dim=0)

    # Concatenate feature centers, widths and heights to get feature boxes
    feat_boxes = torch.cat([feat_cts, feat_wh], dim=1)
    feat_boxes = Boxes(feat_boxes, format='cxcywh', normalized='img_with_padding')

    return feat_boxes


def pts_inside_boxes(pts, boxes):
    """
    Function checking for every (point, box) pair whether the point lies inside the box or not.

    Args:
        pts (FloatTensor): Two-dimensional points of shape [N, 2].
        boxes (Boxes): Boxes structure of axis-aligned bounding boxes of size [M].

    Returns:
        inside_boxes (BoolTensor): Tensor of shape [N, M] indicating for each pair whether point lies inside box.
    """

    # Check for degenerate boxes (i.e. boxes with non-positive width or height)
    assert boxes.well_defined().all(), "boxes input contains degenerate boxes"

    # Convert bounding boxes to (left, top, right, bottom) format and get box tensors
    boxes = boxes.clone().to_format('xyxy').boxes

    # Check for every (point, box) pair whether point lies inside box
    left_top_inside = (pts[:, None] - boxes[None, :, :2]) > 0
    right_bottom_inside = (boxes[None, :, 2:] - pts[:, None]) > 0
    inside_boxes = torch.cat([left_top_inside, right_bottom_inside], dim=2).all(dim=2)

    return inside_boxes
