"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xywh(boxes_cxcywh):
    """
    Function transforming boxes from (center_x, center_y, width, height) to (left, top, width, height) format.

    Args:
        boxes_cxcywh (Tensor): Boxes in (center_x, center_y, width, height) format of shape [*, 4].

    Returns:
        boxes_xywh (Tensor): Transformed boxes in (left, top, width, height) format of shape [*, 4].
    """

    cx, cy, w, h = boxes_cxcywh.unbind(dim=-1)
    boxes_xywh = torch.stack([cx-0.5*w, cy-0.5*h, w, h], dim=-1)

    return boxes_xywh


def box_cxcywh_to_xyxy(boxes_cxcywh):
    """
    Function transforming boxes from (center_x, center_y, width, height) to (left, top, right, bottom) format.

    Args:
        boxes_cxcywh (Tensor): Boxes in (center_x, center_y, width, height) format of shape [*, 4].

    Returns:
        boxes_xyxy (Tensor): Transformed boxes in (left, top, right, bottom) format of shape [*, 4].
    """

    cx, cy, w, h = boxes_cxcywh.unbind(dim=-1)
    boxes_xyxy = torch.stack([cx-0.5*w, cy-0.5*h, cx+0.5*w, cy+0.5*h], dim=-1)

    return boxes_xyxy


def box_xyxy_to_cxcywh(boxes_xyxy):
    """
    Function transforming boxes from (left, top, right, bottom) to (center_x, center_y, width, height) format.

    Args:
        boxes_xyxy (Tensor): Boxes in (left, top, right, bottom) format of shape [*, 4].

    Returns:
        boxes_cxcywh (Tensor): Transformed boxes in (center_x, center_y, width, height) format of shape [*, 4].
    """

    x0, y0, x1, y1 = boxes_xyxy.unbind(dim=-1)
    boxes_cxcywh = torch.stack([(x0+x1)/2, (y0+y1)/2, x1-x0, y1-y0], dim=-1)

    return boxes_cxcywh


def box_xyxy_to_xywh(boxes_xyxy):
    """
    Function transforming boxes from (left, top, right, bottom) to (left, top, width, height) format.

    Args:
        boxes_xyxy (Tensor): Boxes in (left, top, right, bottom) format of shape [*, 4].

    Returns:
        boxes_xywh (Tensor): Transformed boxes in (left, top, width, height) format of shape [*, 4].
    """

    x0, y0, x1, y1 = boxes_xyxy.unbind(dim=-1)
    boxes_xywh = torch.stack([x0, y0, x1-x0, y1-y0], dim=-1)

    return boxes_xywh


def box_iou(boxes1, boxes2):
    """
    Function computing the 2D IoU's between every pair of boxes from two sets of boxes.

    It is modified from torchvision to also return the union.

    Args:
        boxes1 (Tensor): First set of boxes in (left, top, right, bottom) format of shape [N, 4].
        boxes2 (Tensor): Second set of boxes in (left, top, right, bottom) format of shape [M, 4].

    Returns:
        iou (Tensor): The 2D IoU's between every pair of boxes from the two given sets of shape [N, M].
        union (Tensor): The areas of the box unions of every pair of boxes from the two given sets of shape [N, M].
    """

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    iou = inter / union

    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Function computing the GIoU's between every pair of boxes from two sets of boxes.

    For more information about the generalized IoU (GIoU), see https://giou.stanford.edu/.

    Args:
        boxes1 (Tensor): First set of boxes in (left, top, right, bottom) format of shape [N, 4].
        boxes2 (Tensor): Second set of boxes in (left, top, right, bottom) format of shape [M, 4].

    Returns:
        giou (Tensor): The GIoU's between every pair of boxes from the two given sets of shape [N, M].
    """

    # Check for degenerate boxes (i.e. boxes with negative width or height)
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all(), "boxes1 input contains degenerate boxes"
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all(), "boxes2 input contains degenerate boxes"

    # Compute area of smallest axis-aligned box containing the union
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    # Compute GIoU based on 2D IoU, union area and axis_aligned union area
    iou, union = box_iou(boxes1, boxes2)
    giou = iou - (area - union) / area

    return giou
