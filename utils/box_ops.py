"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area


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
        boxes_cxcywh (Tensor): Boxes in (left, top, right, bottom) format of shape [*, 4].

    Returns:
        boxes_xyxy (Tensor): Transformed boxes in (center_x, center_y, width, height) format of shape [*, 4].
    """

    x0, y0, x1, y1 = boxes_xyxy.unbind(dim=-1)
    boxes_cxcywh = torch.stack([(x0+x1)/2, (y0+y1)/2, x1-x0, y1-y0], dim=-1)

    return boxes_cxcywh


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


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
