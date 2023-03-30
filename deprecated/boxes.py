"""
Boxes structure and bounding box utilities.
"""

import torch

from structures.boxes import Boxes


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
