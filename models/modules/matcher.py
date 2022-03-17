"""
Collection of matcher modules.
"""

import torch
from torch import nn

from models.build import MODELS
from structures.boxes import box_giou, box_iou


@MODELS.register_module()
class BoxMatcher(nn.Module):
    """
    Class implementing the BoxMatcher module.

    The module matches query boxes with target boxes, assigning one of following match labels to each query:
        - positive label with value 1 (query matches with at least one target);
        - negative label with value 0 (query does not match with any of the targets);
        - ignore label with value -1 (query has no matching verdict).

    Attributes:
        box_metric (str): String containing the metric computing similarities between query and target boxes.
        type (str): String containing the type of box matching procedure.
        static_type (str): String containing the type of static box matching procedure.
        abs_pos (float): Absolute threshold determining positive query labels during static box matching.
        abs_neg (float): Absolute threshold determining negative query labels during static box matching.
        rel_pos (int): Relative threshold determining positive query labels during static box matching.
        rel_neg (int): Relative threshold determining negative query labels during static box matching.
    """

    def __init__(self, box_metric='iou', type='static', static_type='rel', abs_pos=0.5, abs_neg=0.3, rel_pos=5,
                 rel_neg=10):
        """
        Initializes the BoxMatcher module.

        Args:
            box_metric (str): String with metric computing similarities between query and target boxes (default='iou').
            type (str): String containing the type of box matching procedure (default='static').
            static_type (str): String containing the type of static box matching procedure (default='rel').
            abs_pos (float): Absolute threshold determining positive labels during static box matching (default=0.5).
            abs_neg (float): Absolute threshold determining negative labels during static box matching (default=0.3).
            rel_pos (int): Relative threshold determining positive labels during static box matching (default=5).
            rel_neg (int): Relative threshold determining negative labels during static box matching (default=10).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attributes
        self.box_metric = box_metric
        self.type = type
        self.static_type = static_type
        self.abs_pos = abs_pos
        self.abs_neg = abs_neg
        self.rel_pos = rel_pos
        self.rel_neg = rel_neg

    def forward(self, qry_boxes, tgt_boxes):
        """
        Forward method of BoxMatcher module.

        Args:
            qry_boxes (Boxes): Structure containing axis-aligned query boxes of size [num_queries].
            tgt_boxes (Boxes): Structure containing axis-aligned target boxes of size [num_targets].

        Returns:
            match_labels (LongTensor): Match labels corresponding to each query of shape [num_queries].
            matched_qry_ids (LongTensor): Indices of matched queries of shape [num_pos_queries].
            matched_tgt_ids (LongTensor): Indices of corresponding matched targets of shape [num_pos_queries].

        Raises:
            ValueError: Error when an invalid bounding box metric is provided.
            ValueError: Error when an invalid type of box matching procedure is provided.
            ValueError: Error when an invalid type of static box matching procedure is provided.
        """

        # Get number of query and target boxes
        num_queries = len(qry_boxes)
        num_targets = len(tgt_boxes)

        # Get device
        device = qry_boxes.boxes.device

        # Handle case where there are no query or target boxes and return
        if (num_queries == 0) or (num_targets == 0):
            match_labels = torch.zeros(num_queries, device=device)
            matched_qry_ids = torch.zeros(0, device=device)
            matched_tgt_ids = torch.zeros(0, device=device)

            return match_labels, matched_qry_ids, matched_tgt_ids

        # Get similarity matrix between query and target boxes
        if self.box_metric == 'giou':
            sim_matrix = box_giou(qry_boxes, tgt_boxes)

        elif self.box_metric == 'iou':
            sim_matrix = box_iou(qry_boxes, tgt_boxes)

        else:
            error_msg = f"Invalid bounding box metric (got '{self.box_metric}')."
            raise ValueError(error_msg)

        # Get positive and negative label masks
        if self.type == 'static':

            if 'abs' in self.static_type:
                abs_pos_mask = sim_matrix >= self.abs_pos
                abs_neg_mask = sim_matrix < self.abs_neg

            if 'rel' in self.static_type:
                non_neg_ids = torch.topk(sim_matrix, self.rel_neg, dim=0, sorted=True).indices
                pos_ids = non_neg_ids[:self.rel_pos, :]

                rel_pos_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
                rel_pos_mask[pos_ids, torch.arange(num_targets)] = True

                rel_neg_mask = torch.ones_like(sim_matrix, dtype=torch.bool)
                rel_neg_mask[non_neg_ids, torch.arange(num_targets)] = False

            if self.static_type == 'abs':
                pos_mask = abs_pos_mask
                neg_mask = abs_neg_mask

            elif self.static_type == 'abs_and_rel':
                pos_mask = abs_pos_mask & rel_pos_mask
                neg_mask = abs_neg_mask | rel_neg_mask

            elif self.static_type == 'abs_or_rel':
                pos_mask = abs_pos_mask | rel_pos_mask
                neg_mask = abs_neg_mask & rel_neg_mask

            elif self.static_type == 'rel':
                pos_mask = rel_pos_mask
                neg_mask = rel_neg_mask

            else:
                error_msg = f"Unknown type of static box matching procedure (got {self.static_type})."
                raise ValueError(error_msg)

        else:
            error_msg = f"Unknown type of box matching procedure (got {self.type})."
            raise ValueError(error_msg)

        # Get match labels
        match_labels = torch.full(size=(num_queries,), fill_value=-1, device=device)
        match_labels[pos_mask.sum(dim=1) > 0] = 1
        match_labels[neg_mask.sum(dim=1) == num_targets] = 0

        # Get query and target indices of matches
        matched_qry_ids, matched_tgt_ids = pos_mask.nonzero(as_tuple=True)

        return match_labels, matched_qry_ids, matched_tgt_ids
