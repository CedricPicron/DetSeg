"""
Collection of matcher modules.
"""

import torch
from torch import nn

from models.build import build_model, MODELS
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
        qry_key (str): String with key to retrieve the queries boxes from the storage dictionary.
        tgt_key (str): String with key to retrieve the target boxes from the target dictionary.
        box_metric (str): String containing the metric computing similarities between query and target boxes.
        sim_matcher (nn.Module): Module matching queries with targets based on the given similarity matrix.
    """

    def __init__(self, qry_key, sim_matcher_cfg, tgt_key='boxes', box_metric='iou'):
        """
        Initializes the BoxMatcher module.

        Args:
            qry_key (str): String with key to retrieve the queries boxes from the storage dictionary.
            sim_matcher_cfg (Dict): Configuration dictionary specifying the similarity matcher.
            tgt_key (str): String with key to retrieve the target boxes from the target dictionary (default='boxes').
            box_metric (str): String with metric computing similarities between query and target boxes (default='iou').
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build similarity matcher
        self.sim_matcher = build_model(sim_matcher_cfg)

        # Set remaining attributes
        self.qry_key = qry_key
        self.tgt_key = tgt_key
        self.box_metric = box_metric

    def forward(self, storage_dict, tgt_dict, **kwargs):
        """
        Forward method of BoxMatcher module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following key:
                - {self.qry_key} (Boxes): structure containing axis-aligned query boxes of size [num_queries].

            tgt_dict (Dict): Target dictionary containing at least following key:
                - {self.tgt_key} (Boxes): structure containing axis-aligned target boxes of size [num_targets].

            kwargs (Dict): Dictionary of keyword arguments passed to some underlying modules.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional keys:
                - match_labels (LongTensor): match labels corresponding to each query of shape [num_queries];
                - matched_qry_ids (LongTensor): indices of matched queries of shape [num_pos_queries];
                - matched_tgt_ids (LongTensor): indices of corresponding matched targets of shape [num_pos_queries].

        Raises:
            ValueError: Error when an invalid bounding box metric is provided.
        """

        # Retrieve query and target boxes
        qry_boxes = storage_dict[self.qry_key]
        tgt_boxes = tgt_dict[self.tgt_key]

        # Get device
        device = qry_boxes.boxes.device

        # Get list of query and target boxes per image
        qry_boxes = qry_boxes.split(qry_boxes.boxes_per_img.tolist())
        tgt_boxes = tgt_boxes.split(tgt_boxes.boxes_per_img.tolist())

        if len(qry_boxes) == 1 and len(tgt_boxes) > 1:
            qry_boxes = qry_boxes * len(tgt_boxes)

        # Initialize query and target offsets
        qry_offset = 0
        tgt_offset = 0

        # Intialize lists with matching results
        match_labels_list = []
        matched_qry_ids_list = []
        matched_tgt_ids_list = []

        # Perform matching per image
        for qry_boxes_i, tgt_boxes_i in zip(qry_boxes, tgt_boxes):

            # Get number of query and target boxes
            num_queries = len(qry_boxes_i)
            num_targets = len(tgt_boxes_i)

            # Case where there are both queries and targets
            if (num_queries > 0) and (num_targets > 0):

                # Get similarity matrix between query and target boxes
                if self.box_metric == 'giou':
                    sim_matrix = box_giou(qry_boxes_i, tgt_boxes_i)

                elif self.box_metric == 'iou':
                    sim_matrix = box_iou(qry_boxes_i, tgt_boxes_i)

                else:
                    error_msg = f"Invalid bounding box metric (got '{self.box_metric}')."
                    raise ValueError(error_msg)

                # Perform matching based on similarity matrix
                match_labels_i, matched_qry_ids_i, matched_tgt_ids_i = self.sim_matcher(sim_matrix)

            # Case where there are no queries or targets
            else:
                match_labels_i = torch.zeros(num_queries, dtype=torch.int64, device=device)
                matched_qry_ids_i = torch.zeros(0, dtype=torch.int64, device=device)
                matched_tgt_ids_i = torch.zeros(0, dtype=torch.int64, device=device)

            # Add query and target offsets to indices
            matched_qry_ids_i = matched_qry_ids_i + qry_offset
            matched_tgt_ids_i = matched_tgt_ids_i + tgt_offset

            # Update query and target offsets
            qry_offset += num_queries
            tgt_offset += num_targets

            # Add image-specific mathcing resutls to list
            match_labels_list.append(match_labels_i)
            matched_qry_ids_list.append(matched_qry_ids_i)
            matched_tgt_ids_list.append(matched_tgt_ids_i)

        # Concatenate matching results and add them to storage dictionary
        storage_dict['match_labels'] = torch.cat(match_labels_list, dim=0)
        storage_dict['matched_qry_ids'] = torch.cat(matched_qry_ids_list, dim=0)
        storage_dict['matched_tgt_ids'] = torch.cat(matched_tgt_ids_list, dim=0)

        return storage_dict


@MODELS.register_module()
class SimMatcher(nn.Module):
    """
    Class implementing the SimMatcher module.

    The module matches queries with targets based on the given query-target similarity matrix.

    The module assigns one of following match labels to each query:
        - positive label with value 1 (query matches with at least one target);
        - negative label with value 0 (query does not match with any of the targets);
        - ignore label with value -1 (query has no matching verdict).

    Attributes:
        mode (str): String containing the matching mode.
        static_mode (str): String containing the static matching mode.
        abs_pos (float): Absolute threshold determining positive query labels during static matching.
        abs_neg (float): Absolute threshold determining negative query labels during static matching.
        rel_pos (int): Relative threshold determining positive query labels during static matching.
        rel_neg (int): Relative threshold determining negative query labels during static matching.
        multi_tgt (bool): Boolean indicating whether queries can be matched with multiple targets.
    """

    def __init__(self, mode='static', static_mode='rel', abs_pos=0.5, abs_neg=0.3, rel_pos=5, rel_neg=10,
                 multi_tgt=True):
        """
        Initializes the SimMatcher module.

        Args:
            mode (str): String containing the matching mode (default='static').
            static_mode (str): String containing the static matching mode (default='rel').
            abs_pos (float): Absolute threshold determining positive labels during static matching (default=0.5).
            abs_neg (float): Absolute threshold determining negative labels during static matching (default=0.3).
            rel_pos (int): Relative threshold determining positive labels during static matching (default=5).
            rel_neg (int): Relative threshold determining negative labels during static matching (default=10).
            multi_tgt (bool): Boolean indicating whether queries can be matched with multiple targets (default=True).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attributes
        self.mode = mode
        self.static_mode = static_mode
        self.abs_pos = abs_pos
        self.abs_neg = abs_neg
        self.rel_pos = rel_pos
        self.rel_neg = rel_neg
        self.multi_tgt = multi_tgt

    def forward(self, sim_matrix):
        """
        Forward method of SimMatcher module.

        Args:
            sim_matrix (FloatTensor): Query-target similarity matrix of shape [num_queries, num_targets].

        Returns:
            match_labels (LongTensor): Match labels corresponding to each query of shape [num_queries].
            matched_qry_ids (LongTensor): Indices of matched queries of shape [num_pos_queries].
            matched_tgt_ids (LongTensor): Indices of corresponding matched targets of shape [num_pos_queries].

        Raises:
            ValueError: Error when an invalid static matching mode is provided.
            ValueError: Error when an invalid matching mode is provided.
        """

        # Get number of queries and targets
        num_queries, num_targets = sim_matrix.size()

        # Get device
        device = sim_matrix.device

        # Handle case where there are no queries or targets and return
        if (num_queries == 0) or (num_targets == 0):
            match_labels = torch.zeros(num_queries, dtype=torch.int64, device=device)
            matched_qry_ids = torch.zeros(0, dtype=torch.int64, device=device)
            matched_tgt_ids = torch.zeros(0, dtype=torch.int64, device=device)

            return match_labels, matched_qry_ids, matched_tgt_ids

        # Get positive and negative label masks
        if self.mode == 'static':

            if 'abs' in self.static_mode:
                abs_pos_mask = sim_matrix >= self.abs_pos
                abs_neg_mask = sim_matrix < self.abs_neg

            if 'rel' in self.static_mode:
                non_neg_ids = torch.topk(sim_matrix, self.rel_neg, dim=0, sorted=True).indices
                pos_ids = non_neg_ids[:self.rel_pos, :]

                rel_pos_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
                rel_pos_mask[pos_ids, torch.arange(num_targets)] = True

                rel_neg_mask = torch.ones_like(sim_matrix, dtype=torch.bool)
                rel_neg_mask[non_neg_ids, torch.arange(num_targets)] = False

            if self.static_mode == 'abs':
                pos_mask = abs_pos_mask
                neg_mask = abs_neg_mask

            elif self.static_mode == 'abs_and_rel':
                pos_mask = abs_pos_mask & rel_pos_mask
                neg_mask = abs_neg_mask | rel_neg_mask

            elif self.static_mode == 'abs_or_rel':
                pos_mask = abs_pos_mask | rel_pos_mask
                neg_mask = abs_neg_mask & rel_neg_mask

            elif self.static_mode == 'rel':
                pos_mask = rel_pos_mask
                neg_mask = rel_neg_mask

            else:
                error_msg = f"Invalid static matching mode (got {self.static_mode})."
                raise ValueError(error_msg)

        else:
            error_msg = f"Invalid matching mode (got {self.mode})."
            raise ValueError(error_msg)

        # Remove matches if queries cannot be matched with multiple targets
        if not self.multi_tgt:
            best_qry_ids = torch.arange(num_queries, device=device)
            best_tgt_ids = torch.argmax(pos_mask.to(torch.float) * sim_matrix, dim=1)

            best_tgt_mask = torch.zeros_like(pos_mask)
            best_tgt_mask[best_qry_ids, best_tgt_ids] = True
            pos_mask = pos_mask & best_tgt_mask

        # Get match labels
        match_labels = torch.full(size=(num_queries,), fill_value=-1, device=device)
        match_labels[pos_mask.sum(dim=1) > 0] = 1
        match_labels[neg_mask.sum(dim=1) == num_targets] = 0

        # Get query and target indices of matches
        matched_qry_ids, matched_tgt_ids = pos_mask.nonzero(as_tuple=True)

        return match_labels, matched_qry_ids, matched_tgt_ids
