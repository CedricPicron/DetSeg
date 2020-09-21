"""
Matcher modules and build function.
"""
from scipy.optimize import linear_sum_assignment as lsa
import torch
from torch import nn

from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are unmatched (and thus treated as non-objects).

    Attributes:
        cost_class: Relative weight of the classification error in the matching cost.
        cost_bbox: Relative weight of the L1 error of the bounding box coordinates in the matching cost.
        cost_giou: Relative weight of the GIoU loss of the bounding box in the matching cost.
    """

    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 1.0, cost_giou: float = 1.0):
        """
        Initializes the HungarianMatcher module.

        Args:
            cost_class: Relative weight of the classification error in the matching cost.
            cost_bbox: Relative weight of the L1 error of the bounding box coordinates in the matching cost.
            cost_giou: Relative weight of the giou loss of the bounding box in the matching cost.

        Raises:
            ValueError: Error when all cost coefficients are zero.
        """

        if cost_class == 0 and cost_bbox == 0 and cost_giou == 0:
            raise ValueError("All cost coefficients can't be zero.")

        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @staticmethod
    def get_sizes(batch_idx, batch_size):
        """
        Computes the cumulative number of predictions across batch entries.

        Args:
            batch_idx (IntTensor): Tensor of shape [num_slots] with batch indices of slots (in ascending order);
            batch_size (int): Total number of images in batch.
        """

        prev_idx = 0
        sizes = [0]

        for i, curr_idx in enumerate(batch_idx):
            if curr_idx != prev_idx:
                sizes.extend([i] * (curr_idx-prev_idx))
                prev_idx = curr_idx

        current_length = len(sizes)
        sizes.extend([len(batch_idx)] * (batch_size+1 - current_length))

        return sizes

    @torch.no_grad()
    def forward(self, pred_dict, tgt_list):
        """
        Forward method of the HungarianMatcher module. Performs the hungarian matching.

        Args:
            pred_dict (Dict): Dictionary containing at least following keys:
                 - logits (FloatTensor): tensor of shape [num_slots, num_classes] with the classification logits;
                 - boxes (FloatTensor): tensor of shape [num_slots, 4] with the predicted box coordinates.
                 - batch_idx (IntTensor): tensor of shape [num_slots] with batch indices of slots (ascending order);

            tgt_list (List): List of targets of shape[batch_size], where each entry is a dict containing the keys:
                 - labels (IntTensor): tensor of dim [num_target_boxes] (where num_target_boxes is the number of
                                       ground-truth objects in the target) containing the ground-truth class indices;
                 - boxes (FloatTensor): tensor of dim [num_target_boxes, 4] containing the target box coordinates.

        Returns:
            idx: List of shape [batch_size], containing tuples (pred_idx, tgt_idx) with:
                - pred_idx (IntTensor): tensor of chosen predictions of shape [min(num_slots_batch, num_target_boxes)];
                - tgt_idx (IntTensor): tensor of matching targets of shape [min(num_slots_batch, num_target_boxes)].

        Raises:
            ValueError: Raised when predictions['batch_idx'] is not sorted in ascending order.
        """

        # Check whether batch_idx is sorted
        batch_idx = pred_dict['batch_idx']
        if len(batch_idx) > 1:
            if not ((batch_idx[1:]-batch_idx[:-1]) >= 0).all():
                raise ValueError("pred_dict['batch_idx'] must be sorted in ascending order.")

        # Compute class probablities of predictions
        pred_prob = pred_dict['logits'].softmax(-1)
        pred_bbox = pred_dict['boxes']

        # Concatenate the target class indices and boxes accros batch entries
        tgt_ids = torch.cat([t['labels'] for t in tgt_list])
        tgt_bbox = torch.cat([t['boxes'] for t in tgt_list])

        # Compute the classification cost. Contrary to the criterion loss, we don't use the NLL, but approximate it
        # by 1 - probability[target class]. The 1 is omitted, as the constant doesn't change the matching.
        cost_class = -pred_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(pred_bbox, tgt_bbox, p=1)

        # Compute the GIoU cost between boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(pred_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Weighted cost matrix
        C = (self.cost_bbox*cost_bbox + self.cost_class*cost_class + self.cost_giou*cost_giou).cpu()

        # Compute number of predictions and targets in each batch
        batch_size = len(tgt_list)
        pred_sizes = self.get_sizes(batch_idx, batch_size)
        tgt_sizes = [0].extend([len(t['boxes']) for t in tgt_list])
        tgt_sizes = torch.cumsum(torch.tensor(tgt_sizes), dim=0)

        # Match predictions with targets
        idx = [lsa(C[pred_sizes[i]:pred_sizes[i+1], tgt_sizes[i]:tgt_sizes[i+1]]) for i in range(batch_size)]
        idx = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in idx]

        return idx


def build_matcher(args):
    """
    Build matcher from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        matcher (HungarianMatcher): The specified HungarianMatcher module.
    """

    matcher = HungarianMatcher(args.match_coef_class, args.match_coef_bbox, args.match_coef_giou)

    return matcher
