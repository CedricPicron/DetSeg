"""
Matcher modules and build function.
"""
from scipy.optimize import linear_sum_assignment as lsa
import torch
from torch import nn

from structures.boxes import box_giou


class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are unmatched (and thus treated as non-objects).

    Attributes:
        cost_class: Relative weight of the classification loss in the matching cost.
        cost_l1: Relative weight of the L1 bounding box loss in the matching cost.
        cost_giou: Relative weight of the GIoU bounding box loss in the matching cost.
    """

    def __init__(self, cost_class: float = 1.0, cost_l1: float = 1.0, cost_giou: float = 1.0):
        """
        Initializes the HungarianMatcher module.

        Args:
            cost_class: Relative weight of the classification loss in the matching cost.
            cost_l1: Relative weight of the L1 bounding box loss in the matching cost.
            cost_giou: Relative weight of the GIoU bounding box loss in the matching cost.

        Raises:
            ValueError: Error when all cost coefficients are zero.
        """

        if cost_class == 0 and cost_l1 == 0 and cost_giou == 0:
            raise ValueError("All cost coefficients can't be zero.")

        super().__init__()
        self.cost_class = cost_class
        self.cost_l1 = cost_l1
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, out_dict, tgt_dict):
        """
        Forward method of the HungarianMatcher module. Performs the hungarian matching.

        Args:
            out_dict (Dict): Output dictionary containing at least following keys:
                - logits (FloatTensor): classification logits of shape [num_slots_total, num_classes];
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_slots_total];
                - sizes (LongTensor): cumulative number of predictions across batch entries of shape [batch_size+1].

            tgt_dict (Dict): Target dictionary containing at least following keys:
                - labels (LongTensor): tensor of shape [num_targets_total] containing the class indices;
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_slots_total];
                - sizes (LongTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries.

        Returns:
            pred_idx (LongTensor): Chosen predictions of shape [sum(min(num_slots_batch, num_targets_batch))].
            tgt_idx (LongTensor): Matching targets of shape [sum(min(num_slots_batch, num_targets_batch))].
        """

        # Compute class probablities of predictions
        pred_probs = out_dict['logits'].softmax(dim=-1)

        # Some renaming for code readability
        pred_boxes = out_dict['boxes']
        tgt_labels = tgt_dict['labels']
        tgt_boxes = tgt_dict['boxes']

        # Compute the classification cost. Contrary to the criterion loss, we don't use the NLL, but approximate it
        # by 1 - probability[target class]. The 1 is omitted, as the constant doesn't change the matching.
        cost_class = -pred_probs[:, tgt_labels]

        # Compute the L1 cost between boxes in (center_x, center_y, width, height) format
        pred_boxes = pred_boxes.to_format('cxcywh')
        tgt_boxes = tgt_boxes.to_format('cxcywh')
        cost_l1 = torch.cdist(pred_boxes.boxes, tgt_boxes.boxes, p=1)

        # Compute the GIoU cost between boxes
        cost_giou = -box_giou(pred_boxes, tgt_boxes)

        # Weighted cost matrix
        C = (self.cost_class*cost_class + self.cost_l1*cost_l1 + self.cost_giou*cost_giou).cpu()

        # Compute cumulative number of predictions and targets in each batch entry
        batch_size = len(tgt_dict['sizes'])-1
        pred_sizes = out_dict['sizes'].tolist()
        tgt_sizes = tgt_dict['sizes'].tolist()

        # Match predictions with targets
        lsa_idx = [lsa(C[pred_sizes[i]:pred_sizes[i+1], tgt_sizes[i]:tgt_sizes[i+1]]) for i in range(batch_size)]
        pred_idx = torch.cat([torch.as_tensor(pred_sizes[i]+j, dtype=torch.int64) for i, (j, _) in enumerate(lsa_idx)])
        tgt_idx = torch.cat([torch.as_tensor(tgt_sizes[i]+j, dtype=torch.int64) for i, (_, j) in enumerate(lsa_idx)])

        return pred_idx, tgt_idx


def build_matcher(args):
    """
    Build matcher from command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        matcher (HungarianMatcher): The specified HungarianMatcher module.
    """

    matcher = HungarianMatcher(args.match_coef_class, args.match_coef_l1, args.match_coef_giou)

    return matcher
