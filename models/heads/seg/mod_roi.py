"""
Modular RoI (ModRoI) head.
"""

from models.build import build_model, MODELS
from models.heads.seg.base import BaseSegHead


@MODELS.register_module()
class ModRoIHead(BaseSegHead):
    """
    Class implementing the ModRoIHead module.

    Attributes:
        cls_agn_masks (bool): Boolean indicating whether computation of segmentation masks is class-agnostic.
        roi_ext (nn.Module): Module extracting the initial RoI features.
        mask_logits (nn.Module): Module computing the mask logits at RoI resolution.
        roi_paster (nn.Module): Module pasting the mask scores inside the RoI boxes.
    """

    def __init__(self, cls_agn_masks, roi_ext_cfg, mask_logits_cfg, roi_paster_cfg, **kwargs):
        """
        Initializes the ModRoIHead module.

        Args:
            cls_agn_masks (bool): Boolean indicating whether computation of segmentation masks is class-agnostic.
            roi_ext_cfg (Dict): Configuration dictionary specifying the RoI extractor module.
            mask_logits_cfg (Dict): Configuration dictionary specifying the mask logits module.
            roi_paster_cfg (Dict): Configuration dictionary specifying the RoI paster module.
            kwargs (Dict): Dictionary of keyword arguments passed to the BaseSegHead __init__ method.
        """

        # Initialization of BaseSegHead module
        super().__init__(**kwargs)

        # Build modules
        self.roi_ext = build_model(roi_ext_cfg)
        self.mask_logits = build_model(mask_logits_cfg)
        self.roi_paster = build_model(roi_paster_cfg)

        # Set additional attribute
        self.cls_agn_masks = cls_agn_masks

    def get_mask_scores(self, pred_qry_ids, pred_labels, storage_dict, **kwargs):
        """
        Method computing the segmentation mask scores at image resolution.

        Args:
            pred_qry_ids (LongTensor): Query indices of predictions of shape [num_preds].
            pred_labels (LongTensor): Class indices of predictions of shape [num_preds].

            storage_dict (Dict): Storage dictionary containing at least following key:
                - images (Images): Images structure containing the batched images of size [batch_size];
                - batch_ids (LongTensor): batch indices of queries of shape [num_qrys];
                - qry_feats (FloatTensor): query features of shape [num_qrys, qry_feat_size];
                - qry_boxes (Boxes): 2D bounding boxes of queries of size [num_qrys].

            kwargs (Dict): Dictionary of keyword arguments passed to some underlying methods.

        Returns:
            mask_scores (FloatTensor): Segmentation mask scores of shape [num_preds, iH, iW].
        """

        # Avoid computing same mask multiple times if masks are class-agnostic
        if self.cls_agn_masks:
            pred_qry_ids, unique_inv_ids = pred_qry_ids.unique(sorted=True, return_inverse=True)

        # Get local storage dictionary
        local_storage_dict = storage_dict.copy()

        # Add RoI-based keys to local storage dictionary
        local_storage_dict['roi_batch_ids'] = storage_dict['batch_ids'][pred_qry_ids]
        local_storage_dict['roi_qry_feats'] = storage_dict['qry_feats'][pred_qry_ids]
        local_storage_dict['roi_labels'] = pred_labels
        local_storage_dict['roi_boxes'] = storage_dict['qry_boxes'][pred_qry_ids]

        # Add additional key to local storage dictionary
        local_storage_dict['is_inference'] = True

        # Extract RoI features
        self.roi_ext(local_storage_dict)

        # Get mask logits at RoI resolution
        self.mask_logits(local_storage_dict)
        mask_logits = local_storage_dict.pop('mask_logits')

        # Get mask scores at RoI resolution
        mask_scores = mask_logits.sigmoid()
        local_storage_dict['mask_scores'] = mask_scores

        # Get mask scores at image resolution
        self.roi_paster(local_storage_dict)
        mask_scores = local_storage_dict.pop('mask_scores')

        # Get mask scores for each prediction
        if self.cls_agn_masks:
            mask_scores = mask_scores[unique_inv_ids]

        return mask_scores

    def compute_losses(self, storage_dict, tgt_dict, loss_dict, analysis_dict=None, id=None, **kwargs):
        """
        Method computing the losses and collecting analysis metrics.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - batch_ids (LongTensor): batch indices of queries of shape [num_qrys];
                - qry_feats (FloatTensor): query features of shape [num_qrys, qry_feat_size];
                - qry_boxes (Boxes): 2D bounding boxes of queries of size [num_qrys].
                - matched_qry_ids (LongTensor): indices of matched queries of shape [num_pos_qrys];
                - matched_tgt_ids (LongTensor): indices of corresponding matched targets of shape [num_pos_qrys].

            tgt_dict (Dict): Target dictionary containing at least following key:
                - labels (LongTensor): target class indices of shape [num_targets].

            loss_dict (Dict): Dictionary containing different weighted loss terms.
            analysis_dict (Dict): Dictionary containing different analysis metrics (default=None).
            id (int or str): Integer or string containing the head id (default=None).
            kwargs (Dict): Dictionary of keyword arguments passed to some underlying modules.

        Returns:
            loss_dict (Dict): Loss dictionary (possibly) containing additional weighted loss terms.
            analysis_dict (Dict): Analysis dictionary (possibly) containing additional analysis metrics (or None).
        """

        # Retrieve desired items from storage dictionary
        matched_qry_ids = storage_dict['matched_qry_ids']
        matched_tgt_ids = storage_dict['matched_tgt_ids']

        # Get local storage dictionary
        local_storage_dict = storage_dict.copy()

        # Add RoI-based keys to local storage dictionary
        local_storage_dict['roi_batch_ids'] = storage_dict['batch_ids'][matched_qry_ids]
        local_storage_dict['roi_qry_feats'] = storage_dict['qry_feats'][matched_qry_ids]
        local_storage_dict['roi_labels'] = tgt_dict['labels'][matched_tgt_ids]
        local_storage_dict['roi_boxes'] = storage_dict['qry_boxes'][matched_qry_ids]

        # Add additional key to local storage dictionary
        local_storage_dict['is_training'] = True

        # Extract RoI features
        self.roi_ext(local_storage_dict)

        # Get empty local loss and analysis dictionaries
        local_loss_dict = {}
        local_analysis_dict = {}

        # Get losses and analysis metrics
        kwargs.update({'tgt_dict': tgt_dict, 'loss_dict': local_loss_dict, 'analysis_dict': local_analysis_dict})
        self.mask_logits(local_storage_dict, **kwargs)

        # Update local keys with head id if needed
        if id is not None:
            local_loss_dict = {f'{k}_{id}': v for k, v in local_loss_dict.items()}
            local_analysis_dict = {f'{k}_{id}': v for k, v in local_analysis_dict.items()}

        # Merge local dictionaries with global ones
        loss_dict.update(local_loss_dict)
        analysis_dict.update(local_analysis_dict)

        return loss_dict, analysis_dict
