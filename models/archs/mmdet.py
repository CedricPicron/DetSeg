"""
Module building cores from MMDetection.
"""

from mmcv import Config
from mmdet.models import build_detector as build_arch
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from structures.boxes import Boxes


class MMDetArch(nn.Module):
    """
    Class implementing the MMDetArch module.

    Attributes:
        requires_masks (bool): Boolean indicating whether target masks are required during training.
        arch (nn.Module): Module containing the MMDetection architecture.
    """

    def __init__(self, backbone, core, cfg_path):
        """
        Initializes the MMDetArch module.

        Args:
            backbone (nn.Module): Module implementing the backbone.
            core (nn.Module): Module implementing the core.
            cfg_path (str): Path to configuration file specifying the MMDetection architecture.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Get config specifying the MMDetection architecture
        cfg = Config.fromfile(cfg_path)

        # Set attribute indicating whether target masks are required during training
        self.requires_masks = cfg.model.pop('requires_masks')

        # Get architecture
        self.arch = build_arch(cfg.model)
        self.arch.init_weights()

        # Replace backbone and core/neck with given input modules
        self.arch.backbone = backbone
        self.arch.neck = core

    @staticmethod
    def get_param_families():
        """
        Method returning the recurring MMDetArch parameter families.

        Returns:
            List of strings containing the recurring MMDetArch parameter families.
        """

        return ['backbone', 'neck']

    def forward(self, images, tgt_dict=None, optimizer=None, max_grad_norm=-1, visualize=False, **kwargs):
        """
        Forward method of the MMDetArch module.

        Args:
            images (Images): Images structure containing the batched images.

            tgt_dict (Dict): Optional target dictionary used during trainval containing at least following keys:
                - labels (LongTensor): tensor of shape [num_targets_total] containing the class indices;
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_targets_total];
                - sizes (LongTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries.

            optimizer (torch.optim.Optimizer): Optimizer updating the model parameters during training (default=None).
            max_grad_norm (float): Maximum gradient norm of parameters throughout model (default=-1).
            visualize (bool): Boolean indicating whether to compute dictionary with visualizations (default=False).
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            out_dicts (List): List of output dictionaries, potentially containing following items:
                - pred_dicts (List): list of dictionaries with predictions;
                - loss_dict (Dict): dictionary of different loss terms used for backpropagation during training;
                - analysis_dict (Dict): dictionary of different analyses used for logging purposes only.

        Raises:
            ValueError: Error when visualizations are requested.
        """

        # Check whether 'visualize' is False
        if visualize:
            raise ValueError("The MMDetArch architecture does not provide visualizations.")

        # Get MMDetection image metas
        batch_size = len(images)
        img_metas = [{} for _ in range(batch_size)]

        img_shapes = [(iH, iW, 3) for iW, iH in images.size(mode='without_padding')]
        scale_factors = images.resize_ratios()
        hflipped = images.hflipped()
        padded_width, padded_height = images.size(mode='with_padding')
        orig_img_shapes = [(iH, iW, 3) for iW, iH in images.size(mode='original')]

        for i in range(batch_size):
            img_metas[i]['img_shape'] = img_shapes[i]
            img_metas[i]['scale_factor'] = [*scale_factors[i], *scale_factors[i]]
            img_metas[i]['flip'] = hflipped[i]
            img_metas[i]['pad_shape'] = (padded_height, padded_width, 3)
            img_metas[i]['ori_shape'] = orig_img_shapes[i]

        # Get loss and analysis dictionaries if desired
        if tgt_dict is not None:
            tgt_boxes = tgt_dict['boxes']
            tgt_boxes = tgt_boxes.to_format('xyxy').to_img_scale(images).boxes

            tgt_sizes = tgt_dict['sizes']
            tgt_boxes = [tgt_boxes[i0:i1] for i0, i1 in zip(tgt_sizes[:-1], tgt_sizes[1:])]
            tgt_labels = [tgt_dict['labels'][i0:i1] for i0, i1 in zip(tgt_sizes[:-1], tgt_sizes[1:])]

            arch_kwargs = {}
            arch_kwargs['gt_bboxes'] = tgt_boxes
            arch_kwargs['gt_labels'] = tgt_labels

            out_dict = self.arch(images, img_metas, return_loss=True, **arch_kwargs)
            out_dict.update({k: sum(v) for k, v in out_dict.items() if isinstance(v, list)})

            loss_dict = {k: v for k, v in out_dict.items() if 'loss' in k}
            analysis_dict = {k: v for k, v in out_dict.items() if 'loss' not in k}

        # Get prediction dictionaries if desired
        if not self.training:
            preds = self.arch([images], [img_metas], return_loss=False)
            preds = [[torch.as_tensor(cls_preds) for cls_preds in preds_i] for preds_i in preds]

            num_classes = len(preds[0])
            pred_keys = ('labels', 'boxes', 'scores', 'batch_ids')
            pred_dict = {pred_key: [] for pred_key in pred_keys}

            for i, preds_i in enumerate(preds):
                cls_freqs = torch.tensor([len(cls_preds) for cls_preds in preds_i])
                labels = torch.arange(num_classes).repeat_interleave(cls_freqs, dim=0)
                batch_ids = torch.full_like(labels, i)

                cat_preds = torch.cat(preds_i, dim=0)
                boxes = Boxes(cat_preds[:, :4], format='xyxy')
                scores = cat_preds[:, 4]

                pred_dict['labels'].append(labels)
                pred_dict['boxes'].append(boxes)
                pred_dict['scores'].append(scores)
                pred_dict['batch_ids'].append(batch_ids)

            pred_dict.update({k: torch.cat(v, dim=0) for k, v in pred_dict.items() if k != 'boxes'})
            pred_dict['boxes'] = Boxes.cat(pred_dict['boxes'])
            pred_dicts = [pred_dict]

        # Update model parameters during training
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

            loss = sum(loss_dict.values())
            loss.backward()

            clip_grad_norm_(self.parameters(), max_grad_norm) if max_grad_norm > 0 else None
            optimizer.step()

        # Get architecture output dictionaries
        if self.training:
            out_dicts = [loss_dict, analysis_dict]
        elif tgt_dict is not None:
            out_dicts = [pred_dicts, loss_dict, analysis_dict]
        else:
            out_dicts = [pred_dicts]

        return out_dicts
