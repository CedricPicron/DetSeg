"""
Graph-Connecting Trees (GCT) architecture.
"""

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from models.build import build_model, MODELS
from models.functional.graph import map_to_graph


@MODELS.register_module()
class GCT(nn.Module):
    """
    Class implementing the Graph-Connecting Trees (GCT) architecture.

    Attributes:
        map (nn.Module): Module performing the initial map-based processing.
        struc (nn.Module): Module computing the initial structure features from normalized node locations.
        graph (nn.Module): Module performing the subsequent graph-based processing.
        heads (nn.ModuleDict): Dictionary of head dictionaries containing sub-head modules.
    """

    def __init__(self, map_cfg, graph_cfg, heads):
        """
        Initializes the GCT module.

        Args:
            map_cfg (Dict): Configuration dictionary specifying the initial map-based processing module.
            graph_cfg (Dict): Configuration dictionary specifying the subsequent graph-based processing module.
            heads (Dict): Dictionary of head dictionaries containing sub-head configuration dictionaries.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build initial map-based processing module
        self.map = build_model(map_cfg)

        # Build subsequent graph-based processing module
        self.graph = build_model(graph_cfg)

        # Build sub-head modules
        self.heads = nn.ModuleDict()

        for head_name, head_dict in heads.items():
            self.heads[head_name] = nn.ModuleDict()

            for cfg_name, sub_head_cfg in head_dict.items():
                sub_head_name = cfg_name.removesuffix('_cfg')
                self.heads[head_name][sub_head_name] = build_model(sub_head_cfg)

    @staticmethod
    def get_param_families():
        """
        Method returning the GCT parameter families.

        Returns:
            List of strings containing the GCT parameter families.
        """

        return ['map', 'graph', 'heads']

    def forward(self, images, tgt_dict=None, optimizer=None, max_grad_norm=-1, visualize=False, **kwargs):
        """
        Forward method of the GCT module.

        Args:
            images (Images): Images structure containing the batched images.
            tgt_dict (Dict): Target dictionary with ground-truth information used during trainval (default=None).
            optimizer (torch.optim.Optimizer): Optimizer updating the GCT parameters during training (default=None).
            max_grad_norm (float): Maximum gradient norm of parameters throughout model (default=-1).
            visualize (bool): Boolean indicating whether to compute dictionary with visualizations (default=False).
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            output_dicts (List): List of output dictionaries, potentially containing following items:
                - pred_dicts (List): list of dictionaries with predictions;
                - loss_dict (Dict): dictionary of different loss terms used for backpropagation during training;
                - analysis_dict (Dict): dictionary of different analyses used for logging purposes only;
                - images_dict (Dict): dictionary of different annotated images based on predictions and targets.

        Raises:
            RuntimeError: Error when an optimizer is provided, but no target dictionary.
            ValueError: Error when visualizations are requested.
            ValueError: Error when the GCT heads dictionary contains an unknown head type.
        """

        # Check inputs
        if tgt_dict is None and optimizer is not None:
            error_msg = "An optimizer is provided, but no target dictionary to learn from."
            raise RuntimeError(error_msg)

        if visualize:
            error_msg = "The GCT architecture currently does not provide visualizations."
            raise ValueError(error_msg)

        # Initialize empty list for prediction dictionaries if needed
        if not self.training:
            pred_dicts = []

        # Initialize empty loss dictionary if needed
        if tgt_dict is not None:
            loss_dict = {}

        # Initialize empty analysis dictionary
        analysis_dict = {}

        # Perform map-based processing
        feat_map = self.map(images)
        feat_map = feat_map[-1] if isinstance(feat_map, (list, tuple)) else feat_map

        # Get initial graph from feature map
        graph = map_to_graph(feat_map)
        graph['graph_id'] = 0
        graph['con_feats'] = graph.pop('node_feats')
        analysis_dict['num_nodes_0'] = len(graph['con_feats'])

        # Get initial structure features
        node_xy = graph.pop('node_xy')
        graph['struc_feats'] = node_xy

        # Get initial edge weights
        num_edges = graph['edge_ids'].size(dim=1)
        graph['edge_weights'] = torch.ones(num_edges, dtype=torch.float, device=feat_map.device)

        # Get initial graph-based segmentation maps
        batch_size, _, fH, fW = feat_map.size()
        graph['seg_maps'] = torch.arange(batch_size * fH * fW, device=feat_map.device).view(batch_size, fH, fW)

        # Add analysis dictionary to graph dictionary
        graph['analysis_dict'] = analysis_dict

        # Get final graph
        graph = self.graph(graph)

        # Unpack final graph dictionary
        struc_feats = graph['struc_feats']
        graph_seg_maps = graph['seg_maps']
        node_batch_ids = graph['node_batch_ids']
        analysis_dict = graph['analysis_dict']

        # Get useful final graph properties
        num_nodes = len(struc_feats)
        img_node_xy = node_xy.view(batch_size, -1, 2)[0]
        _, nodes_per_img = torch.unique_consecutive(node_batch_ids, return_counts=True)
        cum_nodes_per_img = torch.tensor([0, *nodes_per_img.cumsum(dim=0)])

        # Apply the various heads
        for head_type, head_dict in self.heads.items():

            # Apply graph-segmentation head
            if head_type == 'graph_seg':

                # Get prediction maps
                struc_seg_feats = head_dict['struc'](struc_feats)
                pos_feats = head_dict['pos'](img_node_xy)
                pred_maps = torch.mm(struc_seg_feats, pos_feats.t()).view(num_nodes, fH, fW)

                # Get target masks
                tgt_masks = []

                for i in range(batch_size):
                    graph_seg_map = graph_seg_maps[i]
                    range_obj = range(cum_nodes_per_img[i], cum_nodes_per_img[i+1])

                    tgt_masks_i = torch.stack([graph_seg_map == j for j in range_obj], dim=0)
                    tgt_masks.append(tgt_masks_i)

                tgt_masks = torch.cat(tgt_masks, dim=0)

                # Get graph-segmentation loss
                loss = head_dict['loss'](pred_maps, tgt_masks, reduction='sum')

                if tgt_dict is not None:
                    loss_dict[f'{head_type}_loss'] = loss
                else:
                    analysis_dict[f'{head_type}_loss'] = loss

                # Get graph-segmentation accuracy
                pred_masks = pred_maps >= 0
                accuracy = torch.eq(pred_masks, tgt_masks).sum() / pred_masks.numel()
                analysis_dict[f'{head_type}_acc'] = 100 * accuracy

            else:
                error_msg = f"The GCT heads dictionary contains an unknown head type '{head_type}'."
                raise ValueError(error_msg)

        # Update model parameters with given optimizer
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

            loss = sum(loss_dict.values())
            loss.backward()

            if max_grad_norm > 0:
                clip_grad_norm_(self.parameters(), max_grad_norm)

            optimizer.step()

        # Get list of output dictionaries
        output_dicts = []
        output_dicts.append(pred_dicts) if not self.training else None
        output_dicts.append(loss_dict) if tgt_dict is not None else None
        output_dicts.append(analysis_dict)

        return output_dicts
