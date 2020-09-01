import sys

import numpy as np
import torch
from torch import nn
from torch.autograd.function import Function
import torch.nn.functional as F


class GroupDecoder(nn.Module):
    """
    Concatenation of group decoder layers.

    Attributes:
        layers (nn.ModulesList): List of group decoder layers being concatenated.
        num_layers (int): Number of concatenated group decoder layers.

    * Initialization (first iteration and layer):
        in: encoder_features -> shape = [H*W, batch_size, feature_dim]

        feat_idx_edge_idx = init_fe(H, W, batch_size) -> shape = [num_features, 4]
        edge_idx_feat_idx = init_ef(H, W, batch_size) -> shape = [num_edges, 2]

        out: features = encoder_features.view(-1, feature_dim) -> shape = [H*W*batch_size, feat_dim]
        out: graph = Graph(feat_idx_edge_idx, edge_idxe_feat_idx) -> create Graph instance

        note: num_feat = H*W*batch_size at first iteration
        note: num_edges = (2*H*W - H - W) * batch_size at first iteration
    """


class GroupDecoderLayer(nn.Module):
    """
    Submodules:
        1) Grouping
        2) Cross-attention
        3) Self-attention
        4) FFN
    """


class Graph():
    """
    Class implementing the Graph object used for the feature graphs.
    """

    def __init__(self, height, width):
        height, width = np.array(height), np.array(width)
        batch_size = len(height)

        self.num_nodes = height * width
        self.num_edges = 2*height*width - height - width
        self.num_nodes_total = sum(self.num_nodes)
        self.num_edges_total = sum(self.num_edges)

        num_row_edges = height * (width-1)
        num_col_edges = (height-1) * width
        node_offset = np.concatenate(([0], np.cumsum(self.num_nodes)[:-1]), axis=0)
        row_offset = np.concatenate(([0], np.cumsum(num_row_edges)), axis=0)
        col_offset = np.concatenate(([0], np.cumsum(num_col_edges)), axis=0)

        self.node_to_node = self.init_node_to_node(height, width, batch_size, node_offset)
        self.edge_to_node = self.init_edge_to_node(height, width, batch_size, node_offset)
        self.node_to_edge = self.init_node_to_edge(height, width, batch_size, node_offset, row_offset, col_offset)
        self.edge_to_edge = self.init_edge_to_edge(height, width, batch_size, row_offset, col_offset)

    def init_node_to_node(self, he, wi, bs, no):
        base = torch.arange(self.num_nodes_total)[:, None]
        width_full = torch.cat([wi[i] * torch.ones(self.num_nodes[i], dtype=torch.int) for i in range(bs)], dim=0)
        node_to_node = torch.cat((base-1, base-width_full[:, None], base+1, base+width_full[:, None]), dim=1)

        fill_idx_left = torch.cat([no[i] + wi[i] * torch.arange(he[i]) for i in range(bs)], dim=0)
        fill_idx_top = torch.cat([no[i] + torch.arange(wi[i]) for i in range(bs)], dim=0)
        fill_idx_right = torch.cat([no[i] + (wi[i]-1) + wi[i] * torch.arange(he[i]) for i in range(bs)], dim=0)
        fill_idx_bottom = torch.cat([no[i] + (he[i]-1)*wi[i] + torch.arange(wi[i]) for i in range(bs)], dim=0)

        node_to_node[fill_idx_left, 0] = self.num_nodes_total
        node_to_node[fill_idx_top, 1] = self.num_nodes_total
        node_to_node[fill_idx_right, 2] = self.num_nodes_total
        node_to_node[fill_idx_bottom, 3] = self.num_nodes_total

        return node_to_node

    def init_edge_to_node(self, he, wi, bs, no):
        row_node0_idx = torch.cat([no[i] + h*wi[i] + torch.arange(wi[i]-1) for i in range(bs) for h in range(he[i])])
        row_node1_idx = row_node0_idx + 1
        col_node0_idx = torch.cat([no[i] + torch.arange(wi[i]*(he[i]-1)) for i in range(bs)])
        col_node1_idx = torch.cat([no[i] + wi[i] + torch.arange(wi[i]*(he[i]-1)) for i in range(bs)])

        row_edge_to_node = torch.cat([row_node0_idx[:, None], row_node1_idx[:, None]], dim=1)
        col_edge_to_node = torch.cat([col_node0_idx[:, None], col_node1_idx[:, None]], dim=1)
        edge_to_node = torch.cat([row_edge_to_node, col_edge_to_node], dim=0)

        return edge_to_node

    def init_node_to_edge(self, he, wi, bs, no, ro, co):
        left_edge = torch.cat([ro[i] + h*(wi[i]-1)-1 + torch.arange(wi[i]) for i in range(bs) for h in range(he[i])])
        left_edge = left_edge[:, None]
        right_edge = left_edge + 1

        top_edge = torch.cat([ro[bs] + co[i] - wi[i] + torch.arange(self.num_nodes[i]) for i in range(bs)])[:, None]
        bottom_edge = torch.cat([ro[bs] + co[i] + torch.arange(self.num_nodes[i]) for i in range(bs)])[:, None]
        node_to_edge = torch.cat([left_edge, top_edge, right_edge, bottom_edge], dim=1)

        fill_idx_left = torch.cat([no[i] + wi[i] * torch.arange(he[i]) for i in range(bs)], dim=0)
        fill_idx_top = torch.cat([no[i] + torch.arange(wi[i]) for i in range(bs)], dim=0)
        fill_idx_right = torch.cat([no[i] + (wi[i]-1) + wi[i] * torch.arange(he[i]) for i in range(bs)], dim=0)
        fill_idx_bottom = torch.cat([no[i] + (he[i]-1)*wi[i] + torch.arange(wi[i]) for i in range(bs)], dim=0)

        node_to_edge[fill_idx_left, 0] = self.num_edges_total
        node_to_edge[fill_idx_top, 1] = self.num_edges_total
        node_to_edge[fill_idx_right, 2] = self.num_edges_total
        node_to_edge[fill_idx_bottom, 3] = self.num_edges_total

        return node_to_edge

    def init_edge_to_edge(self, he, wi, bs, ro, co):
        row_base = torch.arange(ro[bs])[:, None]
        col_to_col = torch.cat([wi[i] * torch.ones(he[i]*(wi[i]-1), dtype=torch.int) for i in range(bs)])

        row_left = row_base - 1
        row_topleft = torch.cat([co[i]+(h-1)*wi[i] + torch.arange(wi[i]-1) for i in range(bs) for h in range(he[i])])
        row_topleft = ro[bs] + row_topleft[:, None]
        row_topright = row_topleft + 1
        row_right = row_base + 1
        row_botright = row_topright + col_to_col[:, None]
        row_botleft = row_botright - 1

        row_fill_idx_left = torch.cat([ro[i] + (wi[i]-1) * torch.arange(he[i]) for i in range(bs)])
        row_fill_idx_topleft = torch.cat([ro[i] + torch.arange(wi[i]-1) for i in range(bs)])
        row_fill_idx_topright = row_fill_idx_topleft
        row_fill_idx_right = torch.cat([ro[i] + wi[i]-2 + (wi[i]-1) * torch.arange(he[i]) for i in range(bs)])
        row_fill_idx_botright = torch.cat([ro[i] + (he[i]-1)*(wi[i]-1) + torch.arange(wi[i]-1) for i in range(bs)])
        row_fill_idx_botleft = row_fill_idx_botright

        row_left[row_fill_idx_left, 0] = self.num_edges_total
        row_topleft[row_fill_idx_topleft, 0] = self.num_edges_total
        row_topright[row_fill_idx_topright, 0] = self.num_edges_total
        row_right[row_fill_idx_right, 0] = self.num_edges_total
        row_botright[row_fill_idx_botright, 0] = self.num_edges_total
        row_botleft[row_fill_idx_botleft, 0] = self.num_edges_total

        col_base = ro[bs] + torch.arange(co[bs])[:, None]
        col_to_col = torch.cat([wi[i] * torch.ones((he[i]-1)*wi[i], dtype=torch.int) for i in range(bs)])
        row_to_row = torch.cat([(wi[i]-1) * torch.ones((he[i]-1)*wi[i], dtype=torch.int) for i in range(bs)])

        col_topleft = torch.cat([ro[i]+h*(wi[i]-1)-1 + torch.arange(wi[i]) for i in range(bs) for h in range(he[i]-1)])
        col_topleft = col_topleft[:, None]
        col_top = col_base - col_to_col[:, None]
        col_topright = col_topleft + 1
        col_botright = col_topright + row_to_row[:, None]
        col_bot = col_base + col_to_col[:, None]
        col_botleft = col_botright - 1

        col_fill_idx_topleft = torch.cat([co[i] + wi[i] * torch.arange(he[i]-1) for i in range(bs)])
        col_fill_idx_top = torch.cat([co[i] + torch.arange(wi[i]) for i in range(bs)])
        col_fill_idx_topright = torch.cat([co[i] + wi[i]-1 + wi[i] * torch.arange(he[i]-1) for i in range(bs)])
        col_fill_idx_botright = col_fill_idx_topright
        col_fill_idx_bot = torch.cat([co[i] + (he[i]-2)*wi[i] + torch.arange(wi[i]) for i in range(bs)])
        col_fill_idx_botleft = col_fill_idx_topleft

        col_topleft[col_fill_idx_topleft, 0] = self.num_edges_total
        col_top[col_fill_idx_top, 0] = self.num_edges_total
        col_topright[col_fill_idx_topright, 0] = self.num_edges_total
        col_botright[col_fill_idx_botright, 0] = self.num_edges_total
        col_bot[col_fill_idx_bot, 0] = self.num_edges_total
        col_botleft[col_fill_idx_botleft, 0] = self.num_edges_total

        row_edge_to_edge = torch.cat([row_left, row_topleft, row_topright, row_right, row_botright, row_botleft], dim=1)
        col_edge_to_edge = torch.cat([col_topleft, col_top, col_topright, col_botright, col_bot, col_botleft], dim=1)
        edge_to_edge = torch.cat((row_edge_to_edge, col_edge_to_edge), dim=0)

        return edge_to_edge

    def get_groups(self, mask, max_iterations=5):
        true_idx_list = torch.arange(self.num_edges_total)[mask[:-1]]
        grouped_edges = self.edge_to_edge  # Clone?

        for _ in max_iterations:
            grouped_edges = grouped_edges[true_idx_list, :]

            grouped_edges[~mask[grouped_edges]] = self.num_edges_total
            min_edge = torch.min(grouped_edges, dim=1)
            new_true_idx_list = true_idx_list[true_idx_list <= min_edge]

            if len(new_true_idx_list) == len(true_idx_list):
                break
            else:
                true_idx_list = new_true_idx_list

        return grouped_edges

    def add_neighbor_edges(self, edge_idx):
        return

    def get_group_edge_feat(self, edge_idx):
        return

    def get_num_feat_per_group(self, edge_idx):
        return

    def get_feat_idx(self, edge_idx):
        return

    def update(self, edge_idx):
        return


class HardWeightGate(Function):

    @staticmethod
    def forward(ctx, weights, grp_num_feat):
        dtype = weights.dtype
        weights = torch.cumsum(1/grp_num_feat[:, None], dim=1) <= 1
        weights = weights.to(dtype)

        return weights

    @staticmethod
    def backward(ctx, grad_weights):
        return grad_weights, None


class Grouper(nn.Module):
    """
    * Grouping:
        ** Projection:
            in: features -> shape = [num_feat, feat_dim]
            param: P = projection matrix -> shape = [feat_dim, proj_dim]
            out: proj_features = mm(feature, P) -> shape = [num_feat, proj_dim]

        ** Pair comparison:
            in: proj_features -> shape = [num_feat, proj_dim]
            in: graph -> Graph object
            param: similarity_threshold -> float in [0, 1]

            pair = proj_features[graph.edge_idx_feat_idx, :] -> shape = [num_edges, 2, proj_dim]
            similarities = bmm(pair[:, 0, None, :], pair[:, 1, :, None]).squeeze() -> shape = [num_edges]

            edge_logits = torch.full([num_edges+1], fill=-sys.float_info.max, dtype=torch.float)
            out: edge_logits[:-1] = sigmoid(similarities - similarity_threshold) -> shape = [num_edges+1]

        ** Get soft grouping weights:
            in: edge_logits -> shape = [num_edges+1]
            in: features -> shape = [num_feat, feat_dim]
            in: graph -> Graph instance

            mask = edge_logits > 0.5 -> shape = [num_edges+1]
            grp_edge_idx = graph.get_groups(mask) -> shape = [num_groups, max_edges]
            grp_edge_idx_plus = graph.add_neighbor_edges(grp_edge_idx) -> shape = [num_groups, max_edges_plus]
            note: use fill=num_edges for both index tensors

            grp_edge_logits = edge_logits[grp_edge_idx_plus] -> shape = [num_groups, max_edges_plus]
            grp_edge_weights = softmax(grp_edge_logits, dim=-1) -> shape = [num_groups, max_edges_plus]

            grp_edge_feat = graph.get(grp_edge_idx_plus) -> shape = [num_groups, max_edges_plus, max_feat_plus]
            grp_feat_weights = 0.5 * bmm(grp_edge_weights[:, None, :], grp_edge_feat).squeeze()

            out: grp_feat_weights -> shape = [num_groups, max_feat_plus]
            out: grp_num_feat = graph.get_num_feat(grp_edge_idx) -> shape = [num_groups]

        ** Get hard grouping weights:
            Forward:
                in: grp_feat_weights -> shape = [num_groups, max_feat_plus]
                in: grp_num_feat -> shape = [num_groups]

                grp_feat_weights[:] = 1/grp_num_feat[:, None] -> shape = [num_groups, max_feat_plus]
                grp_mask = cumsum(grp_feat_weights, dim=1) > 1 -> shape = [num_groups, max_feat_plus]
                out: grp_feat_weights = grp_feat_weights[grp_mask] -> shape = [num_groups, max_feat_plus]

            Backward:
                in: grad_grp_feat_weights -> shape = [num_groups, max_feat_plus]
                out: grad_grp_feat_weights -> shape = [num_groups, max_feat_plus]

        ** Compute group means (new features):
            in: graph -> Graph instance
            in: grp_edge_idx_plus -> shape = [num_groups, max_edges_plus]
            in: grp_feat_weights -> shape = [num_groups, max_feat_plus]

            grp_feat_idx_plus = graph.get_feat_idx(grp_edge_idx_plus) -> shape = [num_groups, max_feat_plus]
            note: use fill=0 for simplicity (all integers between in [0, max_feature_plus-1] are allowed
            grp_feat = features[grp_feat_idx_plus, :] -> shape = [num_groups, max_feat_plus, feat_dim]

            out: grp_means = bmm(grp_feat_weights[:, None, :], grp_feat).squeeze() -> shape = [num_groups, feat_dim]
            out: graph.update(grp_edge_idx) -> update graph.{feat_idx_edge_idx, edge_idx_feat_idx}
    """

    def __init__(self, feat_dim=256, proj_dim=128, similarity_threshold=0.8):
        """
        Initializes the Grouper module.

        Args:
            feat_dim (int): Feature size.
            proj_dim (int): Size of projected features, used for comparison.
            similarity_threshold (float): Minimum similarity required for grouping.
        """

        super.__init__()

        self.projector = nn.Linear(feat_dim, proj_dim, bias=False)
        self.similarity_threshold = similarity_threshold
        self.get_hard_weights = HardWeightGate.apply

    def forward(self, in_features, graph):
        """
        Forward method of Grouper module.

        Args:
            in_features (Tensor): Input features of shape [num_groups_before x feat_dim].
            graph (Graph): Graph object storing the feature connections.

        Returns:
            out_features (Tensor): Output features of shape [num_groups_after x feat_dim]
        """

        # Project and normalize features
        proj_features = self.projector(in_features)
        norm_features = F.normalize(proj_features, dim=1)

        # Compute similarities and corresponding edge logits
        pair = norm_features[graph.edge_to_node, :]
        similarities = torch.bmm(pair[:, 0, None, :], pair[:, 1, :, None]).squeeze()
        edge_logits = torch.full([graph.num_edges_total+1], fill=-sys.float_info.max, dtype=torch.float)
        edge_logits[:-1] = torch.sigmoid(similarities - self.similarity_threshold)

        # Compute edge weights
        grp_edge_idx = graph.get_groups(edge_logits > 0.5)
        grp_edge_idx_plus = graph.add_neighbor_edges(grp_edge_idx)
        grp_edge_logits = edge_logits[grp_edge_idx_plus]
        grp_edge_weights = torch.softmax(grp_edge_logits, dim=-1)

        # Compute soft feature weights
        grp_edge_feat = graph.get_group_edge_feat(grp_edge_idx_plus)
        grp_soft_feat_weights = 0.5 * torch.bmm(grp_edge_weights[:, None, :], grp_edge_feat).squeeze()

        # Compute hard feature weights
        grp_num_feat = graph.get_num_feat_per_group(grp_edge_idx)
        grp_hard_feat_weights = self.get_hard_weights(grp_soft_feat_weights, grp_num_feat)

        # Compute group means (new output features)
        grp_feat_idx_plus = graph.get_feat_idx(grp_edge_idx_plus)
        grp_feat = in_features[grp_feat_idx_plus, :]
        out_features = torch.bmm(grp_hard_feat_weights[:, None, :], grp_feat).squeeze()

        # Update graph
        graph.update(grp_edge_idx)

        return out_features


if __name__ == "__main__":
    import datetime
    import time

    test1 = False
    test2 = False

    if test1:
        # Test parameters
        H = 32
        W = 32
        batch_size = 4
        proj_dim = 128

        #  Input
        device = torch.device("cuda")
        num_feat = H*W*batch_size
        proj_features = torch.randn(num_feat, proj_dim, device=device, requires_grad=True)

        num_edges = (2*H*W - H - W) * batch_size
        edge_idx_feat_idx = np.random.randint(0, num_feat-1, size=(num_edges, 2))

        # Parallel
        start_time = time.time()
        pair = proj_features[edge_idx_feat_idx, :]
        similarities = torch.bmm(pair[:, 0, None, :], pair[:, 1, :, None]).squeeze()

        fake_loss = similarities.sum()
        fake_loss.backward()
        end_time = time.time()
        print(f"Memory (parallel): {torch.cuda.max_memory_allocated()/(1024*1024)} MB")
        print(f"Time (parallel): {datetime.timedelta(seconds=end_time-start_time)}")

        del fake_loss, pair, similarities
        torch.cuda.reset_peak_memory_stats()

        # Sequential
        start_time = time.time()
        similarities = torch.zeros(num_edges)
        for i, (idx1, idx2) in enumerate(edge_idx_feat_idx):
            feat1 = proj_features[idx1]
            feat2 = proj_features[idx2]
            similarities[i] = torch.dot(feat1, feat2)

        fake_loss = similarities.sum()
        fake_loss.backward()
        end_time = time.time()
        print(f"Memory (sequential): {torch.cuda.max_memory_allocated()/(1024*1024)} MB")
        print(f"Time (sequential): {datetime.timedelta(seconds=end_time-start_time)}")

    if test2:
        height = [2, 3]
        width = [3, 2]
        graph = Graph(height, width)

        print(f"Node to node: {graph.node_to_node}")
        print(f"Edge to node: {graph.edge_to_node}")
        print(f"Node to edge: {graph.node_to_edge}")
        print(f"Edge to edge: {graph.edge_to_edge}")

        height = [32, 32, 32, 32]
        width = [32, 32, 32, 32]

        start_time = time.time()
        graph = Graph(height, width)
        end_time = time.time()
        print(f"Graph creation time: {datetime.timedelta(seconds=end_time-start_time)}")
