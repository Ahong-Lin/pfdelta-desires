# FlowCAN: Flow Conservation Attention Network for Power Flow Prediction
#
# FlowCAN extends CANOS-PF with Flow Attention (out-edge softmax conservation
# routing) for message aggregation. This file also provides two ablation
# variants:
#   - FlowCANStdAttn: Standard attention (in-edge softmax, like GAT)
#   - FlowCANNoAttn: Equal weight scatter_add (no attention, like CANOS-PF)

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.canos_utils import Encoder, DecoderPF, InteractionNetwork
from core.utils.registry import registry


class FlowAttention(nn.Module):
    """
    Multi-head flow attention for edge message weighting.

    Out-edge softmax ensures conservation: for each source node,
    the attention weights over its outgoing edges sum to 1.
    This is the key physical prior from power flow: Kirchhoff's
    Current Law implies flow conservation at each node.
    """
    def __init__(self, hidden_dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = hidden_dim // n_heads
        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, sender_features, receiver_features, edge_index, num_nodes):
        """
        Compute flow attention weights for each edge.

        Returns attention weights of shape [num_edges, 1] for message scaling.
        """
        src, dst = edge_index
        E = src.shape[0]

        # Multi-head attention scores
        Q = self.W_Q(sender_features).view(E, self.n_heads, self.d_k)
        K = self.W_K(receiver_features).view(E, self.n_heads, self.d_k)

        # Scaled dot-product
        scores = (Q * K).sum(dim=-1) / (self.d_k ** 0.5)  # [E, n_heads]
        scores = scores.mean(dim=-1, keepdim=True)  # [E, 1]

        # Out-edge softmax (conservation at source node)
        # For each source node, softmax over its outgoing edges
        scores_max = torch.zeros(num_nodes, 1, device=scores.device)
        scores_max.scatter_reduce_(0, src.unsqueeze(1), scores, reduce='amax', include_self=True)
        scores_stable = scores - scores_max[src]

        exp_scores = torch.exp(scores_stable)
        sum_exp = torch.zeros(num_nodes, 1, device=scores.device)
        sum_exp.scatter_add_(0, src.unsqueeze(1), exp_scores)
        attn_weights = exp_scores / (sum_exp[src] + 1e-8)  # [E, 1]

        return attn_weights


class StandardAttention(nn.Module):
    """
    Standard attention (in-edge softmax, like GAT).
    For each destination node, softmax over its incoming edges.
    This does NOT satisfy KCL conservation.
    """
    def __init__(self, hidden_dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = hidden_dim // n_heads
        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, sender_features, receiver_features, edge_index, num_nodes):
        src, dst = edge_index
        E = src.shape[0]

        Q = self.W_Q(sender_features).view(E, self.n_heads, self.d_k)
        K = self.W_K(receiver_features).view(E, self.n_heads, self.d_k)

        scores = (Q * K).sum(dim=-1) / (self.d_k ** 0.5)
        scores = scores.mean(dim=-1, keepdim=True)

        # In-edge softmax (standard attention, like GAT)
        scores_max = torch.zeros(num_nodes, 1, device=scores.device)
        scores_max.scatter_reduce_(0, dst.unsqueeze(1), scores, reduce='amax', include_self=True)
        scores_stable = scores - scores_max[dst]

        exp_scores = torch.exp(scores_stable)
        sum_exp = torch.zeros(num_nodes, 1, device=scores.device)
        sum_exp.scatter_add_(0, dst.unsqueeze(1), exp_scores)
        attn_weights = exp_scores / (sum_exp[dst] + 1e-8)

        return attn_weights


class InteractionNetworkFlowAttn(nn.Module):
    """
    Interaction Network with Flow Attention for message aggregation.

    Identical to CANOS-PF's InteractionNetwork, except:
    - Before scatter_add, messages are weighted by flow attention
    - Everything else (EdgeUpdate, NodeUpdate, residual connections) is unchanged
    """
    def __init__(
        self,
        edge_type_dict,
        node_type_dict,
        edge_dim,
        node_dim,
        hidden_dim,
        n_heads=4,
        include_sent_messages=False,
    ):
        super().__init__()
        self.include_sent_messages = include_sent_messages

        from core.models.canos_utils import EdgeUpdate, NodeUpdate
        self.edge_update = EdgeUpdate(edge_dim, node_dim, hidden_dim, edge_type_dict)
        self.node_update = NodeUpdate(
            node_dim, hidden_dim, node_type_dict, self.include_sent_messages
        )

        # Flow Attention - the ONLY addition
        self.flow_attn = FlowAttention(hidden_dim, n_heads)

    def forward(self, nodes, edges, data):
        device = data["bus"].x.device
        edge_hidden_dim = edges.get(
            "('bus', 'ac_line', 'bus')", edges.get("('bus', 'branch', 'bus')")
        ).shape[-1]
        sent_received_node_type = {
            node_type: torch.zeros(n.shape[0], edge_hidden_dim, device=device)
            for node_type, n in nodes.items()
        }
        updated_nodes_dict = {}
        updated_edges_dict = {}

        # Get number of bus nodes for flow attention
        num_bus_nodes = nodes["bus"].shape[0]

        for edge_type, edge_feats in edges.items():
            edge_type_tuple = tuple(edge_type.strip("()").replace("'", "").split(", "))
            sender_type, receiver_type = edge_type_tuple[0], edge_type_tuple[2]
            if sender_type != "bus":
                continue
            senders, receivers = data[edge_type_tuple].edge_index

            # Gather node features
            sender_features = nodes[sender_type][senders]
            receiver_features = nodes[receiver_type][receivers]

            # Calculate edge updates (SAME as CANOS-PF)
            updated_edges = self.edge_update(
                edge_feats, sender_features, receiver_features, edge_type
            )

            # === THE ONLY DIFFERENCE: Flow Attention weighting ===
            attn_weights = self.flow_attn(
                sender_features, receiver_features,
                data[edge_type_tuple].edge_index, num_bus_nodes
            )
            weighted_edges = updated_edges * attn_weights  # [E, hidden_dim]
            # === END OF DIFFERENCE ===

            # Pass messages with attention weights (instead of equal weights)
            sent_received_node_type[receiver_type].scatter_add_(
                0, receivers.unsqueeze(-1).expand_as(weighted_edges), weighted_edges
            )
            if self.include_sent_messages:
                sent_received_node_type[sender_type].scatter_add_(
                    0, senders.unsqueeze(-1).expand_as(weighted_edges), weighted_edges
                )

            updated_edges_dict[edge_type] = updated_edges + edge_feats

        # Apply the object model phi_o (node_update) - SAME as CANOS-PF
        for node_type, node_feats in nodes.items():
            updated_nodes = self.node_update(
                node_feats, sent_received_node_type[node_type], node_type
            )
            updated_nodes_dict[node_type] = updated_nodes + node_feats
        return updated_nodes_dict, updated_edges_dict


class InteractionNetworkStdAttn(nn.Module):
    """
    Interaction Network with Standard Attention for message aggregation.

    Same as InteractionNetworkFlowAttn but uses StandardAttention
    (in-edge softmax, like GAT) instead of FlowAttention.
    """
    def __init__(
        self,
        edge_type_dict,
        node_type_dict,
        edge_dim,
        node_dim,
        hidden_dim,
        n_heads=4,
        include_sent_messages=False,
    ):
        super().__init__()
        self.include_sent_messages = include_sent_messages

        from core.models.canos_utils import EdgeUpdate, NodeUpdate
        self.edge_update = EdgeUpdate(edge_dim, node_dim, hidden_dim, edge_type_dict)
        self.node_update = NodeUpdate(
            node_dim, hidden_dim, node_type_dict, self.include_sent_messages
        )

        # Standard Attention (in-edge softmax, like GAT)
        self.std_attn = StandardAttention(hidden_dim, n_heads)

    def forward(self, nodes, edges, data):
        device = data["bus"].x.device
        edge_hidden_dim = edges.get(
            "('bus', 'ac_line', 'bus')", edges.get("('bus', 'branch', 'bus')")
        ).shape[-1]
        sent_received_node_type = {
            node_type: torch.zeros(n.shape[0], edge_hidden_dim, device=device)
            for node_type, n in nodes.items()
        }
        updated_nodes_dict = {}
        updated_edges_dict = {}

        num_bus_nodes = nodes["bus"].shape[0]

        for edge_type, edge_feats in edges.items():
            edge_type_tuple = tuple(edge_type.strip("()").replace("'", "").split(", "))
            sender_type, receiver_type = edge_type_tuple[0], edge_type_tuple[2]
            if sender_type != "bus":
                continue
            senders, receivers = data[edge_type_tuple].edge_index

            sender_features = nodes[sender_type][senders]
            receiver_features = nodes[receiver_type][receivers]

            updated_edges = self.edge_update(
                edge_feats, sender_features, receiver_features, edge_type
            )

            # Standard Attention weighting (in-edge softmax)
            attn_weights = self.std_attn(
                sender_features, receiver_features,
                data[edge_type_tuple].edge_index, num_bus_nodes
            )
            weighted_edges = updated_edges * attn_weights

            sent_received_node_type[receiver_type].scatter_add_(
                0, receivers.unsqueeze(-1).expand_as(weighted_edges), weighted_edges
            )
            if self.include_sent_messages:
                sent_received_node_type[sender_type].scatter_add_(
                    0, senders.unsqueeze(-1).expand_as(weighted_edges), weighted_edges
                )

            updated_edges_dict[edge_type] = updated_edges + edge_feats

        for node_type, node_feats in nodes.items():
            updated_nodes = self.node_update(
                node_feats, sent_received_node_type[node_type], node_type
            )
            updated_nodes_dict[node_type] = updated_nodes + node_feats
        return updated_nodes_dict, updated_edges_dict


@registry.register_model("flowcan")
class FlowCAN(nn.Module):
    """
    FlowCAN: Flow Conservation Attention Network for Power Flow Prediction.

    Identical to CANOS-PF in every way, except that message aggregation
    uses Flow Attention (out-edge softmax) instead of equal-weight scatter_add.
    """
    def __init__(self, dataset, hidden_dim, include_sent_messages, k_steps, n_heads=4):
        super().__init__()
        edge_feat_dim = node_feat_dim = hidden_dim

        self.case_name = dataset.case_name
        self.encoder = Encoder(data=dataset, hidden_size=hidden_dim)

        node_type_dict = {
            node_type: True for node_type in dataset[0].num_node_features.keys()
        }

        edge_type_dict = {
            str(edge_type): True if "edge_attr" in dataset[0][edge_type] else False
            for edge_type in dataset[0].edge_types
            if "bus" in edge_type[0]
        }

        self.message_passing_layers = nn.ModuleList(
            InteractionNetworkFlowAttn(
                edge_type_dict=edge_type_dict,
                node_type_dict=node_type_dict,
                edge_dim=edge_feat_dim,
                node_dim=node_feat_dim,
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                include_sent_messages=include_sent_messages,
            )
            for _ in range(k_steps)
        )

        self.decoder = DecoderPF(hidden_size=hidden_dim)
        self.k_steps = k_steps

    def forward(self, data):
        projected_nodes, projected_edges = self.encoder(data)

        nodes, edges = projected_nodes, projected_edges
        for l in range(self.k_steps):
            nodes, edges = self.message_passing_layers[l](nodes, edges, data)

        output_dict = self.decoder(nodes, data)

        p_fr, q_fr, p_to, q_to = self.derive_branch_flows(output_dict, data)
        output_dict["edge_preds"] = torch.stack([p_fr, q_fr, p_to, q_to], dim=-1)

        p_slack_net, q_slack_net = self.derive_slack_output(output_dict, data)
        output_dict["slack"] = torch.stack([p_slack_net, q_slack_net], dim=-1)

        output_dict["casename"] = self.case_name
        return output_dict

    def derive_branch_flows(self, output_dict, data):
        va = output_dict["bus"][:, 0]
        vm = output_dict["bus"][:, -1]
        v_complex = vm * torch.exp(1j * va)

        edge_index = data["bus", "branch", "bus"].edge_index
        edge_attr = data["bus", "branch", "bus"].edge_attr

        br_r, br_x = edge_attr[:, 0], edge_attr[:, 1]
        b_fr, b_to = edge_attr[:, 3], edge_attr[:, 5]
        g_fr, g_to = edge_attr[:, 2], edge_attr[:, 4]
        tap = edge_attr[:, 6]
        shift = edge_attr[:, 7]
        branch_mask = (
            (edge_attr[:, 8] > 0).to(edge_attr.dtype)
            if edge_attr.shape[1] > 8
            else torch.ones_like(br_r)
        )

        T_complex = tap * torch.exp(1j * shift)
        Y_branch = 1 / (br_r + 1j * br_x)
        Y_c_fr = 1j * b_fr
        Y_c_to = 1j * b_to

        i, j = edge_index[0], edge_index[1]
        vi = v_complex[i]
        vj = v_complex[j]

        S_fr = (Y_branch + Y_c_fr).conj() * (torch.abs(vi) ** 2) / (
            torch.abs(T_complex) ** 2
        ) - Y_branch.conj() * (vi * vj.conj()) / T_complex

        S_to = (Y_branch + Y_c_to).conj() * (torch.abs(vj) ** 2) - Y_branch.conj() * (
            vj * vi.conj()
        ) / T_complex.conj()

        p_fr, q_fr = S_fr.real, S_fr.imag
        p_to, q_to = S_to.real, S_to.imag
        p_fr, q_fr = p_fr * branch_mask, q_fr * branch_mask
        p_to, q_to = p_to * branch_mask, q_to * branch_mask
        return p_fr, q_fr, p_to, q_to

    def derive_slack_output(self, output_dict, data):
        device = data["bus"].x.device
        n = data["bus"].x.shape[0]
        bus_shunts = data["bus"].shunt.to(device)
        slack_idx = data["slack", "slack_link", "bus"].edge_index[1]
        edge_pred = output_dict["edge_preds"]
        edge_indices = data["bus", "branch", "bus"].edge_index
        flows_rev = edge_pred[:, 2] + 1j * edge_pred[:, 3]
        flows_fwd = edge_pred[:, 0] + 1j * edge_pred[:, 1]
        sum_branch_flows = torch.zeros(n, dtype=torch.cfloat, device=device)
        sum_branch_flows.scatter_add_(0, edge_indices[0], flows_fwd)
        sum_branch_flows.scatter_add_(0, edge_indices[1], flows_rev)
        bus_pred = output_dict["bus"]
        va, vm = bus_pred.T

        shunt_flows = (torch.abs(vm) ** 2) * (
            bus_shunts[:, 1] + 1j * bus_shunts[:, 0]
        )
        slack_net_generation = sum_branch_flows[slack_idx] + shunt_flows[slack_idx]
        p_slack_net = torch.real(slack_net_generation)
        q_slack_net = torch.imag(slack_net_generation)
        return p_slack_net, q_slack_net


@registry.register_model("flowcan_std_attn")
class FlowCANStdAttn(nn.Module):
    """
    FlowCAN with Standard Attention (ablation variant).

    Same as FlowCAN but uses StandardAttention (in-edge softmax, like GAT)
    instead of FlowAttention (out-edge softmax). This does NOT satisfy
    KCL conservation.
    """
    def __init__(self, dataset, hidden_dim, include_sent_messages, k_steps, n_heads=4):
        super().__init__()
        edge_feat_dim = node_feat_dim = hidden_dim

        self.case_name = dataset.case_name
        self.encoder = Encoder(data=dataset, hidden_size=hidden_dim)

        node_type_dict = {
            node_type: True for node_type in dataset[0].num_node_features.keys()
        }

        edge_type_dict = {
            str(edge_type): True if "edge_attr" in dataset[0][edge_type] else False
            for edge_type in dataset[0].edge_types
            if "bus" in edge_type[0]
        }

        self.message_passing_layers = nn.ModuleList(
            InteractionNetworkStdAttn(
                edge_type_dict=edge_type_dict,
                node_type_dict=node_type_dict,
                edge_dim=edge_feat_dim,
                node_dim=node_feat_dim,
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                include_sent_messages=include_sent_messages,
            )
            for _ in range(k_steps)
        )

        self.decoder = DecoderPF(hidden_size=hidden_dim)
        self.k_steps = k_steps

    def forward(self, data):
        projected_nodes, projected_edges = self.encoder(data)

        nodes, edges = projected_nodes, projected_edges
        for l in range(self.k_steps):
            nodes, edges = self.message_passing_layers[l](nodes, edges, data)

        output_dict = self.decoder(nodes, data)

        p_fr, q_fr, p_to, q_to = self.derive_branch_flows(output_dict, data)
        output_dict["edge_preds"] = torch.stack([p_fr, q_fr, p_to, q_to], dim=-1)

        p_slack_net, q_slack_net = self.derive_slack_output(output_dict, data)
        output_dict["slack"] = torch.stack([p_slack_net, q_slack_net], dim=-1)

        output_dict["casename"] = self.case_name
        return output_dict

    def derive_branch_flows(self, output_dict, data):
        va = output_dict["bus"][:, 0]
        vm = output_dict["bus"][:, -1]
        v_complex = vm * torch.exp(1j * va)

        edge_index = data["bus", "branch", "bus"].edge_index
        edge_attr = data["bus", "branch", "bus"].edge_attr

        br_r, br_x = edge_attr[:, 0], edge_attr[:, 1]
        b_fr, b_to = edge_attr[:, 3], edge_attr[:, 5]
        g_fr, g_to = edge_attr[:, 2], edge_attr[:, 4]
        tap = edge_attr[:, 6]
        shift = edge_attr[:, 7]
        branch_mask = (
            (edge_attr[:, 8] > 0).to(edge_attr.dtype)
            if edge_attr.shape[1] > 8
            else torch.ones_like(br_r)
        )

        T_complex = tap * torch.exp(1j * shift)
        Y_branch = 1 / (br_r + 1j * br_x)
        Y_c_fr = 1j * b_fr
        Y_c_to = 1j * b_to

        i, j = edge_index[0], edge_index[1]
        vi = v_complex[i]
        vj = v_complex[j]

        S_fr = (Y_branch + Y_c_fr).conj() * (torch.abs(vi) ** 2) / (
            torch.abs(T_complex) ** 2
        ) - Y_branch.conj() * (vi * vj.conj()) / T_complex

        S_to = (Y_branch + Y_c_to).conj() * (torch.abs(vj) ** 2) - Y_branch.conj() * (
            vj * vi.conj()
        ) / T_complex.conj()

        p_fr, q_fr = S_fr.real, S_fr.imag
        p_to, q_to = S_to.real, S_to.imag
        p_fr, q_fr = p_fr * branch_mask, q_fr * branch_mask
        p_to, q_to = p_to * branch_mask, q_to * branch_mask
        return p_fr, q_fr, p_to, q_to

    def derive_slack_output(self, output_dict, data):
        device = data["bus"].x.device
        n = data["bus"].x.shape[0]
        bus_shunts = data["bus"].shunt.to(device)
        slack_idx = data["slack", "slack_link", "bus"].edge_index[1]
        edge_pred = output_dict["edge_preds"]
        edge_indices = data["bus", "branch", "bus"].edge_index
        flows_rev = edge_pred[:, 2] + 1j * edge_pred[:, 3]
        flows_fwd = edge_pred[:, 0] + 1j * edge_pred[:, 1]
        sum_branch_flows = torch.zeros(n, dtype=torch.cfloat, device=device)
        sum_branch_flows.scatter_add_(0, edge_indices[0], flows_fwd)
        sum_branch_flows.scatter_add_(0, edge_indices[1], flows_rev)
        bus_pred = output_dict["bus"]
        va, vm = bus_pred.T

        shunt_flows = (torch.abs(vm) ** 2) * (
            bus_shunts[:, 1] + 1j * bus_shunts[:, 0]
        )
        slack_net_generation = sum_branch_flows[slack_idx] + shunt_flows[slack_idx]
        p_slack_net = torch.real(slack_net_generation)
        q_slack_net = torch.imag(slack_net_generation)
        return p_slack_net, q_slack_net


@registry.register_model("flowcan_no_attn")
class FlowCANNoAttn(nn.Module):
    """
    FlowCAN without attention (ablation variant).

    Same as FlowCAN but uses equal-weight scatter_add (like CANOS-PF)
    instead of Flow Attention. This is equivalent to CANOS-PF and serves
    as the baseline ablation.
    """
    def __init__(self, dataset, hidden_dim, include_sent_messages, k_steps, n_heads=4):
        super().__init__()
        edge_feat_dim = node_feat_dim = hidden_dim

        self.case_name = dataset.case_name
        self.encoder = Encoder(data=dataset, hidden_size=hidden_dim)

        node_type_dict = {
            node_type: True for node_type in dataset[0].num_node_features.keys()
        }

        edge_type_dict = {
            str(edge_type): True if "edge_attr" in dataset[0][edge_type] else False
            for edge_type in dataset[0].edge_types
            if "bus" in edge_type[0]
        }

        # Use CANOS-PF's InteractionNetwork directly (no attention)
        self.message_passing_layers = nn.ModuleList(
            InteractionNetwork(
                edge_type_dict=edge_type_dict,
                node_type_dict=node_type_dict,
                edge_dim=edge_feat_dim,
                node_dim=node_feat_dim,
                hidden_dim=hidden_dim,
                include_sent_messages=include_sent_messages,
            )
            for _ in range(k_steps)
        )

        self.decoder = DecoderPF(hidden_size=hidden_dim)
        self.k_steps = k_steps

    def forward(self, data):
        projected_nodes, projected_edges = self.encoder(data)

        nodes, edges = projected_nodes, projected_edges
        for l in range(self.k_steps):
            nodes, edges = self.message_passing_layers[l](nodes, edges, data)

        output_dict = self.decoder(nodes, data)

        p_fr, q_fr, p_to, q_to = self.derive_branch_flows(output_dict, data)
        output_dict["edge_preds"] = torch.stack([p_fr, q_fr, p_to, q_to], dim=-1)

        p_slack_net, q_slack_net = self.derive_slack_output(output_dict, data)
        output_dict["slack"] = torch.stack([p_slack_net, q_slack_net], dim=-1)

        output_dict["casename"] = self.case_name
        return output_dict

    def derive_branch_flows(self, output_dict, data):
        va = output_dict["bus"][:, 0]
        vm = output_dict["bus"][:, -1]
        v_complex = vm * torch.exp(1j * va)

        edge_index = data["bus", "branch", "bus"].edge_index
        edge_attr = data["bus", "branch", "bus"].edge_attr

        br_r, br_x = edge_attr[:, 0], edge_attr[:, 1]
        b_fr, b_to = edge_attr[:, 3], edge_attr[:, 5]
        g_fr, g_to = edge_attr[:, 2], edge_attr[:, 4]
        tap = edge_attr[:, 6]
        shift = edge_attr[:, 7]
        branch_mask = (
            (edge_attr[:, 8] > 0).to(edge_attr.dtype)
            if edge_attr.shape[1] > 8
            else torch.ones_like(br_r)
        )

        T_complex = tap * torch.exp(1j * shift)
        Y_branch = 1 / (br_r + 1j * br_x)
        Y_c_fr = 1j * b_fr
        Y_c_to = 1j * b_to

        i, j = edge_index[0], edge_index[1]
        vi = v_complex[i]
        vj = v_complex[j]

        S_fr = (Y_branch + Y_c_fr).conj() * (torch.abs(vi) ** 2) / (
            torch.abs(T_complex) ** 2
        ) - Y_branch.conj() * (vi * vj.conj()) / T_complex

        S_to = (Y_branch + Y_c_to).conj() * (torch.abs(vj) ** 2) - Y_branch.conj() * (
            vj * vi.conj()
        ) / T_complex.conj()

        p_fr, q_fr = S_fr.real, S_fr.imag
        p_to, q_to = S_to.real, S_to.imag
        p_fr, q_fr = p_fr * branch_mask, q_fr * branch_mask
        p_to, q_to = p_to * branch_mask, q_to * branch_mask
        return p_fr, q_fr, p_to, q_to

    def derive_slack_output(self, output_dict, data):
        device = data["bus"].x.device
        n = data["bus"].x.shape[0]
        bus_shunts = data["bus"].shunt.to(device)
        slack_idx = data["slack", "slack_link", "bus"].edge_index[1]
        edge_pred = output_dict["edge_preds"]
        edge_indices = data["bus", "branch", "bus"].edge_index
        flows_rev = edge_pred[:, 2] + 1j * edge_pred[:, 3]
        flows_fwd = edge_pred[:, 0] + 1j * edge_pred[:, 1]
        sum_branch_flows = torch.zeros(n, dtype=torch.cfloat, device=device)
        sum_branch_flows.scatter_add_(0, edge_indices[0], flows_fwd)
        sum_branch_flows.scatter_add_(0, edge_indices[1], flows_rev)
        bus_pred = output_dict["bus"]
        va, vm = bus_pred.T

        shunt_flows = (torch.abs(vm) ** 2) * (
            bus_shunts[:, 1] + 1j * bus_shunts[:, 0]
        )
        slack_net_generation = sum_branch_flows[slack_idx] + shunt_flows[slack_idx]
        p_slack_net = torch.real(slack_net_generation)
        q_slack_net = torch.imag(slack_net_generation)
        return p_slack_net, q_slack_net
