import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import is_undirected, to_undirected, remove_self_loops, k_hop_subgraph, degree
from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence, pad_packed_sequence
import copy

from torch_scatter import scatter_max

from .Convs import *
from .utils import *

convs = {
    "GAT": GATConv,
    'GCN': GCNConv,
    'MGTConv': MGTConv
}
class GNN(nn.Module):
    # embedding_dim, hidden_dim, n_nodes, n_layers, N_heads, N_nodes_type, N_edges_type, n_months=97, embedding=True, conv_name='MGTConv', use_norm=True,  k=20, dropout=0.2):
    def __init__(self, args, n_nodes):
        super(GNN, self).__init__()
        self.embedding_dim = args.embedding_dim
        self.embed = torch.nn.ModuleList()
        self.N_heads = args.N_heads
        self.drop = nn.Dropout(args.dropout)
        self.Convs = nn.ModuleList()
        self.embedding = True  # args.embedding
        self.n_layers = args.n_layers
        self.layer_nrom = nn.LayerNorm(self.embedding_dim)
        conv_name = args.conv_name
        self.h_gnn = False  # Heterogeneous Graph Neural Network
        print(conv_name)
        for _ in range(args.n_months):
            self.embed.append(torch.nn.Embedding(n_nodes, args.embedding_dim))
        for _ in range(args.n_layers):
            if conv_name in ['MGTConv']:
                self.Convs.append(convs[args.conv_name](
                    args.embedding_dim, args.N_heads, use_norm=args.use_norm, dropout=args.dropout))
            elif conv_name == 'GAT':
                self.Convs.append(convs[args.conv_name](
                    args.embedding_dim, int(args.hidden_dim/args.N_heads), args.N_heads))
            elif conv_name == 'GCN':
                self.Convs.append(convs[args.conv_name](
                    args.embedding_dim, args.embedding_dim))
            elif conv_name == 'KGT':
                self.h_gnn = True
                self.Convs.append(KGTConv(
                    args.embedding_dim, args.hidden_dim, args.N_nodes_type, args.N_edges_type, args.N_heads, args.use_norm, args.dropout))

    def forward(self, nodes, edges, node_type, edge_type, time_step, embed=False):
        if embed:
            nodes = nodes
        else:
            embed = self.embed[time_step]
            if self.embedding:
                nodes = self.layer_nrom(embed(nodes.view(-1)))#.tanh()
        for c in self.Convs:
            if self.h_gnn:
                nodes = c(nodes, edges, node_type, edge_type)

            else:
                nodes = c(nodes, edges)  # .sigmoid()
        return nodes

    def copy_next(self, index, nodes_node=None):
        self.embed[index].weight.data = self.embed[index -
                                                   1].weight.data.detach().clone()
        self.embed[index-1].weight.requires_grad = False

    def copy_embed(self, from_index, to_index):
        self.embed[to_index].weight.data = self.embed[from_index].weight.data.detach(
        ).clone()


class Matcher(nn.Module):

    def __init__(self, embedding_dim):
        super(Matcher, self).__init__()
        self.l_linear = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, s, t, infer=False):
        s = self.l_linear(s).tanh()
        t = self.l_linear(t).tanh()
        if infer:
            return (s*t), sum(-1).sigmoid()
        return torch.mm(s, t.transpose(1, 0))


class Matcher_dot(nn.Module):

    def __init__(self, embedding_dim):
        super(Matcher_dot, self).__init__()
        self.r_linear = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, s, t, infer=False):
        s = self.r_linear(s).tanh()
        t = self.r_linear(t).tanh()
        if infer:
            return (s*t), sum(-1).sigmoid()
        return (s*t).sum(-1)

class dot_matcher(nn.Module):
    def __init__(self,hidden_dim):
        super(dot_matcher,self).__init__()

    def forward(self, s, t):
        return torch.mm(s, t.transpose(1, 0))

class clf(nn.Module):

    def __init__(self, embedding_dim, n_label=2):
        super(clf, self).__init__()
        self.clf = nn.Linear(embedding_dim, n_label)

    def forward(self, embed):
        return self.clf(embed)


class Predict_model(nn.Module):

    def __init__(self, embedding_size, n_heads, n_layers, args,drop_out=0.2):
        super(Predict_model, self).__init__()
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.rnn = nn.LSTM(embedding_size, embedding_size,
                           batch_first=True, bidirectional=True)
        self.neighbors_gnn = nn.ModuleList()
        for _ in range(n_layers):
            self.neighbors_gnn.append(MGTConv(embedding_size, n_heads))
            self.clf_gnn = MGTConv(embedding_size, n_heads, infer=True)
        self.clf1 = nn.Linear(embedding_size, 32)
        self.clf2 = nn.Linear(32, 16)
        self.clf3 = nn.Linear(16, 1)
        self.drop = nn.Dropout(drop_out)
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.layer_nrom1 = nn.LayerNorm(embedding_size)
        self.batch_norm = nn.BatchNorm1d(32)

    def forward(self, embeddings, edges, node_type, edge_type, clf_neighbors, clf_nodes):
        embeds = []

        for e, edge, et in zip(embeddings, edges, edge_type):
            for i in range(self.n_layers):
                e = self.neighbors_gnn[i](nodes = self.layer_nrom1(e), edge_index = edge)#, node_type=node_type, edge_type=et)
            embeds = [e.unsqueeze(1)]+embeds
        embeds = torch.cat(embeds, 1)

        temp_embeds = embeds[clf_neighbors]
        temp_embeds_len = ((temp_embeds != 0).sum(2) != 0).sum(1)

        rnn_in = pack_padded_sequence(
            temp_embeds, temp_embeds_len.clamp(max=5).cpu(), batch_first=True, enforce_sorted=False)
        rnn_out = self.rnn(rnn_in)[0]
        seq_unpacked, lens_unpacked = pad_packed_sequence(
            rnn_out, batch_first=True)
        seq_unpacked = seq_unpacked.view(
            seq_unpacked.shape[0], seq_unpacked.shape[1], 2, self.embedding_size).mean(-2)
        embedding = torch.zeros_like(embeddings[-1]).to(embeddings[-1].device)
        embedding[clf_neighbors] = seq_unpacked.mean(1)
        clf_out, att = self.clf_gnn(embedding, edges[-1])#, node_type, edge_type[-1], att=True)
        clf_in = self.layer_norm(clf_out[clf_nodes])
        prediction = self.clf1(clf_in).tanh()  # .sigmoid()
        prediction = self.clf2(prediction).tanh()
        prediction = self.clf3(prediction).sigmoid()
        att = att.sum(1).view(-1)
        out, argmax = scatter_max(att, edges[-1][1].view(-1))

        return prediction, edges[-1][0][argmax[clf_nodes]]


class Predict_model_with_att(nn.Module):

    def __init__(self, embedding_size, n_heads, n_layers, drop_out=0.2):
        super(Predict_model_with_att, self).__init__()
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.rnn = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.neighbors_gnn = nn.ModuleList()
        for _ in range(n_layers):
            self.neighbors_gnn.append(MGTConv(64, n_heads))
        self.clf_gnn = MGTConv(embedding_size, n_heads, infer=True)
        self.clf1 = nn.Linear(embedding_size, 32)
        self.clf2 = nn.Linear(32, 16)
        self.clf3 = nn.Linear(16, 1)
        self.drop = nn.Dropout(drop_out)
        self.layer_norm = nn.LayerNorm(embedding_size)

    def forward(self, embeddings, edges, clf_neighbors, clf_nodes):
        embeds = []

        for e, edge in zip(embeddings, edges):
            for i in range(self.n_layers):
                e = self.neighbors_gnn[i](e[:, :64], edge)
            embeds = [e.unsqueeze(1)]+embeds
        embeds = torch.cat(embeds, 1)

        temp_embeds = embeds[clf_neighbors]
        temp_embeds_len = ((temp_embeds != 0).sum(2) != 0).sum(1)

        rnn_in = pack_padded_sequence(
            temp_embeds, temp_embeds_len.clamp(max=5), batch_first=True, enforce_sorted=False)
        rnn_out = self.rnn(rnn_in)[0]
        seq_unpacked, lens_unpacked = pad_packed_sequence(
            rnn_out, batch_first=True)
        seq_unpacked = seq_unpacked.view(
            seq_unpacked.shape[0], seq_unpacked.shape[1], 2, 64).mean(-2)
        embedding = torch.zeros_like(embeddings[-1]).to(embeddings[-1].device)
        embedding[clf_neighbors] = torch.cat(
            [seq_unpacked.mean(1), embeddings[-1][:, 64:][clf_neighbors]], -1)
        clf_out, att = self.clf_gnn(embedding, edges[-1])
        clf_in = clf_out[clf_nodes]
        prediction = self.clf1(clf_in).tanh()  # .sigmoid()
        prediction = self.clf2(prediction).tanh()
        prediction = self.clf3(prediction).sigmoid()
        att = att.sum(1).view(-1)
        out, argmax = scatter_max(att, edges[-1][1].view(-1))

        return prediction, edges[-1][0][argmax[clf_nodes]]


class clf_binary(nn.Module):

    def __init__(self, embedding_size, n_layers, n_heads):
        super(clf_binary, self).__init__()
        self.gnns = nn.ModuleList()
        self.n_layers = n_layers
        for _ in range(n_layers):
            self.gnns.append(MGTConv(embedding_size, n_heads))
        self.clf = nn.Linear(embedding_size, 1)

    def forward(self, nodes, edges):
        for i in range(self.n_layers):
            nodes = self.gnns[i](nodes, edges)
        return self.clf(nodes).sigmoid()
