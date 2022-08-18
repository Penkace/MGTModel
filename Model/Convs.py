import math
from time import sleep

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax

from .utils import *

class KGTConv(MessagePassing):

    def __init__(self, embedding_dim, hidden_dim, N_nodes_type, N_edges_type, N_heads, use_norm=True, dropout=0.2):
        '''
        embedding_size: The size of inputs.
        hidden_dim: The size of outputs.
        N_nodes_type: Number of classes of the nodes.
        N_edges_type: # of classes of edges.
        N_heads: # of the attention heads.
        '''
        super(KGTConv, self).__init__(aggr='add')
        self.embedding_dim = embedding_dim
        self.hidden_dim = int(hidden_dim/N_heads)
        self.sqrt_dk = math.sqrt(self.hidden_dim)
        self.N_nodes_type = N_nodes_type
        self.N_edges_type = N_edges_type
        self.N_heads = N_heads
        self.dropout = dropout
        self.use_norm = use_norm
        # self.k = k
        self.K_Linear = nn.ModuleList()
        self.Q_Linear = nn.ModuleList()
        self.V_Linear = nn.ModuleList()
        self.M_Linear = nn.ModuleList()
        if use_norm:
            self.norms = nn.ModuleList()
        for _ in range(N_nodes_type):
            self.K_Linear.append(
                nn.Linear(embedding_dim, self.hidden_dim*N_heads, bias=False))
            self.M_Linear.append(
                nn.Linear(hidden_dim, hidden_dim, bias=False))
            self.V_Linear.append(
                nn.Linear(embedding_dim, self.hidden_dim*N_heads, bias=False))
            if use_norm:
                self.norms.append(nn.LayerNorm(hidden_dim))
        self.W = nn.Linear(hidden_dim, hidden_dim,
                           bias=False)  # nn.ModuleList()
        for _ in range(N_edges_type):
            #self.W.append(                )
            self.Q_Linear.append(
                nn.Linear(embedding_dim, self.hidden_dim*N_heads, bias=False))
        self.att = None
        
        self.drop = nn.Dropout(self.dropout)

    def forward(self, nodes, edge_index, node_type, edge_type, att=False):
        '''
        nodes: nodes' embedding.
        edge_index: [2,N] matrix, [center,neighbor] every cloumn.
        '''
        return self.propagate(edge_index, nodes=nodes, node_type=node_type, edge_type=edge_type ,need_att=att)

    def message(self, edge_index_i, edge_index_j, nodes_i, nodes_j, node_type, edge_type, need_att):
        
        node_type_i = node_type[edge_index_i]
        node_type_j = node_type[edge_index_j]
        data_size = edge_index_i.size(0)
        K = torch.zeros(data_size, self.N_heads,
                        self.hidden_dim).to(nodes_i.device)
        Q = torch.zeros(data_size, self.N_heads,
                        self.hidden_dim).to(nodes_i.device)
        V = torch.zeros(data_size, self.N_heads,
                        self.hidden_dim).to(nodes_i.device)

        for nt in range(self.N_nodes_type):
            Q[node_type_i.view(-1) == int(nt)] = self.Q_Linear[nt](
                nodes_i[node_type_i.view(-1) == int(nt)]).view(-1, self.N_heads, self.hidden_dim)
            K[node_type_j.view(-1) == int(nt)] = self.K_Linear[nt](
                nodes_j[node_type_j.view(-1) == int(nt)]).view(-1, self.N_heads, self.hidden_dim)
        res_att = (Q * K).sum(dim=-1) / self.sqrt_dk

        #att_mask = attention_topk(torch.cat([edge_index_i.view(1,-1),edge_index_j.view(1,-1)],0),res_att.mean(1),self.k)
        for et in range(self.N_edges_type):
            n_ids = (edge_type.view(-1) == int(et))
            V[n_ids] = self.V_Linear[nt](
                nodes_j[n_ids]).view(-1, self.N_heads, self.hidden_dim)
        att = softmax(res_att, edge_index_i).float()
        if need_att:
            self.att = att

        del Q,K
        V = (V*att.view(-1, self.N_heads, 1)
             ).view(-1, self.hidden_dim*self.N_heads)
        V = self.W(V)
        return V

    # def edge_mask(self, edge_index, nodes, node_type, edge_type):
    #     edge_index_i = edge_index[0]
    #     edge_index_j = edge_index[1]
    #     node_type_i = node_type[edge_index_i]
    #     node_type_j = node_type[edge_index_j]
    #     nodes_i = nodes[edge_index_i]
    #     nodes_j = nodes[edge_index_j]
    #     data_size = edge_index_i.size(0)
    #     K = torch.zeros(data_size, self.N_heads,
    #                     self.hidden_dim).to(nodes_i.device)
    #     Q = torch.zeros(data_size, self.N_heads,
    #                     self.hidden_dim).to(nodes_i.device)
    #     for nt in range(self.N_nodes_type):
    #         Q[node_type_i.view(-1) == int(nt)] = self.Q_Linear[nt](
    #             nodes_i[node_type_i.view(-1) == int(nt)]).view(-1, self.N_heads, self.hidden_dim)
    #         K[node_type_j.view(-1) == int(nt)] = self.K_Linear[nt](
    #             nodes_j[node_type_j.view(-1) == int(nt)]).view(-1, self.N_heads, self.hidden_dim)
    #     res_att = (Q * K).sum(dim=-1) / self.sqrt_dk
    #     att_mask = attention_topk(nodes.shape[0],torch.cat([edge_index_i.view(
    #         1, -1), edge_index_j.view(1, -1)], 0), res_att.mean(1), self.k)
    #     return att_mask

    def update(self, agg_out, nodes, node_type, edge_index, need_att):
        agg_out = agg_out.tanh()
        res = torch.zeros_like(nodes).to(nodes.device)
        for nt in range(self.N_nodes_type):
            n_ids = (node_type.view(-1) == int(nt)).to(nodes.device)
            out = self.drop(self.M_Linear[nt](agg_out[n_ids]))+nodes[n_ids]
            if self.use_norm:
                out = self.norms[nt](out)
            res[n_ids] = out
        if need_att:
            return res, self.att
        return res
    
class MGTConv(MessagePassing):
    def __init__(self, embedding_dim, n_heads, dropout=0.2, use_norm=True,infer=False):
        super(MGTConv, self).__init__(aggr='add')
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads

        self.aggregate_Linear_k = nn.Linear(
            embedding_dim, self.embedding_dim*n_heads, bias=False)
        self.aggregate_Linear_q = nn.Linear(
            embedding_dim, self.embedding_dim*n_heads, bias=False)
        self.aggregate_Linear_v = nn.Linear(
            embedding_dim, self.embedding_dim*n_heads, bias=False)

        self.agg_W = nn.Linear(embedding_dim*n_heads,
                               embedding_dim, bias=False)
        self.agg_dk = math.sqrt(self.embedding_dim)
        self.drop = nn.Dropout(dropout)
        self.out_linear = nn.Linear(embedding_dim*2, embedding_dim)
        self.use_norm = use_norm
        if use_norm:
            self.Norm = nn.LayerNorm(embedding_dim)
        self.skip = nn.Parameter(torch.ones(1))
        self.infer=infer
        self.att=None

    def forward(self, nodes, edge_index):
        return self.propagate(edge_index, nodes=nodes)

    def message(self, edge_index_i, edge_index_j, nodes_i, nodes_j):
        neighbors = nodes_j
        data_size = edge_index_i.shape[0]
        agg_k = self.aggregate_Linear_k(neighbors).view(
            data_size, self.n_heads, self.embedding_dim)
        agg_q = self.aggregate_Linear_q(nodes_i).view(
            data_size, self.n_heads, self.embedding_dim)
        agg_v = self.aggregate_Linear_v(neighbors).view(
            data_size, self.n_heads, self.embedding_dim)
        att = (agg_k*agg_q).sum(-1) / self.agg_dk
        att = softmax(att, edge_index_i).view(data_size,self.n_heads,1)
        agg_v = (att*agg_v).view(-1, self.embedding_dim*self.n_heads)
        agg_v = self.agg_W(agg_v).tanh()
        if self.infer:
            self.att=att
        return agg_v

    def update(self, agg_out, nodes, edge_index_i, edge_index_j):
        #agg_out = self.agg_W(agg_out)
        agg_out = F.gelu(self.out_linear(torch.cat([agg_out, nodes], dim=1)))
        alpha = self.skip.sigmoid()
        if self.use_norm:
            agg_out = self.Norm(
                alpha*agg_out + (1-alpha)*nodes)
        else:
            agg_out = alpha*agg_out + \
                (1-alpha)*nodes
        if self.infer:
            return agg_out,self.att
        return agg_out

class NTAConv(MessagePassing):
    def __init__(self, embedding_dim, use_norm=True, dropout=0.2):
        '''

        '''
        super(NTAConv, self).__init__(aggr='add')
        self.N_nodes_type = N_nodes_type
        self.hid_linear = nn.ModuleList()
        #for _ in range(N_nodes_type):
        self.hid_linear=nn.Linear(embedding_dim, embedding_dim)
            # Embeding the nodes into same space
        self.att_linear = nn.Linear(embedding_dim*2, 1)
        self.agg_linear = nn.Linear(embedding_dim*2, embedding_dim)
        self.use_norm = use_norm
        self.drop = nn.Dropout(dropout)

    def forward(self, nodes, edges, node_type):
        return self.propagate(edges, nodes=nodes, node_type=node_type)

    def message(self, nodes_i, nodes_j, edges_i, edges_j):
        source = torch.zeros_like(nodes_j).to(nodes_i.device)
        target = torch.zeros_like(nodes_i).to(nodes_i.device)
        source= self.hid_linear(nodes_j)
        target = self.hid_linear(nodes_i)
        # Embeding the nodes into same space
        att = self.att_linear(torch.cat([source, target], 1))
        # calculate the attention value
        att = softmax(att, edges_i)
        return source*att

    def update(self, agg_out, nodes):
        res = torch.zeros_like(nodes).to(nodes.device)
        res = self.drop(self.agg_linear(torch.cat([nodes, agg_out], 1)))+nodes
        if self.use_norm:
            res = self.norm(res)
        return res


class NATandKGTConvs(nn.Module):

    def __init__(self, embedding_dim, n_heads, dropout=0.2, use_norm=True):
        super(NATandKGTConvs, self).__init__(aggr='add')
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads

        self.aggregate_Linear_k = nn.Linear(
            embedding_dim, self.embedding_dim*n_heads, bias=False)
        self.aggregate_Linear_q = nn.Linear(
            embedding_dim, self.embedding_dim*n_heads, bias=False)
        self.aggregate_Linear_v = nn.Linear(
            embedding_dim, self.embedding_dim*n_heads, bias=False)

        self.agg_W = nn.Linear(embedding_dim*n_heads,
                               embedding_dim, bias=False)
        self.agg_dk = math.sqrt(self.embedding_dim)
        self.drop = nn.Dropout(dropout)
        self.out_linear = nn.Linear(embedding_dim*2, embedding_dim)
        self.use_norm = use_norm
        if use_norm:
            self.Norm = nn.LayerNorm(embedding_dim)
        self.skip = nn.Parameter(torch.ones(1))

    def forward(self, nodes, edge_index):
        return self.propagate(edge_index, nodes=nodes)

    def message(self, edge_index_i, edge_index_j, nodes_i, nodes_j):
        neighbors = nodes_j
        data_size = edge_index_i.shape[0]
        agg_k = self.aggregate_Linear_k(neighbors).view(
            data_size, self.n_heads, self.embedding_dim)
        agg_q = self.aggregate_Linear_q(nodes_i).view(
            data_size, self.n_heads, self.embedding_dim)
        agg_v = self.aggregate_Linear_v(neighbors).view(
            data_size, self.n_heads, self.embedding_dim)
        att = (agg_k*agg_q).sum(-1) / self.agg_dk
        att = softmax(att, edge_index_i).view(data_size,self.n_heads,1)
        agg_v = (att*agg_v).view(-1, self.embedding_dim*self.n_heads)
        agg_v = self.agg_W(agg_v)
        return agg_v

    def update(self, agg_out, nodes, edge_index_i, edge_index_j):
        #agg_out = self.agg_W(agg_out)
        agg_out = F.gelu(self.out_linear(torch.cat([agg_out, nodes], dim=1)))
        alpha = self.skip.sigmoid()
        if self.use_norm:
            agg_out = self.Norm(
                alpha*agg_out + (1-alpha)*nodes)
        else:
            agg_out = alpha*agg_out + \
                (1-alpha)*nodes
        return agg_out
