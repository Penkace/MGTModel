
import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch_geometric.data import (Data, GraphSAINTEdgeSampler,
                                  GraphSAINTNodeSampler,
                                  GraphSAINTRandomWalkSampler)
from torch_geometric.utils import *
import wandb
from Model.Model import *
from Model.utils import *
parser = argparse.ArgumentParser(description='Training model')

'''Data Setting'''
parser.add_argument('--data_dir', type=str, default='Data',
                    help='The address of csv data.')
parser.add_argument('--Model_dir', type=str, default='Saved_model',
                    help='The address to save the trained model.')
parser.add_argument('--val_size', type=float, default=0.5,
                    help='Val set size.')
parser.add_argument('--save_data_dir', type=str,
                    default='Data/PitchBook/H5file')
parser.add_argument('--years', type=int,
                    default=8)
parser.add_argument('--comparison', type=boolean_string,
                    default=False)
parser.add_argument('--sim_threshold', type=float,
                    default=0.98)
parser.add_argument('--n_months', type=int, default=97)
parser.add_argument('--count', type=int,
                    default=50)
'''Model Arg'''
parser.add_argument('--N_heads', type=int, default=4,
                    help='Number of attention heads.')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout ratio.')
parser.add_argument('--embedding_dim', type=int, default=16,
                    help='Embedding size.')
parser.add_argument('--hidden_dim', type=int, default=16,
                    help='Hidden layer size.')
parser.add_argument('--conv_name', type=str, default='KGT',
                    help='Type of Convs.')
parser.add_argument('--N_nodes_type', type=int, default=2,
                    help='Type of Convs.')
parser.add_argument('--N_edges_type', type=int, default=12,
                    help='Type of Convs.')
parser.add_argument('--n_layers', type=int, default=1,
                    help='Number of layers of Convs.')
parser.add_argument('--n_layers_clf', type=int, default=1,
                    help='Number of layers of Convs.')
parser.add_argument('--use_norm', type=bool, default=True,
                    help='Use norm?')
parser.add_argument('--gpus', type=str, default='cuda:0',
                    help='')

'''Optimization arguments'''
parser.add_argument('--optimizer', type=str, default='adam',
                    choices=['adamw', 'adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate.')
parser.add_argument('--batch_size', type=int, default=512,
                    help='batch size')
parser.add_argument('--n_epoch_update', type=int, default=100,
                    help='Number of epoch to init the embedding')
parser.add_argument('--init_epoch', type=int, default=50)
parser.add_argument('--n_epoch_init', type=int, default=50,
                    help='Number of the epoch to update the embedding')
parser.add_argument('--n_epoch_predic', type=int, default=100,
                    help='Number of epoch to run')
parser.add_argument('--lr_clf', type=float, default=1e-4,
                    help='Learning rate.')
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--random_threshold', type = float, default=0.1)

'''Task arguments'''
parser.add_argument('--nub_node_type', type=int, default=1)

parser.add_argument('--train_embed', type=boolean_string, default=True)

parser.add_argument('--train_comparison', type=boolean_string, default=True)

parser.add_argument('--task_name', default='TKDE_Version_1', type=str)

parser.add_argument('--loss_type', type=str, default='LPNC',
                    choices=['NC', 'LP', 'CL', 'LPNC'])
parser.add_argument('--n_predict_step', type=int, default=10)
parser.add_argument('--dynamic_clf', type=boolean_string, default=True)
args = parser.parse_args()

wandb.init(project="VC", name=args.task_name)
wandb.config.update(vars(args))


setup_seed(7)
'''
load data
'''
try:
    graph_edges, edge_date, edge_type, all_nodes, new_companies, labels, new_nodes, new_edges, nodetypes, ID2index = load_pb_from_h5(
        args.save_data_dir)
except:
    graph_edges, edge_date, edge_type, all_nodes, new_companies, labels, new_nodes, new_edges, nodetypes, ID2index = load_pitchbook(
        args.data_dir, args.save_data_dir)


assert graph_edges[-1].shape[1]==edge_date.shape[0]
assert graph_edges[-1].shape[1]==edge_type.shape[0]
print(edge_type.max())
print(edge_date.max())
print(len(graph_edges))


for c, e in zip(new_companies, labels):
    assert len(c) == len(e)

# load model
Model = GNN(args, graph_edges[-1].max()+1).to(args.gpus)
matcher = Matcher(args.hidden_dim).to(args.gpus)
clf = clf(args.hidden_dim, args.nub_node_type).to(args.gpus)
if args.optimizer == 'adamw':
    optimizers = torch.optim.AdamW
elif args.optimizer == 'adam':
    optimizers = torch.optim.Adam
elif args.optimizer == 'sgd':
    optimizers = torch.optim.SGD
elif args.optimizer == 'adagrad':
    optimizers = torch.optim.Adagrad

optimizer = optimizers([
    {'params': filter(lambda p: p.requires_grad,
                        Model.parameters()), 'lr': args.lr},
    {'params': matcher.parameters(), 'lr': args.lr},
    {'params': clf.parameters(), 'lr': args.lr}
])


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 800, eta_min=1e-6)

'''
Training embedding
'''
if args.train_embed:
    '''
    Node classification and link prediction tasks are used to train node embedding
    '''
    if args.loss_type == 'LPNC':
        edges_t0 = graph_edges[0]
        Node_type = sorted(ID2index.items(), key=lambda x: x[1], reverse=True)
        Node_type = torch.tensor(
            [0.0 if 'P' in x[0] else 1 for x in Node_type]).to(args.gpus)
        node_set = TensorDataset(edges_t0.transpose(0,1))
        node_loader = DataLoader(node_set, args.batch_size, shuffle=True,drop_last=True)

        nodes_all = torch.range(0, graph_edges[-1].max()).long().to(args.gpus)
        bar = tqdm.tqdm(range(args.n_epoch_update))
        for i in bar:
            for nodes in node_loader:
                o_nodes = nodes[0].unique()
                nodes, edges, node_idx, edge_mask = k_hop_subgraph(
                    o_nodes, args.n_layers, edges_t0, relabel_nodes=True)
                type_nodes_input = nodetypes[nodes]
                type_nodes = nodetypes[node_idx]
                type_edges = edge_type[edge_date == 0][edge_mask]
                nodes, edges, type_nodes, type_edges = [
                    i.to(args.gpus) for i in [nodes, edges, type_nodes, type_edges]]
                embed = Model(nodes, edges, type_nodes_input,
                              type_edges, -1)[node_idx, :]
                predict_NC = clf(embed)
                #print(predict_NC.shape, Node_type[o_nodes].shape)
                predict_LP = matcher(embed, embed)
                target = to_dense_adj(edges)[0][:, node_idx][node_idx, :]
                pos = (target.shape[0]**2-target.sum())/target.sum()
                label_nc = Node_type[o_nodes]
                pos_NC = (len(label_nc)-label_nc.sum())/label_nc.sum()
                loss = args.alpha*F.binary_cross_entropy_with_logits(
                    predict_LP, target, pos_weight=pos)+(1-args.alpha)*F.binary_cross_entropy_with_logits(predict_NC.view(-1),label_nc,pos_weight=pos_NC)
                bar.set_description('%f' % (loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        Model.copy_embed(-1, 0)
        # updating
        for k, v in Model.named_parameters():
            if 'embed' not in k:
                v.requires_grad = False
        optimizer = optimizers([
            {'params': filter(lambda p: p.requires_grad,
                              Model.parameters()), 'lr': args.lr},
        ])
        torch.cuda.empty_cache()
        bar = tqdm.tqdm(range(1, args.years*12+1))
        for index in bar:
            # print('*'*10+str(index)+'*'*10)
            edges_new = graph_edges[index].long().to(args.gpus)
            edges_old = graph_edges[index-1].to(args.gpus)

            nodes_new = new_edges[index-1].unique().to(args.gpus)
            nodes_old = nodes_new[isin(
                nodes_new, edges_old.unique())].unique()

            nodes_sub, edges_sub, node_idx, edge_mask = k_hop_subgraph(
                nodes_new, args.n_layers, edges_new, relabel_nodes=True)

            node_type_new = Node_type[nodes_sub]
            edge_type_new = edge_type[edge_date <= index][edge_mask]

            node_clf_new = Node_type[nodes_new]

            nodes_old_sub = node_idx[isin(
                nodes_new, edges_old.unique())]

            nodes_new_sub = node_idx

            target = to_dense_adj(edges_sub.cpu())[
                0][:, nodes_old_sub].to(args.gpus)

            pos = (target.shape[0]**2-target.sum())/target.sum()
            label_NC = Node_type[nodes_new]
            pos_NC = (len(label_NC)-label_NC.sum())/label_NC.sum()
            old_embed = Model.embed[index-1].weight.data.detach().clone()
            new_added_nodes = new_nodes[index-1]
            for i in range(args.n_epoch_init):
                embed_index = Model.embed[index].weight.data.detach().clone()
                embed_new = Model(nodes_sub, edges_sub,
                                  node_type_new, edge_type_new, -1)
                with torch.no_grad():
                    embed_old = Model(old_embed, edges_old, Node_type, edge_type[edge_date < index],
                                      index-1, True)[nodes_old, :]
                predict = matcher(embed_new, embed_old)
                predict_NC = clf(embed_new[node_idx])
                loss = args.alpha*F.binary_cross_entropy_with_logits(
                    predict, target, pos_weight=pos)+(1-args.alpha)*F.binary_cross_entropy_with_logits(predict_NC.view(-1),label_NC,pos_NC)
                bar.set_description('%f' % (loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            Model.copy_embed(-1, index)
            torch.cuda.empty_cache()
    elif args.loss_type == 'LP':
        '''
        Link prediction task is used to train node embedding
        '''
        edges_t0 = graph_edges[0]
        Node_type = sorted(ID2index.items(), key=lambda x: x[1], reverse=True)
        Node_type = torch.tensor(
            [0.0 if 'P' in x[0] else 1 for x in Node_type]).to(args.gpus)
        node_set = TensorDataset(edges_t0.transpose(0,1))
        node_loader = DataLoader(node_set, args.batch_size, shuffle=True,drop_last=True)

        nodes_all = torch.range(0, graph_edges[-1].max()).long().to(args.gpus)
        bar = tqdm.tqdm(range(args.n_epoch_update))
        for i in bar:
            for nodes in node_loader:
                o_nodes = nodes[0].unique()
                nodes, edges, node_idx, edge_mask = k_hop_subgraph(
                    o_nodes, args.n_layers, edges_t0, relabel_nodes=True)
                type_nodes_input = nodetypes[nodes]
                type_nodes = nodetypes[node_idx]
                type_edges = edge_type[edge_date == 0][edge_mask]
                nodes, edges, type_nodes, type_edges = [
                    i.to(args.gpus) for i in [nodes, edges, type_nodes, type_edges]]
                embed = Model(nodes, edges, type_nodes_input,
                              type_edges, -1)[node_idx, :]
                predict_NC = clf(embed)
                #print(predict_NC.shape, Node_type[o_nodes].shape)
                predict_LP = matcher(embed, embed)
                target = to_dense_adj(edges)[0][:, node_idx][node_idx, :]
                pos = (target.shape[0]**2-target.sum())/target.sum()
                label_nc = Node_type[o_nodes]
                pos_NC = (len(label_nc)-label_nc.sum())/label_nc.sum()
                loss = F.binary_cross_entropy_with_logits(
                    predict_LP, target, pos_weight=pos)
                bar.set_description('%f' % (loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        Model.copy_embed(-1, 0)
        # updating
        for k, v in Model.named_parameters():
            if 'embed' not in k:
                v.requires_grad = False
        optimizer = optimizers([
            {'params': filter(lambda p: p.requires_grad,
                              Model.parameters()), 'lr': args.lr},
        ])
        torch.cuda.empty_cache()
        bar = tqdm.tqdm(range(1, args.years*12+1))
        for index in bar:
            edges_new = graph_edges[index].long().to(args.gpus)
            edges_old = graph_edges[index-1].to(args.gpus)

            nodes_new = new_edges[index-1].unique().to(args.gpus)
            nodes_old = nodes_new[isin(
                nodes_new, edges_old.unique())].unique()

            nodes_sub, edges_sub, node_idx, edge_mask = k_hop_subgraph(
                nodes_new, args.n_layers, edges_new, relabel_nodes=True)

            node_type_new = Node_type[nodes_sub]
            edge_type_new = edge_type[edge_date <= index][edge_mask]

            node_clf_new = Node_type[nodes_new]

            nodes_old_sub = node_idx[isin(
                nodes_new, edges_old.unique())]

            nodes_new_sub = node_idx

            target = to_dense_adj(edges_sub.cpu())[
                0][:, nodes_old_sub].to(args.gpus)

            pos = (target.shape[0]**2-target.sum())/target.sum()
            label_NC = Node_type[nodes_new]
            pos_NC = (len(label_NC)-label_NC.sum())/label_NC.sum()
            old_embed = Model.embed[index-1].weight.data.detach().clone()
            new_added_nodes = new_nodes[index-1]
            for i in range(args.n_epoch_init):
                embed_index = Model.embed[index].weight.data.detach().clone()
                embed_new = Model(nodes_sub, edges_sub,
                                  node_type_new, edge_type_new, -1)
                with torch.no_grad():
                    embed_old = Model(old_embed, edges_old, Node_type, edge_type[edge_date < index],
                                      index-1, True)[nodes_old, :]
                predict = matcher(embed_new, embed_old)
                predict_NC = clf(embed_new[node_idx])
                loss = F.binary_cross_entropy_with_logits(
                    predict, target, pos_weight=pos)+(1-args.alpha)
                bar.set_description('%f' % (loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            Model.copy_embed(-1, index)
            torch.cuda.empty_cache()
    
    elif args.loss_type == 'NC':  # node classification
        '''
        Node classification task is used to train node embedding
        '''
        edges_t0 = graph_edges[0]
        Node_type = sorted(ID2index.items(), key=lambda x: x[1], reverse=True)
        Node_type = torch.tensor(
            [0.0 if 'P' in x[0] else 1 for x in Node_type]).to(args.gpus)
        node_set = TensorDataset(edges_t0.transpose(0,1))
        node_loader = DataLoader(node_set, args.batch_size, shuffle=True,drop_last=True)

        nodes_all = torch.range(0, graph_edges[-1].max()).long().to(args.gpus)
        bar = tqdm.tqdm(range(args.n_epoch_update))
        for i in bar:
            for nodes in node_loader:
                o_nodes = nodes[0].unique()
                nodes, edges, node_idx, edge_mask = k_hop_subgraph(
                    o_nodes, args.n_layers, edges_t0, relabel_nodes=True)
                type_nodes_input = nodetypes[nodes]
                type_nodes = nodetypes[node_idx]
                type_edges = edge_type[edge_date == 0][edge_mask]
                nodes, edges, type_nodes, type_edges = [
                    i.to(args.gpus) for i in [nodes, edges, type_nodes, type_edges]]
                embed = Model(nodes, edges, type_nodes_input,
                              type_edges, -1)[node_idx, :]
                predict_NC = clf(embed)
                predict_LP = matcher(embed, embed)
                target = to_dense_adj(edges)[0][:, node_idx][node_idx, :]
                pos = (target.shape[0]**2-target.sum())/target.sum()
                label_nc = Node_type[o_nodes]
                pos_NC = (len(label_nc)-label_nc.sum())/label_nc.sum()
                loss = F.binary_cross_entropy_with_logits(predict_NC.view(-1),label_nc,pos_weight=pos_NC)
                bar.set_description('%f' % (loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        Model.copy_embed(-1, 0)
        # updating
        for k, v in Model.named_parameters():
            if 'embed' not in k:
                v.requires_grad = False
        optimizer = optimizers([
            {'params': filter(lambda p: p.requires_grad,
                              Model.parameters()), 'lr': args.lr},
        ])
        torch.cuda.empty_cache()
        bar = tqdm.tqdm(range(1, args.years*12+1))
        for index in bar:
            # print('*'*10+str(index)+'*'*10)
            edges_new = graph_edges[index].long().to(args.gpus)
            edges_old = graph_edges[index-1].to(args.gpus)

            nodes_new = new_edges[index-1].unique().to(args.gpus)
            nodes_old = nodes_new[isin(
                nodes_new, edges_old.unique())].unique()

            nodes_sub, edges_sub, node_idx, edge_mask = k_hop_subgraph(
                nodes_new, args.n_layers, edges_new, relabel_nodes=True)

            node_type_new = Node_type[nodes_sub]
            edge_type_new = edge_type[edge_date <= index][edge_mask]

            node_clf_new = Node_type[nodes_new]

            nodes_old_sub = node_idx[isin(
                nodes_new, edges_old.unique())]

            nodes_new_sub = node_idx

            target = to_dense_adj(edges_sub.cpu())[
                0][:, nodes_old_sub].to(args.gpus)

            pos = (target.shape[0]**2-target.sum())/target.sum()
            label_NC = Node_type[nodes_new]
            pos_NC = (len(label_NC)-label_NC.sum())/label_NC.sum()
            old_embed = Model.embed[index-1].weight.data.detach().clone()
            new_added_nodes = new_nodes[index-1]
            for i in range(args.n_epoch_init):
                embed_index = Model.embed[index].weight.data.detach().clone()
                # nodes,edges,node_idx,_=k_hop_subgraph(nodes,args.n_layers,edges_t0,relabel_nodes=True)
                embed_new = Model(nodes_sub, edges_sub,
                                  node_type_new, edge_type_new, -1)
                with torch.no_grad():
                    embed_old = Model(old_embed, edges_old, Node_type, edge_type[edge_date < index],
                                      index-1, True)[nodes_old, :]
                predict = matcher(embed_new, embed_old)
                predict_NC = clf(embed_new[node_idx])
                loss = (1-args.alpha)*F.binary_cross_entropy_with_logits(predict_NC.view(-1),label_NC,pos_NC)
                bar.set_description('%f' % (loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            Model.copy_embed(-1, index)
            torch.cuda.empty_cache()  
    #############Contrastive Loss#####################
    elif args.loss_type == 'CL':
        node_set = TensorDataset(
            graph_edges[0].unique().long())  # .to(args.gpus)
        node_loader = DataLoader(node_set, args.batch_size, shuffle=True)

        if not is_undirected(graph_edges[0]):
            edges_t0 = to_undirected(graph_edges[0]).long().to(args.gpus)
        else:
            edges_t0 = graph_edges[0].long().to(args.gpus)
        nodes, edges, node_idx, _ = k_hop_subgraph(
            edges_t0.unique(), args.n_layers, edges_t0, relabel_nodes=True)
        bar = tqdm.tqdm(range(args.n_epoch_update))
        for i in bar:
            target_edge, target = random_neg_select(
                edges, nodes, number_of_neg=10)
            embed0 = Model(nodes, random_add_edge(edges, 1), -1)
            embed1 = Model(nodes, edges, -1)
            #predict = matcher(embed0[target_edge[0]], embed1[target_edge[1]])
            #pos = (target.shape[0]-target.sum())//target.sum()
            # print((embed0[target_edge[0]]-embed1[target_edge[1]]).shape)
            loss = F.hinge_embedding_loss(
                embed0[target_edge[0]]-embed1[target_edge[1]], (torch.ones_like(embed0[target_edge[0]])*(target*2-1).unsqueeze(1)))
            # loss = F.binary_cross_entropy_with_logits(
            #    predict, target, pos_weight=pos)
            bar.set_description('%.2f' % (loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # updating
        bar = tqdm.tqdm(range(1, args.years*12+1))
        for index in bar:

            if not is_undirected(graph_edges[index]):
                edges_new = to_undirected(
                    graph_edges[index].long()).to(args.gpus)
            else:
                edges_new = graph_edges[index].long().to(args.gpus)
            if not is_undirected(graph_edges[index-1]):
                edges_old = to_undirected(graph_edges[index-1]).to(args.gpus)
            else:
                edges_old = graph_edges[index-1].to(args.gpus)
            nodes_new = new_edges[index-1].unique()
            nodes_old = nodes_new[isin(
                nodes_new, graph_edges[index-1].unique())]
            Model.copy_next(index, edges_old.unique())
            optimizer = optimizers([
                {'params': filter(lambda p: p.requires_grad,
                                  Model.parameters()), 'lr': args.lr},
                {'params': matcher.parameters(), 'lr': args.lr},
            ])
            for i in range(args.n_epoch_init):
                target_edge, target = random_neg_select(
                    edges_new, nodes_new, number_of_neg=10)
                #edges_new = random_add_edge(edges_new, 0.2)
                embed_new = Model(nodes_all, edges_new, index)
                with torch.no_grad():
                    embed_old = Model(nodes_all, edges_old,
                                      index-1)
                # predict = matcher(
                #     embed_new[target_edge[0]], embed_old[target_edge[1]])
                # #print(embed_new[target_edge[0]].shape, embed_old[target_edge[1]].shape,predict.shape)
                # pos = (target.shape[0]-target.sum())//target.sum()
                # loss = F.binary_cross_entropy_with_logits(
                #     predict, target, pos_weight=pos)
                loss = F.hinge_embedding_loss(
                    embed_new[target_edge[0]]-embed_old[target_edge[1]], (torch.ones_like(embed_new[target_edge[0]])*(target*2-1).unsqueeze(1)), margin=0.5)
                bar.set_description('%f' % (loss))
                optimizer.zero_grad()
                loss.backward()
                bar.set_description('%.2f' % (loss))
                optimizer.step()
# torch.cuda.empty_cache()

'''
Save the node representation
'''
if args.train_embed:
    save_path = args.Model_dir+'/'+args.task_name
    check_dir(save_path)
    torch.save(Model, save_path+'/Embedding.pth')
else:
    save_path = args.Model_dir+'/'+args.task_name + '/'
    Model = torch.load(save_path+'Embedding.pth').to(args.gpus)