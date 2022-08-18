#from apex import amp
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
#from apex import amp
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch_geometric.data import (Data, GraphSAINTEdgeSampler,
                                  GraphSAINTNodeSampler,
                                  GraphSAINTRandomWalkSampler)
from torch_geometric.utils import *
from sklearn.metrics import classification_report,roc_auc_score,accuracy_score,f1_score,precision_score,recall_score
# import wandb
from Model.Model import *
from Model.utils import *
import warnings 
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='Training model')



'''Data Setting'''
parser.add_argument('--data_dir', type=str, default='/public/VC_Project/Data',
                    help='The address of csv data.')
parser.add_argument('--Model_dir', type=str, default='Saved_model',
                    help='The address to save the trained model.')
parser.add_argument('--val_size', type=float, default=0.5,
                    help='Val set size.')
parser.add_argument('--save_data_dir', type=str,
                    default='/public/VC_Project/Data/PitchBook/H5file')
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
parser.add_argument('--n_layers_clf', type=int, default=2,
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
parser.add_argument('--n_epoch_predic', type=int, default=50,
                    help='Number of epoch to run')
parser.add_argument('--lr_clf', type=float, default=1e-4,
                    help='Learning rate.')
parser.add_argument('--alpha', type=float, default=0.1)# 0.1
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


setup_seed(7)

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



for _ in range(5):
    '''

    '''
    Model = GNN(args, graph_edges[-1].max()+1).to(args.gpus)
    # Matcher继承Module类
    matcher = Matcher(args.hidden_dim).to(args.gpus)
    clf_model = clf(args.hidden_dim, args.nub_node_type).to(args.gpus)
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
        {'params': clf_model.parameters(), 'lr': args.lr}
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 800, eta_min=1e-6)



    # 训练Embedding
    if args.train_embed:
        if args.loss_type == 'LPNC':
            edges_t0 = graph_edges[0]
            Node_type = sorted(ID2index.items(), key=lambda x: x[1], reverse=True)
            Node_type = torch.tensor(
                [0.0 if 'P' in x[0] else 1 for x in Node_type]).to(args.gpus)
            node_set = TensorDataset(edges_t0.transpose(0,1))
            node_loader = DataLoader(node_set, args.batch_size, shuffle=True,drop_last=True)

            nodes_all = torch.range(0, graph_edges[-1].max()).long().to(args.gpus)
            # bar = tqdm.tqdm(range(args.n_epoch_update))
            print('start training')
            for i in range(args.n_epoch_update):
            # for i in bar:
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
                    predict_NC = clf_model(embed)
                    #print(predict_NC.shape, Node_type[o_nodes].shape)
                    predict_LP = matcher(embed, embed)
                    target = to_dense_adj(edges)[0][:, node_idx][node_idx, :]
                    pos = (target.shape[0]**2-target.sum())/target.sum()
                    label_nc = Node_type[o_nodes]
                    pos_NC = (len(label_nc)-label_nc.sum())/label_nc.sum()
                    loss = args.alpha*F.binary_cross_entropy_with_logits(
                        predict_LP, target, pos_weight=pos)+(1-args.alpha)*F.binary_cross_entropy_with_logits(predict_NC.view(-1),label_nc,pos_weight=pos_NC)
                    # bar.set_description('%f' % (loss))
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
                #{'params': matcher.parameters(), 'lr': args.lr},
            ])
            torch.cuda.empty_cache()
            # bar = tqdm.tqdm(range(1, args.years*12+1))
            for index in range(1,args.years*12+1):
            # for index in bar:
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
                    predict_NC = clf_model(embed_new[node_idx])
                    loss = args.alpha*F.binary_cross_entropy_with_logits(
                        predict, target, pos_weight=pos)+(1-args.alpha)*F.binary_cross_entropy_with_logits(predict_NC.view(-1),label_NC,pos_NC)
                    # bar.set_description('%f' % (loss))
                    # print(loss)
                    optimizer.zero_grad()
                    # with amp.scale_loss(loss, optimizer) as scaled_loss:
                    #    scaled_loss.backward()
                    loss.backward()
                    optimizer.step()
                Model.copy_embed(-1, index)
                torch.cuda.empty_cache()
        elif args.loss_type=='NC':
            print('################## Start Training NC ################# ')
            edges_t0 = graph_edges[0]
            Node_type = sorted(ID2index.items(), key=lambda x: x[1], reverse=True)
            Node_type = torch.tensor(
                [0.0 if 'P' in x[0] else 1 for x in Node_type]).to(args.gpus)
            node_set = TensorDataset(edges_t0.transpose(0,1))
            node_loader = DataLoader(node_set, args.batch_size, shuffle=True,drop_last=True)

            nodes_all = torch.range(0, graph_edges[-1].max()).long().to(args.gpus)
            # bar = tqdm.tqdm(range(args.n_epoch_update))
            # for i in bar:
            for i in range(args.n_epoch_update):
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
                    predict_NC = clf_model(embed)
                    #print(predict_NC.shape, Node_type[o_nodes].shape)
                    predict_LP = matcher(embed, embed)
                    target = to_dense_adj(edges)[0][:, node_idx][node_idx, :]
                    pos = (target.shape[0]**2-target.sum())/target.sum()
                    label_nc = Node_type[o_nodes]
                    pos_NC = (len(label_nc)-label_nc.sum())/label_nc.sum()
                    loss = F.binary_cross_entropy_with_logits(predict_NC.view(-1),label_nc,pos_weight=pos_NC)
                    # bar.set_description('%f' % (loss))
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
                #{'params': matcher.parameters(), 'lr': args.lr},
            ])
            torch.cuda.empty_cache()
            # bar = tqdm.tqdm(range(1, args.years*12+1))
            # for index in bar:
            for index in range(1,args.years*12+1):
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
                    predict_NC = clf_model(embed_new[node_idx])
                    loss = (1-args.alpha)*F.binary_cross_entropy_with_logits(predict_NC.view(-1),label_NC,pos_NC)
                    # bar.set_description('%f' % (loss))
                    # print(loss)
                    optimizer.zero_grad()
                    # with amp.scale_loss(loss, optimizer) as scaled_loss:
                    #    scaled_loss.backward()
                    loss.backward()
                    optimizer.step()
                Model.copy_embed(-1, index)
                torch.cuda.empty_cache()
        
        elif args.loss_type == 'LP':
            print('################## Start Training LP ################# ')
            edges_t0 = graph_edges[0]
            Node_type = sorted(ID2index.items(), key=lambda x: x[1], reverse=True)
            Node_type = torch.tensor(
                [0.0 if 'P' in x[0] else 1 for x in Node_type]).to(args.gpus)
            node_set = TensorDataset(edges_t0.transpose(0,1))
            node_loader = DataLoader(node_set, args.batch_size, shuffle=True,drop_last=True)

            nodes_all = torch.range(0, graph_edges[-1].max()).long().to(args.gpus)
            # bar = tqdm.tqdm(range(args.n_epoch_update))
            for i in range(args.n_epoch_update):
            # for i in bar:
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
                    predict_NC = clf_model(embed)
                    #print(predict_NC.shape, Node_type[o_nodes].shape)
                    predict_LP = matcher(embed, embed)
                    target = to_dense_adj(edges)[0][:, node_idx][node_idx, :]
                    pos = (target.shape[0]**2-target.sum())/target.sum()
                    label_nc = Node_type[o_nodes]
                    pos_NC = (len(label_nc)-label_nc.sum())/label_nc.sum()
                    loss = F.binary_cross_entropy_with_logits(
                        predict_LP, target, pos_weight=pos)
                    # bar.set_description('%f' % (loss))
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
                #{'params': matcher.parameters(), 'lr': args.lr},
            ])
            torch.cuda.empty_cache()
            # bar = tqdm.tqdm(range(1, args.years*12+1))
            # for index in bar:
            for index in range(1,args.years*12+1):
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
                    predict_NC = clf_model(embed_new[node_idx])
                    loss = F.binary_cross_entropy_with_logits(
                        predict, target, pos_weight=pos)+(1-args.alpha)
                    # bar.set_description('%f' % (loss))
                    # print(loss)
                    optimizer.zero_grad()
                    # with amp.scale_loss(loss, optimizer) as scaled_loss:
                    #    scaled_loss.backward()
                    loss.backward()
                    optimizer.step()
                Model.copy_embed(-1, index)
                torch.cuda.empty_cache()
            
    # 保存对应任务的Embedding的结果
    if args.train_embed:
        save_path = args.Model_dir+'/'+args.task_name
        check_dir(save_path)
        torch.save(Model, save_path+'/Embedding.pth')
    else:
        save_path = args.Model_dir+'/'+args.task_name + '/'
        Model = torch.load(save_path+'Embedding.pth').to(args.gpus)

    print('################### Finishing Training Embedding ###################')


# predictor = Predict_model(args.embedding_dim, args.N_heads, args.n_layers_clf, args, args.dropout).to(args.gpus)



    embeddings = []

    for i, e in zip(Model.embed, graph_edges):
        i = i.weight.data.to(args.gpus).clone().detach()
        temp = torch.zeros_like(i).to(args.gpus)
        temp[e.unique()] += i[e.unique()]
        embeddings.append(temp.unsqueeze(0))
        # print((((i[e.unique()])==0)!=0).sum(1).sum(),i[e.unique()].sum(1).shape,i.shape)
        # torch.cuda.empty_cache()
    embeddings = torch.cat(embeddings, 0)



    traning_index = list(range(1, 23))
    val_index = list(range(23, 25))
    test_index = [i+84 for i in range(1, 1+12)]

    for _ in range(5):
        predictor = Predict_modelv2(args.embedding_dim, args.N_heads, args.n_layers_clf, args, args.dropout).to(args.gpus)


        optimizer = optimizers([
            {'params': predictor.parameters(), 'lr': args.lr_clf, 'weight_decay': 0.0001}
        ])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 800, eta_min=1e-6)

            
        final_f1_score_list = []
        final_accuracy_list = []
        final_recall_list = []
        final_precision_list = []
        final_ap10 = []
        final_ap20 = []
        final_ap50 = []
        final_auc = []
        
        this_epoch_best_score = 0
        this_epoch_best_recall = 0
        this_epoch_best_precision = 0
        this_epoch_best_f1 = 0
        this_epoch_best_ap10 = 0
        this_epoch_best_ap20 = 0
        this_epoch_best_ap50 = 0
        this_epoch_best_auc = 0
        this_epoch_rep = ''
        total_loss = 0
        best_auc = 0
        CRLoss = nn.CrossEntropyLoss()
        for ep in range(args.n_epoch_predic):
            predictor.train()
            
            for epp in range(int(2/(1-args.random_threshold))):
                # print('xun huan : ',epp)
                for index in traning_index:
                    if index < args.n_predict_step:
                        edges_train = [i.to(args.gpus) for i in graph_edges[:index+1]]
                        edge_type_train = [edge_type[edge_date <= i]
                                        for i in range(index+1)]
                        train_embeds = embeddings[:index+1, :, :]
                    else:
                        edges_train = [i.to(args.gpus)
                                    for i in graph_edges[index-args.n_predict_step+1:index+1]]
                        edge_type_train = [edge_type[edge_date <= i]
                                        for i in range(index-args.n_predict_step+1, index+1)]
                        train_embeds = embeddings[index -
                                                args.n_predict_step+1:index+1, :, :]
                    assert len(edges_train)==len(edge_type_train)
                    #edges_train=[random_add_edge(i,0.5) for i in edges_train]
                    predict_nodes = new_companies[index -
                                                1].clone().detach().to(args.gpus)

                    neighbors = K_hop_nodes(predict_nodes, edges_train[-1])
                    edges_train = [K_hop_neighbors(neighbors, i, j)
                                for i, j in zip(edges_train, edge_type_train)]
                    # nodes_type_train = [i[2] for i in edges_train]
                    edges_type_train = [i[1] for i in edges_train]
                    edges_train = [i[0] for i in edges_train]
                    # prediction, att = predictor(train_embeds, edges_train, nodetypes, edges_type_train,
                    #                             neighbors, predict_nodes)
                    prediction, _ = predictor(train_embeds, edges_train, nodetypes, edges_type_train,
                                            neighbors, predict_nodes)
                    
                    # label = labels[index-1].clone().detach().to(args.gpus).float()
                    label = labels[index-1].clone().detach().to(args.gpus)
                    random_mask = torch.rand_like(label.float())>args.random_threshold
                    predict_random_mask = torch.stack([random_mask,random_mask],dim=1)
                    label = label*random_mask                
                    prediction_mask = prediction*predict_random_mask
                    loss = CRLoss(prediction_mask,label)
                    total_loss+=loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            scheduler.step()
            predictor.eval()
            
            with torch.no_grad():
                val_prediction = []
                val_label = []
                month = []
                for index in val_index:
                    if len(labels[index-1] == 1) and labels[index-1][0] == -1:
                        continue
                    edges_val = [i.to(args.gpus)
                                for i in graph_edges[index-args.n_predict_step+1:index+1]]
                    edge_type_val = [edge_type[edge_date <= i]
                                    for i in range(index-args.n_predict_step+1, index+1)]
                    val_embeds = embeddings[index-args.n_predict_step+1:index+1]
                    predict_nodes = new_companies[index-1].to(args.gpus)
                    neighbors = K_hop_nodes(predict_nodes, edges_val[-1])
                    #edges_val = [K_hop_neighbors(neighbors, i) for i in edges_val]
                    prediction, _ = predictor(val_embeds, edges_val, nodetypes, edge_type_val,neighbors, predict_nodes)
                    _,val_predict_node = torch.max(prediction.data,1)
                    val_predict_node = val_predict_node.cpu().detach()
                    label = labels[index-1].clone().detach().view(-1)
                    val_prediction.append(val_predict_node)
                    val_label.append(label)
                    month+=[index for i in val_predict_node]
                
                val_prediction = torch.cat(val_prediction)
                val_label = torch.cat(val_label)
                val_ap = np.mean(list(calc_apatk(val_prediction, val_label, month).values()))
                auc_val, aupr_val, f1_val, best_threshold = eval_metric(val_label, val_prediction)
                val_rep = classification_report(val_label,val_prediction,digits=6,target_names=['false','true'])
                val_score = accuracy_score(val_label, val_prediction)# 计算验证集的准确率
                val_precision = precision_score(val_label,val_prediction,average='macro')
                val_recall = recall_score(val_label, val_prediction,average='macro')
                val_f1 = f1_score(val_label, val_prediction,average='macro')
                # print('val size: ',val_label.size(),val_prediction.size())
                
                
                test_prediction = []
                test_label = []
                month_test = []
                # ap = {}
                predict_companies = []
                atts = []
                for index in test_index:
                    edges_test = [i.to(args.gpus)
                                for i in graph_edges[index-args.n_predict_step+1:index+1]]
                    edge_type_test = [edge_type[edge_date <= i]
                                    for i in range(index-args.n_predict_step+1, index+1)]
                    test_embeds = embeddings[index-args.n_predict_step+1:index+1]
                    predict_nodes = new_companies[index-1].clone().detach()
                    predict_companies.append(predict_nodes)
                    neighbors = K_hop_nodes(predict_nodes, edges_test[-1])
                    #edges_test = [K_hop_neighbors(neighbors, i) for i in edges_test]
                    # prediction, att = predictor(test_embeds, edges_test, nodetypes, edge_type_test,
                    #                             neighbors, predict_nodes)
                    
                    prediction, _ = predictor(test_embeds, edges_test, nodetypes, edge_type_test,neighbors, predict_nodes)
                    _,test_node_label = torch.max(prediction.data,1)
                    label = labels[index-1].view(-1).clone().detach()
                    
                    # print('Test Prediction : ',test_node_label.size(),label.size())
                    
                    test_prediction.append(test_node_label.detach().cpu())
                    test_label.append(label)
                    
                    
                    month_test+=[index for i in test_node_label]
                    
                test_prediction = torch.cat(test_prediction)
                test_label = torch.cat(test_label)
                # print('test label : ',test_prediction.size(),test_label.size())
                auc_test,aupr_test,f1_test = eval_metric(test_label,test_prediction,best_threshold)
                
                test_rep = classification_report(test_label,test_prediction,digits=6,target_names=['false','true'])
                test_score = accuracy_score(test_label, test_prediction)# 计算验证集的准确率
                test_precision = precision_score(test_label,test_prediction,average='macro')
                test_recall = recall_score(test_label, test_prediction,average='macro')
                test_f1 = f1_score(test_label, test_prediction,average='macro')
                test_ap = np.mean(list(calc_apatk(test_prediction, test_label, month_test).values()))
                
                if best_auc<auc_val:
                    best_auc = auc_val
                    # save_path = args.Model_dir+'/'+args.task_name + '/'
                    # check_dir(save_path)
                    # torch.save(Model, save_path+'Best_clf.pth')
                    test_ap = calc_apatk(test_prediction, test_label, month_test)
                    this_epoch_best_score = test_score
                    this_epoch_best_recall = test_recall
                    this_epoch_best_precision = test_precision
                    this_epoch_best_f1 = test_f1
                    this_epoch_best_ap10 = test_ap['10']
                    this_epoch_best_ap20 = test_ap['20']
                    this_epoch_best_ap50 = test_ap['50']
                    this_epoch_best_auc = auc_test
                    this_epoch_rep = test_rep
        # print('The result after epoch : ',ep)      
        print('ap result : ',this_epoch_best_ap10,this_epoch_best_ap20,this_epoch_best_ap50)
    #     print('Test Accuracy : ',this_epoch_best_score)
    #     print('Test Precision : ',this_epoch_best_precision)
    #     print('Test Recall : ',this_epoch_best_recall)
    #     print('Test f1_score : ',this_epoch_best_f1)
    #     print('Test auc : ',this_epoch_best_auc)
    #     final_f1_score_list.append(this_epoch_best_f1)
    #     final_accuracy_list.append(this_epoch_best_score)
    #     final_recall_list.append(this_epoch_best_recall)
    #     final_precision_list.append(this_epoch_best_precision)
    #     final_ap10.append(this_epoch_best_ap10)
    #     final_ap20.append(this_epoch_best_ap20)
    #     final_ap50.append(this_epoch_best_ap50)
    #     final_auc.append(this_epoch_best_auc)
    # final_f1_score_list = np.array(final_f1_score_list)
    # final_accuracy_list = np.array(final_accuracy_list)
    # final_recall_list = np.array(final_recall_list)
    # final_precision_list = np.array(final_precision_list)
    # final_ap10 = np.array(final_ap10)
    # final_ap20 = np.array(final_ap20)
    # final_ap50  = np.array(final_ap50) 
    # final_auc = np.array(final_auc)
    # print('The final score : ',final_accuracy_list.mean(),final_accuracy_list.std())
    # print('The final recall : ',final_recall_list.mean(),final_recall_list.std())
    # print('The final precision : ',final_precision_list.mean(),final_precision_list.std())
    # print('The final f1 : ',final_f1_score_list.mean(),final_f1_score_list.std())
    # print('The final ap10 : ',final_ap10.mean(),final_ap10.std())
    # print('The final ap20 : ',final_ap20.mean(),final_ap20.std())
    # print('The final ap50 : ',final_ap50.mean(),final_ap50.std())
    # print('The final auc : ',final_auc.mean(),final_auc.std())           
                    
    


    