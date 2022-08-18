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
# from sklearn.metrics import classification_report,roc_auc_score,accuracy_score
from sklearn.metrics import classification_report,roc_auc_score,accuracy_score,f1_score,precision_score,recall_score

# import wandb
from Model.Model import *
from Model.utils import *
from sklearn.metrics import classification_report,roc_auc_score
parser = argparse.ArgumentParser(description='Training model')

import warnings
warnings.filterwarnings('ignore')

'''Data Setting'''
parser.add_argument('--data_dir', type=str, default='/public/VC_Project/Data',
                    help='The address of csv data.')
parser.add_argument('--Model_dir', type=str, default='/public/VC_Project/Saved_model',
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
parser.add_argument('--n_layers_clf', type=int, default=1,
                    help='Number of layers of Convs.')
parser.add_argument('--use_norm', type=bool, default=True,
                    help='Use norm?')
parser.add_argument('--gpus', type=str, default='cuda:0',
                    help='')
parser.add_argument('--atts_shape',type=int, default=49)

'''Optimization arguments'''
parser.add_argument('--optimizer', type=str, default='adam',
                    choices=['adamw', 'adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate.')
parser.add_argument('--batch_size', type=int, default=256,
                    help='batch size')
parser.add_argument('--n_epoch_update', type=int, default=100,
                    help='Number of epoch to init the embedding')
parser.add_argument('--init_epoch', type=int, default=50)
parser.add_argument('--n_epoch_init', type=int, default=20,
                    help='Number of the epoch to update the embedding')
parser.add_argument('--n_epoch_predict', type=int, default=50,
                    help='Number of epoch to run')
parser.add_argument('--lr_clf', type=float, default=1e-3,
                    help='Learning rate.')
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--hiddem_dim',type=int,default=32)
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

# wandb.init(project="VC", name=args.task_name)
# wandb.config.update(vars(args))


setup_seed(7)

# load data
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

if args.optimizer == 'adamw':
    optimizers = torch.optim.AdamW
elif args.optimizer == 'adam':
    optimizers = torch.optim.Adam
elif args.optimizer == 'sgd':
    optimizers = torch.optim.SGD
elif args.optimizer == 'adagrad':
    optimizers = torch.optim.Adagrad

for c, e in zip(new_companies, labels):
    assert len(c) == len(e)


save_path = args.Model_dir+'/'+args.task_name + '/'
# Model_emb  = torch.load('/public/VC_Project/Saved_model/NC/Embedding.pth').cpu()
Model_emb = torch.load(save_path+'Embedding.pth').cpu()#.to(args.gpus)
args.embedding_dim = 120
args.hidden_dim = 120
print(args.embedding_dim,args.hidden_dim)


# 这个att会对结果产生影响
embeddings = []
# atts=np.load('atts.npy')
atts = np.load('/public/VC_Project/atts.npy')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
atts = scaler.fit_transform(atts)

atts=torch.tensor(atts)#.to(args.gpus)
for i, e in zip(Model_emb.embed, graph_edges):
    i = i.weight.data.clone().detach()#.to(args.gpus)

    temp = torch.zeros_like(i)#.to(args.gpus)
    temp[e.unique()] += i[e.unique()]
    temp=torch.cat([temp,atts],1)
    embeddings.append(temp.unsqueeze(0))

embeddings = torch.cat(embeddings, 0)
embeddings = torch.cat([embeddings,torch.zeros([embeddings.shape[0],embeddings.shape[1],120-embeddings.shape[2]])],-1).to(args.gpus)

if args.comparison and not args.train_comparison:
    data_compair = pd.read_pickle(args.data_dir+'/PitchBook/sim_edge_remove_stopwords.pkl')
    No = torch.tensor(data_compair['No'].values)
    edges_compair = torch.tensor([data_compair['edge_i'].values,data_compair['edge_j'].values])
    edges_compair_date = torch.tensor(data_compair['date'])
    edges_compair_value = torch.tensor(data_compair['value'])
    
    print(edges_compair.shape)
    edges_compair = edges_compair[:,No<=args.count]
    print(edges_compair.shape)
    edges_compair_date = edges_compair_date[No<=args.count]
    edges_compair_value = edges_compair_value[No<=args.count]
    
    edges_compair = edges_compair[:,edges_compair_value>args.sim_threshold]
    print(edges_compair.shape)
    edges_compair_date = edges_compair_date[edges_compair_value>args.sim_threshold]
    edge_date = torch.cat([edge_date,edges_compair_date])
    edge_type = torch.cat([edge_type,11*torch.ones_like(edges_compair_date)])
    for i in range(edge_date.max()+1):
        graph_edges[i] = torch.cat([graph_edges[i],edges_compair[:,edges_compair_date<=i]],1)

traning_index = list(range(1, 23))
val_index = list(range(23, 25))
test_index = [i+84 for i in range(1, 13)]
#test_index = [i for i in range(1, 1+12*13) if i>84 ]

# if args.dynamic_clf:
for _ in range(15):
    best_auc = 0
    best_acc = 0
    # predictor = Predict_model(
    # args.embedding_dim, args.N_heads, args.n_layers_clf, args, args.dropout).to(args.gpus)
    # print('Define the model: ',args.embedding_dim,args.N_heads,args.n_layers_clf,args.dropout)
    # predictor = Predict_modelv2(args.embedding_dim, args.N_heads, args.n_layers_clf, args, args.dropout).to(args.gpus)
    predictor = Predict_modelv5(args.embedding_dim, args.N_heads, args.n_layers_clf, args, args.dropout).to(args.gpus)
    
    # predictor = Predict_modelv3(args.embedding_dim,args.N_heads,args.n_layers_clf,args,args.dropout).to(args.gpus)
    # predictor = Predict_modelv4(args.embedding_dim,args.N_heads,args.n_layers_clf,args,args.dropout).to(args.gpus)
    optimizer = optimizers([
        {'params': predictor.parameters(), 'lr': args.lr_clf, 'weight_decay': 0.001}
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 800, eta_min=1e-6)
    CRLoss = nn.CrossEntropyLoss()
    
    print('######################## in dynamic result ########################')
    for ep in range(args.n_epoch_predict):
        predictor.train()
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

            neighbors = K_hop_nodes(predict_nodes, edges_train[-1]).to(args.gpus)
            edges_train = [K_hop_neighbors(neighbors, i, j)
                           for i, j in zip(edges_train, edge_type_train)]
            # nodes_type_train = [i[2] for i in edges_train]
            edges_type_train = [i[1].to(args.gpus) for i in edges_train]
            edges_train = [i[0] for i in edges_train]
            train_embeds = train_embeds.to(args.gpus)
            nodetypes = nodetypes.to(args.gpus)
            
            # ###################################################################################
            # '''
            # 之前的代码，利用binary_cross_entropy_with_logits
            # '''
            # prediction, att = predictor(train_embeds, edges_train, nodetypes, edges_type_train,
            #                             neighbors, predict_nodes)

            # prediction = prediction.view(-1)
            # label = labels[index-1].clone().detach().to(args.gpus).float()
            # pos_weight = (label.shape[0]-label.sum())/label.sum()
            # loss = F.binary_cross_entropy_with_logits(
            #     prediction, label, pos_weight=pos_weight)
            # ####################################################################################
            
             ###################################################################################
            '''
            利用binary_cross_entropy_
            '''
            prediction, _ = predictor(train_embeds, edges_train, nodetypes, edges_type_train,
                                        neighbors, predict_nodes)
            label = labels[index-1].clone().detach().to(args.gpus)
            # print('The size of label : ',label.size())
            # print('The size of prediction : ',prediction.size())
            loss = CRLoss(prediction,label)
            ####################################################################################
            
            
            
            optimizer.zero_grad()
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #    scaled_loss.backward()
            loss.backward()
            # bar.set_description('%.2f'%(loss))
            optimizer.step()
        # print('########################### Finishing training ###########################')
        torch.cuda.empty_cache()
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
                edge_type_val = [edge_type[edge_date <= i].to(args.gpus)
                                 for i in range(index-args.n_predict_step+1, index+1)]
                val_embeds = embeddings[index-args.n_predict_step+1:index+1].to(args.gpus)
                predict_nodes = new_companies[index-1].to(args.gpus)
                neighbors = K_hop_nodes(predict_nodes, edges_val[-1]).to(args.gpus)
                #edges_val = [K_hop_neighbors(neighbors, i) for i in edges_val]
                prediction, _ = predictor(val_embeds, edges_val, nodetypes, edge_type_val,
                                            neighbors, predict_nodes)
                
                
                # ###########################################################
                # '''
                # 之前的代码
                # '''
                # prediction = prediction.view(-1).sigmoid()
                # val_prediction.append(prediction.view(-1).cpu())
                # val_label.append(labels[index-1].clone().detach().view(-1))
                # month += [index for i in prediction.view(-1)]
                # ############################################################
                
                ###########################################################
                '''
                之前的代码
                '''
                _,prediction = torch.max(prediction.data,1)
                # print('The size in val: ',(prediction.clone().detach().cpu()).size())
                # print('The size in val label :',(labels[index-1].clone().detach().view(-1).cpu()).size())
                val_prediction.append(prediction.clone().detach().cpu())
                val_label.append(labels[index-1].clone().detach().view(-1).cpu())
                month += [index for i in prediction]
                ############################################################
                
                # print(val_prediction[-1].shape,val_label[-1].shape)
            val_prediction = torch.cat(val_prediction)
            val_label = torch.cat(val_label)
            # print('The size of val prediction : ',val_prediction.size())
            # print('The size of val label : ',val_label.size())
            # print('val prediction : ',val_prediction[:10])
            # print('val label : ',val_label[:10])
            # print('########################### Finishing Evaluate ###########################')
            #print(calc_apatk(val_prediction, val_label, month))
            val_ap = np.mean(
                list(calc_apatk(val_prediction, val_label, month).values()))
            # print(val_label)
            auc_val, aupr_val, f1_val, best_threshold = eval_metric(
                val_label, val_prediction)
            test_prediction = []
            test_label = []
            month = []
            ap = {}
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
                prediction, _ = predictor(test_embeds, edges_test, nodetypes, edge_type_test,
                                            neighbors, predict_nodes)
                
                # ############################################################
                # prediction = prediction.view(-1).sigmoid()
                # atts.append(att)
                # test_prediction.append(prediction.view(-1).cpu())
                # test_label.append(labels[index-1].view(-1).clone().detach())
                # month += [index for i in prediction.view(-1)]
                # ##############################################################
                
                ############################################################
                _,prediction = torch.max(prediction.data,1)
                
                # atts.append(att)
                # print('The tests prediction size: ',(prediction.detach().cpu()).size())
                # print('The tests label size : ',labels[index-1].clone().detach().view(-1).cpu().size())
                
                test_prediction.append(prediction.detach().cpu())
                test_label.append(labels[index-1].clone().detach().view(-1).cpu())
                month += [index for i in prediction]
                ##############################################################
                
            test_prediction = torch.cat(test_prediction)
            test_label = torch.cat(test_label)
            # print(test_label.sum(),test_label.shape)
            predict_companies = torch.cat(predict_companies)
            auc_test, aupr_test, f1_test = eval_metric(
                test_label, test_prediction, best_threshold)
            acc_val = accuracy_score(test_label,test_prediction)
            if best_auc < auc_val:
            # if best_acc < acc_val:
                
                # print('The epoch to update : ',ep,index)
                best_auc = auc_val
                # best_acc = acc_val
                # save_path = args.Model_dir+'/'+args.task_name + '/'
                # check_dir(save_path)
                # torch.save(Model_emb, save_path+'Best_clf.pth')
                test_ap = calc_apatk(test_prediction, test_label, month)
                test_auc = roc_auc_score(test_label,test_prediction)
                test_rep = classification_report(test_label,test_prediction,digits=6,target_names=['false','true'])
                
                # print('classification result :\n',test_rep)
                # print('AUC result: ',test_auc)
                # result = pd.DataFrame(
                #     {'att': torch.cat(atts).cpu().detach(), 'predict': test_prediction, 'label': test_label, 'month': month, 'Index': predict_companies})
                # result.to_excel('result_%.3f_%.3f.xlsx'%(ap['5'],ap['10']))
                # print(ap)
            # wandb.log({'test_auc': auc_test, 'test_f1': f1_test, 'aupr_test': aupr_test,
            #            'best_auc': best_auc, 'auc_val': auc_val, 'aupr_val': aupr_val, 'f1_val': f1_val})
            # #ap = calc_apatk(test_prediction, test_label, month)
            # wandb.log(ap)
    # print('##################################################################################')
    # print('The final result : ')
    print('AP : ',test_ap)
    print('Test auc : ',test_auc)
    print('Classification Report: \n',test_rep)
    # print('##################################################################################')
    