import datetime
import itertools
import json
import math
import os
import random

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.utils import (degree, get_laplacian, is_undirected,
                                   k_hop_subgraph, negative_sampling, subgraph,
                                   to_dense_adj, to_undirected,
                                   train_test_split_edges)


def eval_metric(y_true, y_score, threshold=None):
    '''
    evaluate the result, return the AUC, aupr and f1 score.
    '''
    auc_test = roc_auc_score(y_true, y_score)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    aupr_test = auc(recall, precision)
    if threshold is None:
        best_threshold = 0
        best_f1 = 0
        for i in thresholds:
            f1_test = f1_score(y_true, y_score > i)
            if f1_test > best_f1:
                best_threshold = i
                best_f1 = f1_test
        return auc_test, aupr_test, best_f1, best_threshold
    else:
        f1_test = f1_score(y_true, y_score > threshold)
        return auc_test, aupr_test, f1_test


def eval(y_true, y_score):

    auc_test = roc_auc_score(y_true, y_score)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    aupr_test = auc(recall, precision)
    pos_threshold = 0.5
    f1_test = f1_score(y_true, y_score > pos_threshold)

    return auc_test, aupr_test, f1_test


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def calc_auc(labels, predict_prob):
    return roc_auc_score(labels, predict_prob)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels, predict_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    return roc_auc


# def calc_apatk(predicts, label, month, ks=[5, 10, 20, 50, 100]):
#     result = pd.DataFrame(
#         {'predict': predicts, 'label': label, 'month': month})
#     groups = result.groupby('month')
#     patk = {}
#     for k in ks:
#         patk[k] = []

#     for month in set(month):
#         for k in ks:
#             temp_df = result.loc[result['month'] == month]
#             patk[k].append(temp_df.sort_values('predict')[
#                            'label'].iloc[-k:].values.mean())
#     apatk = {}
#     for k in ks:
#         apatk[str(k)] = np.mean(patk[k])
#     return apatk
def calc_apatk(predicts, label, month, ks=[5, 10, 20, 50, 100]):
#     print(predicts.size(),label.size())
    result = pd.DataFrame(
        {'predict': predicts, 'label': label, 'month': month})
    groups = result.groupby('month')
    patk = {}
    for k in ks:
        patk[k] = []
    groups = dict(list(groups))
    for month in groups:
        for k in ks:
            temp_df = groups[month]
            patk[k].append(temp_df.sort_values('predict')['label'].iloc[-k:].values.mean())
    apatk = {}
    for k in ks:
        apatk[str(k)] = np.mean(patk[k])
    return apatk


def evaluate_mertic(labels, predict, threshold):
    return classification_report(labels, predict > threshold, digits=4)


def masked_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you ``nans``.
    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector + (mask + 1e-45).log()
    return torch.nn.functional.softmax(vector, dim=dim)


def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


def isin(ar1, ar2):
    '''
    for each values in ar1, if it in ar2, the mask will be 1, otherwise it will be 0
    '''
    return (ar1[..., None] == ar2).any(-1)


def load_crunchbase(path, save_dir):

    # Load Crunchbase
    Inv = pd.read_csv(path+'Investments.csv')
    Exit = pd.read_csv(path+'Exit.csv')


def get_position(title):
    if 'Chief Executive Officer' in title:
        return 1
    elif 'Chief Technology Officer' in title:
        return 2
    elif 'Chief Operating Officer' in title:
        return 3
    elif 'Chief Financial Officer' in title:
        return 4
    elif 'Chief Product Officer' in title:
        return 5
    elif 'Board Member' in title:
        return 6
    elif 'Co-Founder' in title:
        return 7
    elif 'Founder' in title:
        return 8
    elif 'Advisor' in title:
        return 9
    else:
        return 10


def load_pitchbook(path, save_dir):

    # Load Data
    print('*'*10+'Loading CSV file'+'*'*10)
    PeopleAffiliatedDealRelation = pd.read_csv(
        path+'/PitchBook/PeopleAffiliatedDealRelation.dat', sep='|')
    CompanyTeamRelation = pd.read_csv(
        path+'/PitchBook/CompanyTeamRelation.dat', sep='|')
    Exit = pd.read_excel(
        path+'/PitchBook/Exit/US_Exit_Data_Jan2007-to-June-2010.xlsx', header=7)
    Exit = pd.concat([Exit, pd.read_excel(
        path+'/PitchBook/Exit/US_Exit_Data_Jun2010-to-Sept-2012.xlsx', header=7)])
    Exit = pd.concat([Exit, pd.read_excel(
        path+'/PitchBook/Exit/US_Exit_Data_Sept-2012-to-April-2014.xlsx', header=7)])
    Exit = pd.concat([Exit, pd.read_excel(
        path+'/PitchBook/Exit/US_Exit_Data_April-2014-to-Feb-2016.xlsx', header=7)])
    Exit = pd.concat([Exit, pd.read_excel(
        path+'/PitchBook/Exit/US_Exit_Data_Feb-2016-to-May-2018.xlsx', header=7)])
    Exit = pd.concat([Exit, pd.read_excel(
        path+'/PitchBook/Exit/US_Exit_Data_May-2018-to-Jan-2020.xlsx', header=7)])
    Exit = pd.concat([Exit, pd.read_excel(
        path+'/PitchBook/Exit/Europe, Canada, Israel Exits 2007to2019-pp1-pp10.xlsx', header=7)])
    Exit = pd.concat([Exit, pd.read_excel(
        path+'/PitchBook/Exit/Europe, Canada, Isreal Exits 2007to2019-pp11-pp21.xlsx', header=7)])
    Exit = pd.concat([Exit, pd.read_excel(
        path+'/PitchBook/Exit/Europe, Canada, Israel Exits 2007to2019-pp21-pp32.xlsx', header=7)])
    Exit = pd.concat([Exit, pd.read_excel(
        path+'/PitchBook/Exit/Europe, Canada, Israel Exits 2007to2019-pp22-end.xlsx', header=7)])
    Exit = pd.concat([Exit, pd.read_excel(
        path+'/PitchBook/Exit/Exit Canada Europe PP1-pp2 (top 13 on pp3).xlsx', header=7)])
    Exit = pd.concat([Exit, pd.read_excel(
        path+'/PitchBook/Exit/Exits in Asia.xlsx', header=7)])
    print('*'*10+'Proparing Data'+'*'*10)
    # PeopleAffiliatedDealRelation = PeopleAffiliatedDealRelation.loc[~PeopleAffiliatedDealRelation['DealType'].isin(
    #     ['Merger/Acquisition', "Merger of Equals", 'Reverse Merger', "IPO", 'Out of Business', 'Dividend Recapitalization', 'Sale-Lease back facility', 'Buyout/LBO',
    #      'Bankruptcy: Admin/Reorg', 'Bankruptcy: Liquidation', 'Share Repurchase']
    # )]
    PeopleAffiliatedDealRelation = PeopleAffiliatedDealRelation.loc[PeopleAffiliatedDealRelation['DealType'].isin(
        ['Early Stage VC', 'Seed Round', 'Angel (individual)', 'Accelerator/Incubator', 'Later Stage VC',
         'Corporate', 'Restart - Angel', 'Restart - Early VC', 'Restart - Later VC', 'Restart - Corporate']
    )]

    Exit = Exit.loc[~Exit['Deal Type'].isin(
        ['Dividend', 'Dividend Recapitalization', 'Bankruptcy: Admin/Reorg', 'Bankruptcy: Liquidation'])]
    Exit['Deal Date'] = pd.to_datetime(Exit['Deal Date'])
    PeopleAffiliatedDealRelation = PeopleAffiliatedDealRelation.loc[~PeopleAffiliatedDealRelation['DealDate'].isna(
    )]
    CompanyTeamRelation = CompanyTeamRelation.loc[CompanyTeamRelation['CompanyID'].isin(
        PeopleAffiliatedDealRelation['CompanyID'])]
    CompanyTeamRelation = CompanyTeamRelation.loc[~CompanyTeamRelation['StartDate'].isna(
    )]
    CompanyTeamRelation = CompanyTeamRelation.drop(
        CompanyTeamRelation.loc[CompanyTeamRelation['StartDate'].map(lambda x:int(x[-4:])) > 2020].index, axis=0)
    PeopleAffiliatedDealRelation['DealDate'] = pd.to_datetime(
        PeopleAffiliatedDealRelation['DealDate'])
    CompanyTeamRelation['StartDate'] = pd.to_datetime(
        CompanyTeamRelation['StartDate'])
    CompanyTeamRelation['EndDate'] = pd.to_datetime(
        CompanyTeamRelation['EndDate'])
    PeopleAffiliatedDealRelation = PeopleAffiliatedDealRelation.sort_values(
        'DealDate').reset_index(drop=True)
    PeopleAffiliatedDealRelation['first'] = PeopleAffiliatedDealRelation['DealID'].isin(
        PeopleAffiliatedDealRelation.drop_duplicates('CompanyID')['DealID'])
    PeopleAffiliatedDealRelation = PeopleAffiliatedDealRelation.loc[
        PeopleAffiliatedDealRelation['DealDate'] < datetime.datetime(2015, 1, 1)]
    CompanyTeamRelation = CompanyTeamRelation.loc[CompanyTeamRelation['StartDate'] < datetime.datetime(
        2015, 1, 1)]

    # Build Graph
    print('*'*10+'Building Graph'+'*'*10)
    ID2index = {}
    all_ids = list(set(PeopleAffiliatedDealRelation['CompanyID'].tolist(
    )+PeopleAffiliatedDealRelation['PersonID'].tolist()+CompanyTeamRelation['PersonID'].tolist()+CompanyTeamRelation['CompanyID'].tolist()))
    for i, j in enumerate(all_ids):
        ID2index[j] = i
    PeopleAffiliatedDealRelation['PersonIndex'] = PeopleAffiliatedDealRelation['PersonID'].map(
        ID2index)
    CompanyTeamRelation['PersonIndex'] = CompanyTeamRelation['PersonID'].map(
        ID2index)
    PeopleAffiliatedDealRelation['CompanyIndex'] = PeopleAffiliatedDealRelation['CompanyID'].map(
        ID2index)
    CompanyTeamRelation['CompanyIndex'] = CompanyTeamRelation['CompanyID'].map(
        ID2index)
    CompanyTeamRelation['PositionType'] = CompanyTeamRelation['FullTitle'].map(
        lambda x: get_position(x))
    print("max type ID is %d"%(CompanyTeamRelation['PositionType'].max()))
    # for t=0 (before 2007)
    timestep = 0
    edges = []
    all_nodes = []
    graph_nodes = []
    start_date = datetime.datetime(2007, 1, 1)
    temp_PADR = PeopleAffiliatedDealRelation.loc[PeopleAffiliatedDealRelation['DealDate'] < start_date]
    temp_CTR = CompanyTeamRelation.loc[CompanyTeamRelation['StartDate'] < start_date]
    edges += temp_PADR.apply(lambda x: str(
                x['CompanyIndex'])+'-'+str(x['PersonIndex'])+'-0-'+str(timestep), axis=1).tolist()
    edges += temp_CTR.apply(lambda x: str(
                x['PersonIndex'])+'-'+str(x['CompanyIndex'])+'-%d-' % (x['PositionType'])+str(timestep), axis=1).tolist()
    edges += temp_PADR.apply(lambda x: str(
                x['CompanyIndex'])+'-'+str(x['PersonIndex'])+'-0-'+str(timestep), axis=1).tolist()
    edges += temp_CTR.apply(lambda x: str(
                x['PersonIndex'])+'-'+str(x['CompanyIndex'])+'-%d-' % (x['PositionType'])+str(timestep), axis=1).tolist()
    edges = list(set(edges))
    graph_edges = edges
    new_edges = [graph_edges]
    graph_nodes += [int(i.split('-')[0]) for i in edges]
    graph_nodes += [int(i.split('-')[1]) for i in edges]
    graph_nodes = [list(set(graph_nodes))]
    all_nodes = graph_nodes[-1]
    new_companies = []
    labels = []
    new_nodes_list=[]
    # add nodes and edges to graph
    for year in range(2007, 2015):
        for month in tqdm.tqdm(range(1, 13)):
            timestep += 1
            if month == 12:
                start_date = datetime.datetime(year, 12, 1)
                end_date = datetime.datetime(year+1, 1, 1)
                exit_temp = Exit.loc[Exit['Deal Date']
                                     < datetime.datetime(year+6, 1, 1)]
            else:
                start_date = datetime.datetime(year, month, 1)
                end_date = datetime.datetime(year, month+1, 1)
                exit_temp = Exit.loc[Exit['Deal Date'] <
                                     datetime.datetime(year+5, month+1, 1)]
            if year in [2007, 2008, 2014]:
                new_company = PeopleAffiliatedDealRelation.loc[PeopleAffiliatedDealRelation['DealDate'] >= start_date].loc[PeopleAffiliatedDealRelation['DealDate'] < end_date].drop_duplicates(
                    'CompanyID').loc[PeopleAffiliatedDealRelation['first'] == True]
                label = new_company['CompanyID'].isin(exit_temp['Company ID'])
                new_company = new_company['CompanyIndex']
                new_companies.append(new_company.values)
                labels.append(torch.tensor([1 if i else 0 for i in label]))

                assert new_companies[-1].shape[-1] == labels[-1].shape[-1]
            elif year in [2009, 2010, 2011, 2012, 2013]:
                new_company = PeopleAffiliatedDealRelation.loc[PeopleAffiliatedDealRelation['DealDate'] >= start_date].loc[PeopleAffiliatedDealRelation['DealDate'] < end_date].drop_duplicates(
                    'CompanyID').loc[PeopleAffiliatedDealRelation['first'] == True]  # ['CompanyIndex']
                new_company = new_company.loc[new_company['CompanyID'].isin(
                    exit_temp['Company ID'])]
                label = new_company['CompanyID'].isin(exit_temp['Company ID'])
                new_company = new_company['CompanyIndex']
                if len(new_company) != 0:
                    new_companies.append(new_company.values)
                    labels.append(torch.tensor([1 if i else 0 for i in label]))
                    assert new_companies[-1].shape[-1] == labels[-1].shape[-1]
                else:
                    new_companies.append(np.array(-1))
                    labels.append(np.array(-1))
            temp_PADR = PeopleAffiliatedDealRelation.loc[PeopleAffiliatedDealRelation['DealDate']
                                                         < end_date].loc[PeopleAffiliatedDealRelation['DealDate'] >= start_date]
            temp_CTR = CompanyTeamRelation.loc[CompanyTeamRelation['StartDate']
                                               < end_date].loc[CompanyTeamRelation['StartDate'] >= start_date]

            # new nodes
            nodes = temp_PADR['CompanyIndex'].tolist()+temp_CTR['CompanyIndex'].tolist(
            )+temp_PADR['PersonIndex'].tolist()+temp_CTR['PersonIndex'].tolist()
            nodes = list(set(nodes))
            new_nodes = [i for i in nodes if i not in all_nodes]
            all_nodes += new_nodes
            graph_nodes.append(new_nodes)
            
            # new edges
            edges = temp_PADR.apply(lambda x: str(
                x['PersonIndex'])+'-'+str(x['CompanyIndex'])+'-0-'+str(timestep), axis=1).tolist()
            edges += temp_CTR.apply(lambda x: str(
                x['PersonIndex'])+'-'+str(x['CompanyIndex'])+'-%d-' % (x['PositionType'])+str(timestep), axis=1).tolist()
            edges += temp_PADR.apply(lambda x: str(
                x['CompanyIndex'])+'-'+str(x['PersonIndex'])+'-0-'+str(timestep), axis=1).tolist()
            edges += temp_CTR.apply(lambda x: str(
                x['CompanyIndex'])+'-'+str(x['PersonIndex'])+'-%d-' % (x['PositionType'])+str(timestep), axis=1).tolist()
            edges = list(set([i for i in edges if i not in graph_edges]))
            graph_edges = list(set(graph_edges+edges))
            new_edges.append(edges)
    print('*'*10+'Saving Data'+'*'*10)
    edge_date = torch.tensor([int(j.split('-')[-1]) for j in graph_edges])
    edge_type = torch.tensor([int(j.split('-')[-2]) for j in graph_edges])
    graph_edges = torch.tensor(
        [[int(j) for j in i.split('-')[:2]] for i in graph_edges]).long()
    
    new_edges = [torch.tensor([[int(j) for j in i.split('-')[:2]]
                               for i in j]) for j in new_edges]
    graph_nodes = [torch.tensor(i) for i in graph_nodes]
    all_nodes = torch.tensor(all_nodes)
    index2id = {value: key for key, value in ID2index.items()}
    new_nodes = graph_nodes
    edges_all=[]
    new_added_edges=[]
    with h5py.File(save_dir+'/data.h5', 'w') as out:

        out.create_dataset('all_nodes', data=all_nodes)
        out.create_dataset('graph_edges', data=graph_edges)
        out.create_dataset('edge_date', data=edge_date)
        out.create_dataset('edge_type', data=edge_type)
        for i in range(len(new_edges)):
            if i == 0:
                edges_all.append(graph_edges[edge_date<=i].transpose(1,0))
            else:
                new_added_edges.append(graph_edges[edge_date==i])
                edges_all.append(graph_edges[edge_date<=i].transpose(1,0))
                out.create_dataset('new_edges_%d' % (i), data=graph_edges[edge_date==i])
                out.create_dataset('new_companies_%d' %
                                   (i), data=new_companies[i-1])
                out.create_dataset('labels_%d' %
                                   (i), data=labels[i-1])
                new_node = graph_nodes[i].unique()
                out.create_dataset('new_nodes_%d' %
                                   (i), data=new_node)
    with open(save_dir+'/ID2index.json', 'w') as file_obj:
        json.dump(ID2index, file_obj)
    nodetypes = []
    for i in range(max(list(index2id.keys()))+1):
        if 'P' in index2id[i]:
            nodetypes.append(0)
        else:
            nodetypes.append(1)
    nodetypes = torch.tensor(nodetypes)
    return edges_all, edge_date, edge_type, all_nodes, new_companies, labels, new_nodes, new_added_edges, nodetypes, ID2index


def load_pitchbook_new(path, save_dir):

    # Load Data
    print('*'*10+'Loading CSV file'+'*'*10)
    PeopleAffiliatedDealRelation = pd.read_csv(
        path+'/PitchBook/PeopleAffiliatedDealRelation.dat', sep='|')
    CompanyTeamRelation = pd.read_csv(
        path+'/PitchBook/CompanyTeamRelation.dat', sep='|')
    Exit = pd.read_excel(
        path+'/PitchBook/Exit/US_Exit_Data_Jan2007-to-June-2010.xlsx', header=7)
    Exit = pd.concat([Exit, pd.read_excel(
        path+'/PitchBook/Exit/US_Exit_Data_Jun2010-to-Sept-2012.xlsx', header=7)])
    Exit = pd.concat([Exit, pd.read_excel(
        path+'/PitchBook/Exit/US_Exit_Data_Sept-2012-to-April-2014.xlsx', header=7)])
    Exit = pd.concat([Exit, pd.read_excel(
        path+'/PitchBook/Exit/US_Exit_Data_April-2014-to-Feb-2016.xlsx', header=7)])
    Exit = pd.concat([Exit, pd.read_excel(
        path+'/PitchBook/Exit/US_Exit_Data_Feb-2016-to-May-2018.xlsx', header=7)])
    Exit = pd.concat([Exit, pd.read_excel(
        path+'/PitchBook/Exit/US_Exit_Data_May-2018-to-Jan-2020.xlsx', header=7)])
    Exit = pd.concat([Exit, pd.read_excel(
        path+'/PitchBook/Exit/Europe, Canada, Israel Exits 2007to2019-pp1-pp10.xlsx', header=7)])
    Exit = pd.concat([Exit, pd.read_excel(
        path+'/PitchBook/Exit/Europe, Canada, Isreal Exits 2007to2019-pp11-pp21.xlsx', header=7)])
    Exit = pd.concat([Exit, pd.read_excel(
        path+'/PitchBook/Exit/Europe, Canada, Israel Exits 2007to2019-pp21-pp32.xlsx', header=7)])
    Exit = pd.concat([Exit, pd.read_excel(
        path+'/PitchBook/Exit/Europe, Canada, Israel Exits 2007to2019-pp22-end.xlsx', header=7)])
    Exit = pd.concat([Exit, pd.read_excel(
        path+'/PitchBook/Exit/Exit Canada Europe PP1-pp2 (top 13 on pp3).xlsx', header=7)])
    Exit = pd.concat([Exit, pd.read_excel(
        path+'/PitchBook/Exit/Exits in Asia.xlsx', header=7)])
    print('*'*10+'Proparing Data'+'*'*10)
    # PeopleAffiliatedDealRelation = PeopleAffiliatedDealRelation.loc[~PeopleAffiliatedDealRelation['DealType'].isin(
    #     ['Merger/Acquisition', "Merger of Equals", 'Reverse Merger', "IPO", 'Out of Business', 'Dividend Recapitalization', 'Sale-Lease back facility', 'Buyout/LBO',
    #      'Bankruptcy: Admin/Reorg', 'Bankruptcy: Liquidation', 'Share Repurchase']
    # )]
    PeopleAffiliatedDealRelation = PeopleAffiliatedDealRelation.loc[PeopleAffiliatedDealRelation['DealType'].isin(
        ['Early Stage VC', 'Seed Round', 'Angel (individual)', 'Accelerator/Incubator', 'Later Stage VC',
         'Corporate', 'Restart - Angel', 'Restart - Early VC', 'Restart - Later VC', 'Restart - Corporate']
    )]

    Exit = Exit.loc[~Exit['Deal Type'].isin(
        ['Dividend', 'Dividend Recapitalization', 'Bankruptcy: Admin/Reorg', 'Bankruptcy: Liquidation'])]
    Exit['Deal Date'] = pd.to_datetime(Exit['Deal Date'])
    PeopleAffiliatedDealRelation = PeopleAffiliatedDealRelation.loc[~PeopleAffiliatedDealRelation['DealDate'].isna(
    )]
    CompanyTeamRelation = CompanyTeamRelation.loc[CompanyTeamRelation['CompanyID'].isin(
        PeopleAffiliatedDealRelation['CompanyID'])]
    CompanyTeamRelation = CompanyTeamRelation.loc[~CompanyTeamRelation['StartDate'].isna(
    )]
    CompanyTeamRelation = CompanyTeamRelation.drop(
        CompanyTeamRelation.loc[CompanyTeamRelation['StartDate'].map(lambda x:int(x[-4:])) > 2020].index, axis=0)
    PeopleAffiliatedDealRelation['DealDate'] = pd.to_datetime(
        PeopleAffiliatedDealRelation['DealDate'])
    CompanyTeamRelation['StartDate'] = pd.to_datetime(
        CompanyTeamRelation['StartDate'])
    CompanyTeamRelation['EndDate'] = pd.to_datetime(
        CompanyTeamRelation['EndDate'])
    PeopleAffiliatedDealRelation = PeopleAffiliatedDealRelation.sort_values(
        'DealDate').reset_index(drop=True)
    PeopleAffiliatedDealRelation['first'] = PeopleAffiliatedDealRelation['DealID'].isin(
        PeopleAffiliatedDealRelation.drop_duplicates('CompanyID')['DealID'])
    PeopleAffiliatedDealRelation = PeopleAffiliatedDealRelation.loc[
        PeopleAffiliatedDealRelation['DealDate'] < datetime.datetime(2021, 1, 1)]
    CompanyTeamRelation = CompanyTeamRelation.loc[CompanyTeamRelation['StartDate'] < datetime.datetime(
        2021, 1, 1)]

    # Build Graph
    print('*'*10+'Building Graph'+'*'*10)
    ID2index = {}
    all_ids = list(set(PeopleAffiliatedDealRelation['CompanyID'].tolist(
    )+PeopleAffiliatedDealRelation['PersonID'].tolist()+CompanyTeamRelation['PersonID'].tolist()+CompanyTeamRelation['CompanyID'].tolist()))
    for i, j in enumerate(all_ids):
        ID2index[j] = i
    PeopleAffiliatedDealRelation['PersonIndex'] = PeopleAffiliatedDealRelation['PersonID'].map(
        ID2index)
    CompanyTeamRelation['PersonIndex'] = CompanyTeamRelation['PersonID'].map(
        ID2index)
    PeopleAffiliatedDealRelation['CompanyIndex'] = PeopleAffiliatedDealRelation['CompanyID'].map(
        ID2index)
    CompanyTeamRelation['CompanyIndex'] = CompanyTeamRelation['CompanyID'].map(
        ID2index)
    CompanyTeamRelation['PositionType'] = CompanyTeamRelation['FullTitle'].map(
        lambda x: get_position(x))
    print("max type ID is %d"%(CompanyTeamRelation['PositionType'].max()))
    # for t=0 (before 2007)
    timestep = 0
    edges = []
    all_nodes = []
    graph_nodes = []
    start_date = datetime.datetime(2007, 1, 1)
    temp_PADR = PeopleAffiliatedDealRelation.loc[PeopleAffiliatedDealRelation['DealDate'] < start_date]
    temp_CTR = CompanyTeamRelation.loc[CompanyTeamRelation['StartDate'] < start_date]
    edges += temp_PADR.apply(lambda x: str(
                x['CompanyIndex'])+'-'+str(x['PersonIndex'])+'-0-'+str(timestep), axis=1).tolist()
    edges += temp_CTR.apply(lambda x: str(
                x['PersonIndex'])+'-'+str(x['CompanyIndex'])+'-%d-' % (x['PositionType'])+str(timestep), axis=1).tolist()
    edges += temp_PADR.apply(lambda x: str(
                x['CompanyIndex'])+'-'+str(x['PersonIndex'])+'-0-'+str(timestep), axis=1).tolist()
    edges += temp_CTR.apply(lambda x: str(
                x['PersonIndex'])+'-'+str(x['CompanyIndex'])+'-%d-' % (x['PositionType'])+str(timestep), axis=1).tolist()
    edges = list(set(edges))
    graph_edges = edges
    new_edges = [graph_edges]
    graph_nodes += [int(i.split('-')[0]) for i in edges]
    graph_nodes += [int(i.split('-')[1]) for i in edges]
    graph_nodes = [list(set(graph_nodes))]
    all_nodes = graph_nodes[-1]
    new_companies = []
    labels = []
    new_nodes_list=[]
    # add nodes and edges to graph
    for year in range(2007, 2021):
        for month in tqdm.tqdm(range(1, 13)):
            timestep += 1
            if month == 12:
                start_date = datetime.datetime(year, 12, 1)
                end_date = datetime.datetime(year+1, 1, 1)
                exit_temp = Exit.loc[Exit['Deal Date']
                                     < datetime.datetime(year+6, 1, 1)]
            else:
                start_date = datetime.datetime(year, month, 1)
                end_date = datetime.datetime(year, month+1, 1)
                exit_temp = Exit.loc[Exit['Deal Date'] <
                                     datetime.datetime(year+5, month+1, 1)]
            if year in [2007, 2008, 2014, 2015, 2016, 2017, 2018, 2019, 2020]:
                new_company = PeopleAffiliatedDealRelation.loc[PeopleAffiliatedDealRelation['DealDate'] >= start_date].loc[PeopleAffiliatedDealRelation['DealDate'] < end_date].drop_duplicates(
                    'CompanyID').loc[PeopleAffiliatedDealRelation['first'] == True]
                label = new_company['CompanyID'].isin(exit_temp['Company ID'])
                new_company = new_company['CompanyIndex']
                new_companies.append(new_company.values)
                labels.append(torch.tensor([1 if i else 0 for i in label]))

                assert new_companies[-1].shape[-1] == labels[-1].shape[-1]
            elif year in [2009, 2010, 2011, 2012, 2013]:
                new_company = PeopleAffiliatedDealRelation.loc[PeopleAffiliatedDealRelation['DealDate'] >= start_date].loc[PeopleAffiliatedDealRelation['DealDate'] < end_date].drop_duplicates(
                    'CompanyID').loc[PeopleAffiliatedDealRelation['first'] == True]  # ['CompanyIndex']
                new_company = new_company.loc[new_company['CompanyID'].isin(
                    exit_temp['Company ID'])]
                label = new_company['CompanyID'].isin(exit_temp['Company ID'])
                new_company = new_company['CompanyIndex']
                if len(new_company) != 0:
                    new_companies.append(new_company.values)
                    labels.append(torch.tensor([1 if i else 0 for i in label]))
                    assert new_companies[-1].shape[-1] == labels[-1].shape[-1]
                else:
                    new_companies.append(np.array(-1))
                    labels.append(np.array(-1))
            temp_PADR = PeopleAffiliatedDealRelation.loc[PeopleAffiliatedDealRelation['DealDate']
                                                         < end_date].loc[PeopleAffiliatedDealRelation['DealDate'] >= start_date]
            temp_CTR = CompanyTeamRelation.loc[CompanyTeamRelation['StartDate']
                                               < end_date].loc[CompanyTeamRelation['StartDate'] >= start_date]

            # new nodes
            nodes = temp_PADR['CompanyIndex'].tolist()+temp_CTR['CompanyIndex'].tolist(
            )+temp_PADR['PersonIndex'].tolist()+temp_CTR['PersonIndex'].tolist()
            nodes = list(set(nodes))
            new_nodes = [i for i in nodes if i not in all_nodes]
            all_nodes += new_nodes
            graph_nodes.append(new_nodes)
            
            # new edges
            edges = temp_PADR.apply(lambda x: str(
                x['PersonIndex'])+'-'+str(x['CompanyIndex'])+'-0-'+str(timestep), axis=1).tolist()
            edges += temp_CTR.apply(lambda x: str(
                x['PersonIndex'])+'-'+str(x['CompanyIndex'])+'-%d-' % (x['PositionType'])+str(timestep), axis=1).tolist()
            edges += temp_PADR.apply(lambda x: str(
                x['CompanyIndex'])+'-'+str(x['PersonIndex'])+'-0-'+str(timestep), axis=1).tolist()
            edges += temp_CTR.apply(lambda x: str(
                x['CompanyIndex'])+'-'+str(x['PersonIndex'])+'-%d-' % (x['PositionType'])+str(timestep), axis=1).tolist()
            edges = list(set([i for i in edges if i not in graph_edges]))
            graph_edges = list(set(graph_edges+edges))
            new_edges.append(edges)
    print('*'*10+'Saving Data'+'*'*10)
    edge_date = torch.tensor([int(j.split('-')[-1]) for j in graph_edges])
    edge_type = torch.tensor([int(j.split('-')[-2]) for j in graph_edges])
    graph_edges = torch.tensor(
        [[int(j) for j in i.split('-')[:2]] for i in graph_edges]).long()
    
    new_edges = [torch.tensor([[int(j) for j in i.split('-')[:2]]
                               for i in j]) for j in new_edges]
    graph_nodes = [torch.tensor(i) for i in graph_nodes]
    all_nodes = torch.tensor(all_nodes)
    index2id = {value: key for key, value in ID2index.items()}
    new_nodes = graph_nodes
    edges_all=[]
    new_added_edges=[]
    with h5py.File(save_dir+'/data_new.h5', 'w') as out:

        out.create_dataset('all_nodes', data=all_nodes)
        out.create_dataset('graph_edges', data=graph_edges)
        out.create_dataset('edge_date', data=edge_date)
        out.create_dataset('edge_type', data=edge_type)
        for i in range(len(new_edges)):
            if i == 0:
                edges_all.append(graph_edges[edge_date<=i].transpose(1,0))
            else:
                new_added_edges.append(graph_edges[edge_date==i])
                edges_all.append(graph_edges[edge_date<=i].transpose(1,0))
                out.create_dataset('new_edges_%d' % (i), data=graph_edges[edge_date==i])
                out.create_dataset('new_companies_%d' %
                                   (i), data=new_companies[i-1])
                out.create_dataset('labels_%d' %
                                   (i), data=labels[i-1])
                new_node = graph_nodes[i].unique()
                out.create_dataset('new_nodes_%d' %
                                   (i), data=new_node)
    with open(save_dir+'/ID2index_new.json', 'w') as file_obj:
        json.dump(ID2index, file_obj)
    nodetypes = []
    for i in range(max(list(index2id.keys()))+1):
        if 'P' in index2id[i]:
            nodetypes.append(0)
        else:
            nodetypes.append(1)
    nodetypes = torch.tensor(nodetypes)
    return edges_all, edge_date, edge_type, all_nodes, new_companies, labels, new_nodes, new_added_edges, nodetypes, ID2index

def load_pb_from_h5(path):
    with h5py.File(path+'/data.h5', 'r') as fin:
        new_edges = []
        new_companies = []
        new_nodes = []
        labels = []
        all_nodes = torch.tensor(fin['all_nodes'][:])
        max_id = max([int(i.split('_')[-1])
                      for i in fin.keys() if '_' in i and len(i.split('_')) > 2])
        graph_edges = torch.tensor(
            fin['graph_edges'][:]).long()
        edge_type = torch.tensor(
            fin['edge_type'][:])
        edge_date = torch.tensor(
            fin['edge_date'][:])
        edges_all=[]
        for i in range(max_id+1):
            if i == 0:
                edges_all.append(graph_edges[edge_date<=i].transpose(1,0).long())
            else:
                edges_all.append(graph_edges[edge_date<=i].transpose(1,0).long())
                new_edges.append(graph_edges[edge_date==i])#torch.tensor(fin['new_edges_%d' % (i)][:]))
                new_companies.append(torch.tensor(
                    fin['new_companies_%d' % (i)][:]))
                new_nodes.append(torch.tensor(
                    fin['new_nodes_%d' % (i)][:]))
                labels.append(torch.tensor(
                    fin['labels_%d' % (i)][:]))
    with open(path+'/ID2index.json', 'r') as file_obj:
        ID2index = json.load(file_obj)
    index2id = {value: key for key, value in ID2index.items()}
    nodetypes = []
    for i in range(max(list(index2id.keys()))+1):
        if 'P' in index2id[i]:
            nodetypes.append(0)
        else:
            nodetypes.append(1)
    nodetypes = torch.tensor(nodetypes).long()
    return edges_all, edge_date, edge_type, all_nodes, new_companies, labels, new_nodes, new_edges, nodetypes, ID2index


def random_drop_edge(edges, ratio_same):
    pass


def random_add_edge(edges, ratio):
    ##
    if not is_undirected(edges):
        edges = to_undirected(edges)
    degrees = degree(edges[0])
    indexs = []
    for i in range(len(degrees)):
        indexs += [i]*math.ceil(degrees[i]*ratio)
    target = torch.tensor(indexs).view(1, -1)
    nodes = target.unique()
    indexs = torch.randint(low=0, high=nodes.shape[0], size=target.shape)
    source = nodes[indexs[0]].view(1, -1)
    # source = target[:,idx]
    # source = torch.tensor(np.random.choice(edges.unique().cpu(),target.shape))#.to(edges.device)
    return torch.cat([source, target], 0).to(edges.device)


def random_sub_graph(remain_nodes, edges, ratio):
    all_nodes = edges.unique()
    other_nodes = all_nodes[~isin(remain_nodes, all_nodes)]
    other_nodes = np.random.choice(other_nodes, other_nodes.shape[0]*ratio)
    sub, _ = subgraph(torch.cat([all_nodes, other_nodes]), edges)
    return sub


def check_dir(path):
    if os.path.exists(path):
        return
    else:
        os.makedirs(path)
        return check_dir(path)


def K_hop_nodes(nodes, edges, k=1):
    nodes, edge_index, _, _ = k_hop_subgraph(nodes, k, edges)
    return nodes


def K_hop_neighbors(nodes, edges, edge_type, k=1):
    nodes, edge_index, _, edge_mask = k_hop_subgraph(
        nodes, k, edges, num_nodes=max(nodes.max()+1, edges.max()+1))
    return edge_index, edge_type[edge_mask]#, node_type[nodes]


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def cos_sim_selct(edges, embeddings, target_nodes, number_of_neg=1):
    device = edges.device
    edges = edges.cpu()
    embeddings = embeddings.cpu()
    target_nodes = target_nodes.cpu()
    with torch.no_grad():
        edges, edge_weight = get_laplacian(edges, normalization='sym')
        laplacian = to_dense_adj(edges)[0]
        print(laplacian.shape)
        laplacian[edges[0], :][:, edges[1]] += edge_weight.abs()
        embeddings = laplacian.mm(embeddings)  # .unsqueeze(0)
        similarity = F.cosine_similarity(
            embeddings[None, :, :], embeddings[:, None, :], dim=-1).abs().numpy()
        # temp = torch.zeros_like(similarity)
        # temp[:, target_nodes] = similarity[:, target_nodes]
        # similarity = temp-temp.min(1)[0].unsqueeze(-1)
        # similarity = temp/temp.sum(1).unsqueeze(-1)
        # print(similarity.sum(1),similarity.sum(0))
        selected_index = []
        for i in target_nodes:
            p = similarity[i][target_nodes]
            p /= p.sum()
            selected_index += [[i, j] for j in np.random.choice(
                target_nodes, number_of_neg, p=p)]
        pos_index = torch.tensor([[i, i] for i in target_nodes]).to(device)
        selected_index = torch.tensor(selected_index).to(device)
        labels = torch.tensor([1 for i in pos_index] +
                              [0 for i in selected_index]).to(device)
        # print(selected_index.shape,pos_index.shape)
        selected_index = torch.cat(
            [pos_index, selected_index], 0).transpose(0, 1)
    return selected_index, labels.float()


def random_neg_select(edges, source_nodes, number_of_neg=1):
    nodes = edges.unique()

    target_nodes = np.random.choice(
        nodes.cpu(), source_nodes.cpu().shape[0]*number_of_neg)
    target_nodes = torch.tensor(target_nodes).to(edges.device)

    pos = torch.tensor([[i for i in source_nodes], [
                       i for i in source_nodes]]).to(edges.device)
    neg = torch.tensor([[i for i in source_nodes]*number_of_neg, [
                       i for i in target_nodes]]).to(edges.device)

    labels = [1 for i in source_nodes]+[0 for i in target_nodes]
    labels = torch.tensor(labels).to(edges.device)

    return torch.cat([pos, neg], 1), labels.float()
