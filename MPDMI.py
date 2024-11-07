
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.data_utils import *
from time import time
import numpy as np
import scipy.sparse as sp
import os

import numba
from numba import njit

from torch_scatter import scatter_mean
from numba.typed import List
from utils.parser import parse_args

args = parse_args()
device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")
data_name = args.dataset
@njit
def meta_path_emb(path_dict,n_users):
    head_index = List.empty_list(types.int64)
    # 定义一个空的类型化列表 path_nodes
    path_nodes = List.empty_list(types.int64)
    for ii, path in path_dict.items():
        # nodes = List.empty_list(types.int64)
        for nodeid in path:
            path_nodes.append(int(nodeid))
        head_index.append(int(nodeid))
        # path_nodes = path_nodes + nodes
    head_index = [x - int(n_users) for x in head_index]
    # ii_len = len(head_index)
    # ii_type = [ii_type_single]*ii_len
    return path_nodes,head_index

class Aggregator(nn.Module):
    def __init__(self, n_users,n_items):
        super(Aggregator, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.lamuda = nn.Parameter(torch.tensor(0.5))

    def forward(self,user_emb,item_emb,all_embed,latent_emb,
                ii_instance_file,mean_mat,heads_tensor,tails_tensor,weight,disen_weight_att):

        n_item = self.n_items
        n_users = self.n_users
        n_factors = latent_emb.shape[0]
        gnn_emb = []
        head_dict = []
        tail_dict = []
        for head,tail in zip(heads_tensor,tails_tensor):
            if head < self.n_users:
                continue
            tail_dict.append(tail)
            head_dict.append(head-n_users)
        # tail_dict = torch.tensor(tail_dict).to(device)
        gnn_emb = all_embed[tail_dict]

        head_dict = torch.tensor(head_dict).to(device)
        item_agg = scatter_mean(src=gnn_emb, index= head_dict, dim_size=n_item, dim=0)
        
        if data_name == "Amazon_Book":
            ii_metapahts_list =['ici','ibi','iii','iui']
        elif data_name == "ml-1m":
            ii_metapahts_list =['ici','iii','iui',
                             'icici','iuoui','iuaui','iugui','iuiui']
        elif data_name == "music":
            ii_metapahts_list =['ici','iii','iui',
                             'icici','iciui','iuici','iuiui','iuuii','iiuui']  
        else :
            ii_metapahts_list = []
        path_nodes_3 = List.empty_list(types.int64)
        path_nodes_5 = List.empty_list(types.int64)
        for metapath in ii_metapahts_list:
            outfile = ii_instance_file + metapath + '.paths'
            if os.stat(outfile).st_size == 0:
                continue
            # 获取路径字典
            path_dict = meta_paths_to_dict(outfile)
            # 判断传来的字典为空
            if path_dict==False:
                continue
            # 获取路径表示
            path_nodes_,head_index_ = meta_path_emb(path_dict,n_users)
            if len(metapath) == 3:
                for node in path_nodes_:
                    path_nodes_3.append(node)
            if len(metapath) == 5:
                for node in path_nodes_:
                    path_nodes_5.append(node)

        path_nodes_emb_3 = all_embed[path_nodes_3]
        path_nodes_emb_5 = all_embed[path_nodes_5]
        num_path_3 = path_nodes_emb_3.shape[0] // 3
        num_path_5 = path_nodes_emb_5.shape[0] // 5
        path_nodes_emb_3 = path_nodes_emb_3.reshape(num_path_3, 3, 100)
        path_nodes_emb_5 = path_nodes_emb_5.reshape(num_path_5, 5, 100)
        path_nodes_emb_3 = path_nodes_emb_3.mean(axis=1)
        path_nodes_emb_5 = path_nodes_emb_5.mean(axis=1)
        if path_nodes_emb_5.size(0) == 0:
            path_nodes_emb = path_nodes_emb_3
        else:
            path_nodes_emb = torch.cat((path_nodes_emb_3, path_nodes_emb_5), dim=0)
        # 调整温度参数
        temperature =0.005  # 可以尝试不同的温度值，如 0.5, 0.1, 0.01
        # 对每一行应用调整温度的 Softmax 函数
        score_path = torch.mm(path_nodes_emb, latent_emb.t())
        normalized_score = F.softmax(score_path / temperature, dim=1)
        latent_agg = torch.matmul(normalized_score.T, path_nodes_emb)
        # # # # # 从图上来的
        disen_weight = torch.mm(nn.Softmax(dim=-1)(disen_weight_att),
                                weight)
        # # # # 图+路径
        # lamuda2 = torch.sigmoid(self.lamuda)
        disen_weight =0.4*disen_weight+0.6*latent_agg
        disen_weight = F.normalize(disen_weight)
        # disen_weight = disen_weight
        # disen_weight = F.normalize(disen_weight)

        # 利用矩阵乘法操作
        # 这个矩阵的元素表示了每个用户与每个潜在因子之间的得分或相似度。
        # 在使用之前创建 score_

        score_ = torch.mm(user_emb, latent_emb.t()) # [n_users, n_factors]/
        # # 归一化
        score = nn.Softmax(dim=1)(score_).unsqueeze(-1)  # [n_users, n_factors, 1]

        user_agg = torch.sparse.mm(mean_mat, item_emb)
        user_agg = user_agg * (disen_weight * score).sum(dim=1) + user_agg   # [n_users, channel]

        return user_agg,item_agg



class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self,n_hops, n_users,n_items,meta_len):
        super(GraphConv, self).__init__()
        # 是一个容器，用于存储神经网络的子模块（layers）或层。
        self.convs = nn.ModuleList()
        self.n_users = n_users
        self.n_items = n_items
        self.meta_len = meta_len
        self.n_factors = args.n_factors
        initializer = nn.init.xavier_uniform_
        weight = initializer(torch.empty(self.meta_len, 100))  # not include interact
        self.weight = nn.Parameter(weight)  # [n_relations - 1, in_channel]

        disen_weight_att = initializer(torch.empty(self.n_factors, self.meta_len))
        self.disen_weight_att = nn.Parameter(disen_weight_att)
        
        for i in range(n_hops):
            # 多次更新
            self.convs.append(Aggregator(n_users=n_users,n_items = self.n_items))

    def forward(self,user_emb,item_emb, all_embed, latent_emb,ii_instance_file,mean_mat,heads_tensor,tails_tensor):
        user_res_emb = user_emb
        item_res_emb = item_emb
        # 遍历每一层聚合层
        for i in range(len(self.convs)):
            # 返回的entity_emb是聚合后的实体项目向量
            user_agg,item_agg = self.convs[i](user_emb,item_emb,all_embed, latent_emb,ii_instance_file,mean_mat,heads_tensor,tails_tensor,self.weight,self.disen_weight_att)
            #归一化操作
            user_emb = F.normalize(user_agg)
            item_emb = F.normalize(item_agg)
            """result emb"""
            # 相加得到最终项目和用户向量
            user_res_emb = torch.add(user_res_emb, user_emb)
            item_res_emb = torch.add(item_res_emb, item_emb)
        
        return user_res_emb,item_res_emb


class Recommender(nn.Module):
    # 包括用户和项目数量，用户的项目向量
    # 目的是从用户最新的项目学习出兴趣向量
    def __init__(self, args,user_num,item_num,nodes_num,meta_len):
        super(Recommender, self).__init__()
        self.n_users = user_num
        self.n_items = item_num
        self.nodes_num = nodes_num # u+i+c+b
        self.meta_len = meta_len 
        # 几层聚合层，其实就就是几跳
        self.context_hops = args.context_hops
        self.decay = 1e-5
        self.sim_decay = 1e-4
        self.emb_size = args.channel
        # 超参数
        self.n_factors = args.n_factors
        # 计算距离的方式
        self._init_weight()
        # 转换为可以在模型训练过程中通过反向传播来更新的可学习参数，从而使它们成为模型中需要学习的部分。
        self.all_embed = nn.Parameter(self.all_embed)
        self.latent_emb = nn.Parameter(self.latent_emb)
        # 初始化神经网络模型
        self.gcn = self._init_model()
    

    # 初始化神经网络模型的参数，并将稀疏矩阵转换为适用于神经网络的数据结构。
    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        # 这个参数通常表示所有节点的嵌入向量。
        self.all_embed = initializer(torch.empty(self.nodes_num, self.emb_size))
        # 它通常表示潜在因子（latent factor）的嵌入向量
        self.latent_emb = initializer(torch.empty(self.n_factors, self.emb_size))
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)
    def _init_model(self):
        return GraphConv(n_hops=self.context_hops,
                         n_users=self.n_users,
                         n_items = self.n_items,
                         meta_len = self.meta_len)


    def forward(self, batch_users,batch_item,ii_instance_file,neg,mean_mat_list,heads_tensor,tails_tensor):

        # 将嵌入矩阵 all_embed 中的嵌入向量分别提取为 user_emb 和 item_emb

        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:self.n_users+self.n_items, :]
        mean_mat = self._convert_sp_mat_to_sp_tensor(mean_mat_list).to(device)
        self.ii_instance_file = ii_instance_file
        self.batch_item= batch_item
        self.batch_users= batch_users
        self.heads_tensor = heads_tensor
        self.tails_tensor = tails_tensor
        self.mean_mat= mean_mat

        # 得到最终的实体嵌入向量和用户嵌入向量，以及距离误差
        user_gcn_emb, entity_gcn_emb = self.gcn(user_emb,item_emb,
                                                     self.all_embed,
                                                     self.latent_emb,
                                                     ii_instance_file,mean_mat,
                                                     heads_tensor,tails_tensor,
                                                    )
        
        u_e = user_gcn_emb[batch_users]
        batch_item = batch_item - self.n_users
        pos_e, neg_e = entity_gcn_emb[batch_item], entity_gcn_emb[neg]

        return self.create_bpr_loss(u_e, pos_e, neg_e)

    
    def generate(self):
        # 获取用户和项目向量
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:self.n_users+self.n_items, :]
        # 再一次启动卷积层，更新聚合参数
        return self.gcn(user_emb,item_emb,
                                        self.all_embed,
                                        self.latent_emb,
                                        self.ii_instance_file,
                                        self.mean_mat,
                                        self.heads_tensor,
                                        self.tails_tensor,
                                        )

    def rating(self, u_g_embeddings, i_g_embeddings):
        # 矩阵相乘操作，.t()操作表示转置
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())
    

    def create_bpr_loss(self, users, pos_items, neg_items):

        batch_size = users.shape[0]
        # 乘积加求和，得到最终得分，pos_scores 包含了每个正样本的得分
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        # mf_loss 包含了正负样本之间的差异的对数 Sigmoid 损失
        # BPR损失
        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        # 正则化损失
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        # 嵌入损失
        emb_loss = self.decay * regularizer / batch_size
        # 相关性损失，在潜在意图那里做区分

        # print("losstiem:",time_loss2-time_loss)
        return mf_loss + emb_loss , mf_loss, emb_loss
