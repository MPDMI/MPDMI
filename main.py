import random
import torch
import numpy as np

from utils.parser import parse_args
from prettytable import PrettyTable
from time import time
from collections import defaultdict
from MPDMI import Recommender
import pickle
from utils.data_utils import *
import torch.utils.data as Data
from pathlib import Path
from tqdm import tqdm
from utils.evaluate import test
from random import choices
from scipy import sparse

n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0
def find_neg(user_num,all_items,user_history_dic,batch_users):
    
    neg_list=[]
    for user in batch_users:
        uid = user.cpu().item()
        # 记录用户交互过的物品
        bought = set(user_history_dic[uid])
        # 该用户没有交互过的物品的id
        remain = list(all_items.difference(bought))
        neg_list.append(choices(remain,k=1))

    neg_list = [item[0] - user_num for item in neg_list]
    neg_list = torch.tensor(neg_list)


    return neg_list


def model_pro(args,batch_index,user_num,item_num,model,index,train_file,test_file, 
              ii_instance_file,ui_path,loss_file_txt,all_items,user_history_dic,user_mean_item,time_count,best,best_epoch):

    # 得到转换格式之后的训练集和测试集,转换为torch格式
    train_data = load_train_test_data(train_file)
    # ii_instance_file = "./"+str(model_name)+"/data/"+str(data_name)+"/data_list/15/path/all_ui_ii_instance_paths/ii_random_form.paths"
    BATCH_SIZE = args.batch_size
    # 创建了两个数据加载器（train_loader 和 test_loader），
    # 用于将训练和测试数据划分成小批次进行处理，这在深度学习中是常见的做法。
    train_loader = Data.DataLoader(
        dataset=train_data,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=False,  #是否随机
        # num_workers=0,  #工作线程数
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) 
    # 加载图
    graph_path = ui_path
    G = pickle.load(open(graph_path, 'rb')) #node 包括 user/item/brand/category/also_bought
    heads_tensor = []
    tails_tensor = []
    mean_mat_list = sparse.load_npz(user_mean_item)
    # 使用循环提取头部和尾部
    for edge in G.edges:
        heads_tensor.append(edge[0])
        tails_tensor.append(edge[1])
    train_start_time = time()
    running_loss = 0.0
    progress_bar = tqdm(enumerate(train_loader), desc="Processing")
    for step, batch in progress_bar:
        # batch_labels = batch[:, 2].to(device)
        batch_users = batch[:, 0].to(device)
        batch_item = batch[:, 1].to(device)
        neg = find_neg(user_num,all_items,user_history_dic,batch_users).to(device)
        batch_loss, _, _ = model(batch_users,batch_item,ii_instance_file,neg,mean_mat_list,heads_tensor,tails_tensor)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        running_loss += batch_loss.item()
    train_time = time() - train_start_time
    print(f'index: {index}, training loss: {running_loss}, train time: {train_time}')
    with open(loss_file_txt, "a") as file:
        file.write(f'index: {index}, training loss: {running_loss}, train time: {train_time}\n')
    if index <=time_count-2:
        return running_loss,best,best_epoch
    if batch_index%5!=0:

        return running_loss,best,best_epoch
    ret = test(model,user_num,item_num)
    train_res = PrettyTable()
    train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "recall", "ndcg", "precision", "hit_ratio"]
    train_res.add_row(
            [0, 0,0,0,ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio']]
        )
    print(train_res)
    recall= ret['recall']
    ndcg =ret['ndcg']
    pre = ret['precision']
    hit = ret['hit_ratio']
    with open(loss_file_txt, "a") as file:
        file.write(f'[epoch, 0,0,0,{recall}, {ndcg}, {pre}", "{hit}"]\n')
    if ret['recall'][0] >=best:
        best= ret['recall'][0] 
        best_epoch = batch_index


    return running_loss,best,best_epoch




if __name__ == '__main__':
    args = parse_args()

    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")
    """fix the random seed"""
    seed = 2024
    # 这段代码是用于在深度学习中设置随机数种子以确保实验的可重复性。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model_name = args.model_name
    data_name = args.dataset

    user_rate_item_df = pd.read_csv("./"+str(model_name)+"/data/"+str(data_name)+"/old/user_rate_item.csv", header=None, sep=',')
    time_line =0
    if data_name == "Amazon_Book":
        time_line =100
        meta_len = 4
    elif data_name == "music":
        time_line =300
        meta_len = 8
    elif data_name == "ml-1m":
        time_line =10000  
        meta_len = 9
    time_count= int(max(user_rate_item_df[3])/time_line)+1


    # 记录了所有用户的行为
    user_history_dic = defaultdict(list)
    user_history_file = "./"+str(model_name)+"/data/"+str(data_name)+"/data_list/"+str(time_count-1)+"/path/user_history/user_history.txt"
    with open(user_history_file) as f:
        for line in f:
            s = line.split()
            uid = int(s[0])
            user_history_dic[uid] = [int(item) for item in s[1:]]

    type_path = "./"+str(model_name)+"/data/"+str(data_name)+"/data_list/"+str(time_count-1)+"/refine/map.type2id"
    type2id = pickle.load(open(type_path, 'rb'))
    users_list = type2id['user']
    item_list = type2id['item']
    all_items = set(type2id['item'])
    # 遍历所有键并计算每个键包含的值的数量
    nodes_num = sum(len(value) for value in type2id.values())
    user_num = len(users_list)
    item_num = len(item_list)
    print(user_num)
    print(item_num)
    # node
    print(nodes_num)
    model = Recommender(args,user_num,item_num,nodes_num,meta_len).to(device)
    print("start training ...")
    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False
    best = -1
    best_epoch = -1

    for batch_index in range(args.epoch):
        batch_loss = 0
        batch_train_start_time = time()
        for index in range(time_count):
            # if index < 62:
            #     continue
            train_file = "./"+str(model_name)+"/data/"+str(data_name)+"/data_list/"+str(index)+"/negs/training.links"
            test_file = "./"+str(model_name)+"/data/"+str(data_name)+"/data_list/"+str(index)+"/negs/testing.links"
            ii_instance_file = "./"+str(model_name)+"/data/"+str(data_name)+"/data_list/"+str(index)+"/path/all_ii_paths/"
            user_mean_item = "./"+str(model_name)+"/data/"+str(data_name)+"/data_list/"+str(index)+"/user_mean_item.npz"
            ui_path = "./"+str(model_name)+"/data/"+str(data_name)+"/data_list/"+str(index)+"/graph.nx"
            loss_file_txt = "./"+str(model_name)+"/log/"+str(data_name)+"_no_gra.txt"
    
            loss1,best,best_epoch = model_pro(args,batch_index,user_num,item_num,model,index,train_file,test_file,
                    ii_instance_file,ui_path,loss_file_txt,all_items,user_history_dic,user_mean_item,time_count,best,best_epoch)
            batch_loss+=loss1
            print("success:"+str(index))
            # print("index:",time.time()-time_start)

        
        with open(loss_file_txt, "a") as file:
            file.write(f'finish:batch: {batch_index}, training loss: {batch_loss}, train time: {time()-batch_train_start_time}\n')
        print("finish:batch:"+str(batch_index)+","+"batch_loss:"+str(batch_loss))

        # if batch_index > best_epoch +25:
        #     break
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        # if batch_index%5==0:
        #     cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
        #                                                                     stopping_step, expected_order='acc',
        #                                                                     flag_step=10)
        # if should_stop:
        #     print('early stopping at %d, recall@20:%.4f' % (batch_index, cur_best_pre_0))
        #     break
    















