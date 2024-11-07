#!usr/bin/env python
# -*- coding:utf-8 _*-

import pandas as pd
from collections import defaultdict, Counter
import json
import pickle
import networkx as nx
from pathlib import Path
from time import time

from utils.parser import parse_args
import numpy as np
import scipy.sparse as sp
from scipy import sparse
import csv
import re



def process_initial_ratings(ratings_csv,output_ratings_csv):
    ratings = pd.read_csv(ratings_csv, header=None, sep=',')
    # fit_item_df = pd.read_csv(fit_item_csv, header=None)
    print(len(ratings))
    # 53111621
    ratings = ratings.drop_duplicates()
    print(len(ratings))
    # 51169571
    # ratings[[0, 1]] = ratings[[1, 0]]
    # (user,item,rating,timestamp)
    # project_ids = fit_item_df.iloc[:, 1].tolist()



    print(len(set(ratings[0])))
    # 69205
    print(len(set(ratings[1])))
    # 16882
    # ratings = ratings[ratings[1].isin(project_ids)]
    # print(len(set(ratings[0])))
    # 69205
    # print(len(set(ratings[1])))

    # print(project_ids)

    # 80559
    
    """
    music
    903330   users    get 59189
    112222   items   get 66369

    book
    69205 users    get 53630
    16882 items    get 16882

    video
    1540618 users   get 94323
    71982 items     get 50541
    """ 
    
    
    c_u = Counter(ratings[0])


    filtered_counter = {k: v for k, v in c_u.items() if v >= 4}
    # filtered_counter_item = {k: v for k, v in c_i.items() if v >= 4}

    # 提取所有键  
    selected_users = list(filtered_counter.keys())  
    # selected_items = list(filtered_counter_item.keys())  


    item_filter = ratings[ratings[0].isin(selected_users)]

    # item_filter = item_filter[item_filter[1].isin(selected_users)]


    print(len(set(item_filter[0])))
    print(len(set(item_filter[1])))



    # 按第四列排序,即按时间排序
    item_filter = item_filter.sort_values(by=[3])

    pre = None
    pre_time = 0
    i = 0
    time = []
    braak = []

    # 遍历每一行，新增一列的值
    for index, row in item_filter.iterrows():
        if pre != row[0]:
            # 如果第0列与上一列内容不一致，重置递增计数
            i += 1
            if pre_time == row[3]:
            # 如果第0列与上一列内容一致，且第3列与上一列内容一致，新增列的值不变
                i-=1
        else:
            # 同一用户
            # 否则，递增计数
            i += 1 
        # 将计数值添加到列表
        time.append(i)
        if i % 100 ==0 :
            braak.append(1)
        else:
            braak.append(0)            
        # 更新 pre 和 pre_time
        pre = row[0]
        pre_time = row[3]
    item_filter = item_filter.drop(3, axis=1)

    item_filter[3] = time
    item_filter[4] = braak

    # 设置截取点
    j=0
    flag =-2
    for index, row in item_filter.iterrows():   
        if flag == j-1 and row[4] ==1:
            braak[flag] = 0
        if row[4]==1:
            flag = j
        j+=1

    item_filter = item_filter.drop(4, axis=1)    
    item_filter[4] = braak    

    re_user = Counter(item_filter[0])
    print(len(re_user))
    re_item = Counter(item_filter[1])
    print(len(re_item))
    print()
    #20693
    
    item_filter.to_csv(output_ratings_csv, header=None, index=None, sep=',')

    print(f'saved to {output_ratings_csv}')



def get_item_meta(metafile,outputoldfolder):
    movies = metafile +"music.dat"
    users = metafile +"user_friends.dat"
    item_category = []
    user_friend = []
    user_age = []
    user_occupation = []
    #("category","tech1","description","fit","title","also_buy","tech2","brand","featur","rank","also_view","details","main_cat","similar_item"，data,price,asin,imageURL)
    # with open('./old/meta_Musical_Instruments.json', 'r') as f:
    with open(movies, 'r', encoding='latin-1') as file_item:
        lines = file_item.readlines()[1:]
    for line in lines:
        genres_list =[]
        # print(line)
        user_id,movie_id, tag_id, genres_string = line.strip().split('::')
        # genres = genres_string.split('|')
        # genres_list.extend(genres)
        # for cate in genres_list:
        item_category.append(['i_'+movie_id, 'c_'+tag_id])
    with open(users, 'r') as file_item:
        # 跳过第一行（标题行）
        next(file_item)
        
        # 逐行读取文件内容
        for line in file_item:
            # 去除每行末尾的换行符，并用制表符分隔
            user_id, friend_id = line.strip().split()
            user_friend.append(['u_'+user_id, 'u_'+friend_id])
        # user_age.append(['u_'+user_id, 'a_'+age])
        # user_occupation.append(['u_'+user_id, 'o_'+occupation])



    item_category_df = pd.DataFrame(item_category)
    user_gender_df = pd.DataFrame(user_friend)
    # user_age_df = pd.DataFrame(user_age)
    # user_occupation_df = pd.DataFrame(user_occupation)
    item_category_df=item_category_df.drop_duplicates()
    user_gender_df = user_gender_df.drop_duplicates()
    # user_age_df = user_age_df.drop_duplicates()
    # user_occupation_df = user_occupation_df.drop_duplicates()

    ic_filename = outputoldfolder + 'item_category.csv'
    ug_filename = outputoldfolder + 'user_gender.csv'
    # ua_filename = outputoldfolder + 'user_age.csv'
    # uo_filename = outputoldfolder + 'user_occupation.csv'
    item_category_df.to_csv(ic_filename, header=None, index=None, sep=',')
    user_gender_df.to_csv(ug_filename, header=None, index=None, sep=',')
    # user_age_df.to_csv(ua_filename, header=None, index=None, sep=',')
    # user_occupation_df.to_csv(uo_filename, header=None, index=None, sep=',')
    print(f'saved items meta to folder: {outputoldfolder}')

def form_ids(oldrefinefolder,file_path,refine_path,little_path,index,item_category_df,user_gender_df):

    """
    refine item_category.csv,item_brand.csv,item_item.csv according to user_rate_item.csv
    """

    all_user_rate_item_df = pd.read_csv("./"+str(model_name)+"/data/"+str(data_name)+"/old/user_rate_item.csv", header=None, sep=',')
    user_rate_item_df = pd.read_csv(file_path, header=None, sep=',')
    this_user_rate_item_df = pd.read_csv(little_path, header=None, sep=',')

    # print(user_rate_item_df)
    # 获取user_rate_item_df第二列中的唯一值
    # 目的是为了获取总图的总节点数

    all_user_set = all_user_rate_item_df[0].unique()
    all_item_set = all_user_rate_item_df[1].unique()

    # 目的是为了构建出当前的图
    user_set = user_rate_item_df[0].unique()
    item_set = user_rate_item_df[1].unique()
    # 目的是为了生成当前用户的行为数据集
    this_user_set = this_user_rate_item_df[0].unique()
    this_item_set = this_user_rate_item_df[1].unique()


    # 这段代码的目的似乎是从 item_category_df 数据框中筛选出只包含在 item_set 中的行
    # 并创建一个包含所有不同产品类别的集合 category_set，

    # 筛选出所有项目的集合
    all_item_category_df = item_category_df

    # 筛选出只包含当前项目的集合
    item_category = item_category_df[item_category_df[0].isin(list(item_set))]
    # 筛选
    all_category_set = all_item_category_df[1].unique()
    category_set = item_category[1].unique()


    all_user_gender_df = user_gender_df[user_gender_df[0].isin(list(all_user_set))& user_gender_df[1].isin(list(all_user_set))]
    user_gender = user_gender_df[user_gender_df[0].isin(list(user_set))]
    all_user_gender_set = all_user_gender_df[1].unique()
    gender_set = user_gender[1].unique()
    # print(item_brand_df)

    # all_user_age_df = user_age_df
    # user_age = user_age_df[user_age_df[0].isin(list(user_set))]
    # all_user_age_set = all_user_age_df[1].unique()
    # age_set = user_age[1].unique()


    # all_user_occupation_df = user_occupation_df
    # user_occupation = user_occupation_df[user_occupation_df[0].isin(list(user_set))]
    # all_user_occupation_set = all_user_occupation_df[1].unique()
    # occupation_set = user_occupation[1].unique()





    # print(item_item_df)
    # 完成了数据的筛选，使之只保存符合要求的物品
    all_refinefolder = oldrefinefolder

    ic_refine = all_refinefolder + 'item_category_refine.csv'
    ug_refine = all_refinefolder + 'user_gender_refine.csv'
    # ua_refine = all_refinefolder + 'user_age_refine.csv'
    # uo_refine = all_refinefolder + 'user_occupation_refine.csv'
        
    all_item_category_df.to_csv(ic_refine, header=None, index=None, sep=',')
    all_user_gender_df.to_csv(ug_refine, header=None, index=None, sep=',')

    # all_user_age_df.to_csv(ua_refine, header=None, index=None, sep=',')
    # all_user_occupation_df.to_csv(uo_refine, header=None, index=None, sep=',')




    # 只包含当前的
    refinefolder = refine_path

    ic_refine = refinefolder + 'item_category_refine.csv'
    ug_refine = refinefolder + 'user_gender_refine.csv'
    # ua_refine = refinefolder + 'user_age_refine.csv'
    # uo_refine = refinefolder + 'user_occupation_refine.csv'


    item_category.to_csv(ic_refine, header=None, index=None, sep=',')
    user_gender.to_csv(ug_refine, header=None, index=None, sep=',')
    # user_age.to_csv(ua_refine, header=None, index=None, sep=',')
    # user_occupation.to_csv(uo_refine, header=None, index=None, sep=',')

 

    """
    generate maps, for further using
    """
    # 创建字典
    name2id = defaultdict(int)
    id2name = defaultdict(str)
    name2type = defaultdict(str)
    id2type = defaultdict(str)
    type2name = defaultdict(list)
    type2id = defaultdict(list)
    # 全部节点数
    all_allnodes = list(all_user_set) + list(all_item_set) + list(all_category_set) 

    allnodes = list(user_set) + list(item_set) + list(category_set) 
    """
    ('Players', 2), ('Latin Percussion', 2) category_set,brand_set
    """
    # print('Latin Percussion' in category_set)
    # print('Latin Percussion' in brand_set)
    
    cc = Counter(allnodes)
    # print(cc.most_common())
    print('all:', len(all_allnodes))
    print('当前时间节点的')
    print('user_set',len(user_set))
    print('item_set:', len(item_set))
    print('category_set:', len(category_set))
    # print('gender_set:', len(gender_set))
    # print('age_set',len(age_set))
    # print('occupation_set:', len(occupation_set))


    '''
    all 25099
    25099
    item_set 20145
    category_set 612
    brand_set 2913
    '''

    i = 0
    # 节点的排列顺序是，先用户再项目，再种类，再品牌
    for name in all_user_set:
        # 用户名和id之间的映射关系
        # 相当于name是键，i是内容
        name2id[name] = i
        id2name[i] = name
        # 用户名和类型的映射关系
        name2type[name] = 'user'
        # id和类型之间的映射关系
        id2type[i] = 'user'
        # 将当前用户名加入到user类型中
        type2name['user'].append(name)
        # 将当前id加入到user类型中
        type2id['user'].append(i)
        i = i + 1

    # print(name2id)
    # print(name2type)
    # print(type2name)

    for name in all_item_set:
        name2id[name] = i
        id2name[i] = name
        name2type[name] = 'item'
        id2type[i] = 'item'
        type2name['item'].append(name)
        type2id['item'].append(i)
        i = i + 1
    for name in all_category_set:
        name2id[name] = i
        id2name[i] = name
        name2type[name] = 'category'
        id2type[i] = 'category'
        type2name['category'].append(name)
        type2id['category'].append(i)
        i = i + 1

    # for name in all_user_gender_set:
    #     name2id[name] = i
    #     id2name[i] = name
    #     name2type[name] = 'gender'
    #     id2type[i] = 'gender'
    #     type2name['gender'].append(name)
    #     type2id['gender'].append(i)
    #     i = i + 1
    # for name in all_user_age_set:
    #     name2id[name] = i
    #     id2name[i] = name
    #     name2type[name] = 'age'
    #     id2type[i] = 'age'
    #     type2name['age'].append(name)
    #     type2id['age'].append(i)
    #     i = i + 1
    # for name in all_user_occupation_set:
    #     name2id[name] = i
    #     id2name[i] = name
    #     name2type[name] = 'occupation'
    #     id2type[i] = 'occupation'
    #     type2name['occupation'].append(name)
    #     type2id['occupation'].append(i)
    #     i = i + 1



    name2idfile = refinefolder + 'map.name2id'
    id2namefile = refinefolder + 'map.id2name'
    name2typefile = refinefolder + 'map.name2type'
    id2typefile = refinefolder + 'map.id2type'
    type2namefile = refinefolder + 'map.type2name'
    type2idfile = refinefolder + 'map.type2id'
    # 将之前创建的字典和映射关系数据保存到文件中。
    pickle.dump(name2id, open(name2idfile, 'wb'))
    pickle.dump(id2name, open(id2namefile, 'wb'))
    pickle.dump(name2type, open(name2typefile, 'wb'))
    pickle.dump(id2type, open(id2typefile, 'wb'))
    pickle.dump(type2name, open(type2namefile, 'wb'))
    pickle.dump(type2id, open(type2idfile, 'wb'))

    """
    generate relation file, using new ids
    """


    ic_relation = refinefolder + 'item_category.relation'
    ug_relation = refinefolder + 'user_gender.relation'
    # ua_relation = refinefolder + 'user_age.relation'
    # uo_relation = refinefolder + 'user_occupation.relation'
    ui_relation = refinefolder + 'user_item.relation'
    this_ui_relation = refinefolder + 'this_user_item.relation'


    # ic_relation = refinefolder + 'item_category.relation'
    # ib_relation = refinefolder + 'item_brand.relation'
    # ii_relation = refinefolder + 'item_item.relation'
    # ui_relation = refinefolder + 'user_item.relation'
    # this_ui_relation = refinefolder + 'this_user_item.relation'
    item_category_r = []
    user_gender_r =[]
    user_age_r =[]
    user_occupation_r =[]
    user_item_r = []
    this_user_item_r =[] 

    # 把各个属性代表的id连接起来
    for _, row in item_category.iterrows():
        item_id = name2id[row[0]]
        category_id = name2id[row[1]]
        # [[item_id, category_id]]
        item_category_r.append([item_id, category_id])
    item_category_relation = pd.DataFrame(item_category_r)
    item_category_relation.to_csv(ic_relation, header=None, index=None, sep=',')

    for _, row in user_gender.iterrows():
        item_id = name2id[row[0]]
        gender_id = name2id[row[1]]
        user_gender_r.append([item_id, gender_id])
    user_gender_relation = pd.DataFrame(user_gender_r)
    user_gender_relation.to_csv(ug_relation, header=None, index=None, sep=',')

    # for _, row in user_age.iterrows():
    #     item_id = name2id[row[0]]
    #     age_id = name2id[row[1]]
    #     user_age_r.append([item_id, age_id])
    # user_age_relation = pd.DataFrame(user_age_r)
    # user_age_relation.to_csv(ua_relation, header=None, index=None, sep=',')

    # for _, row in user_occupation.iterrows():
    #     item_id = name2id[row[0]]
    #     occupation_id = name2id[row[1]]
    #     user_occupation_r.append([item_id, occupation_id])
    # user_occupation_relation = pd.DataFrame(user_occupation_r)
    # user_occupation_relation.to_csv(uo_relation, header=None, index=None, sep=',')




    for _, row in user_rate_item_df.iterrows():
        user_id = name2id[row[0]]
        item_id = name2id[row[1]]
        timestamp = int(row[3])
        user_item_r.append([user_id, item_id, timestamp])
    user_item_relation = pd.DataFrame(user_item_r)
    user_item_relation.to_csv(ui_relation, header=None, index=None, sep=',')
    


    for _, row in this_user_rate_item_df.iterrows():
        user_id = name2id[row[0]]
        item_id = name2id[row[1]]
        timestamp = int(row[3])
        this_user_item_r.append([user_id, item_id, timestamp])
    this_user_item_relation = pd.DataFrame(this_user_item_r)
    this_user_item_relation.to_csv(this_ui_relation, header=None, index=None, sep=',')


    print(f'generic id finish')
    return len(all_allnodes)


def gen_graph(num_nodes,refine_path,graph_path):
    # 读取连接点
    refinefolder = refine_path

    ic_relation = refinefolder + 'item_category.relation'
    ug_relation = refinefolder + 'user_gender.relation'
    # ua_relation = refinefolder + 'user_age.relation'
    # uo_relation = refinefolder + 'user_occupation.relation'
    ui_relation = refinefolder + 'user_item.relation'


    item_category = pd.read_csv(ic_relation, header=None, sep=',')
    user_gender = pd.read_csv(ug_relation, header=None, sep=',')
    # user_age = pd.read_csv(ua_relation, header=None, sep=',')
    # user_occupation = pd.read_csv(uo_relation, header=None, sep=',')
    user_item = pd.read_csv(ui_relation, header=None, sep=',')[[0, 1]]

    number_nodes = num_nodes
    # 使用 NetworkX 库创建了一个空图
    G = nx.Graph()
    # 加入节点
    G.add_nodes_from(list(range(number_nodes)))
    # 加入边

    G.add_edges_from(item_category.to_numpy())
    G.add_edges_from(user_gender.to_numpy())
    # G.add_edges_from(user_age.to_numpy())
    # G.add_edges_from(user_occupation.to_numpy())
    G.add_edges_from(user_item.to_numpy())
    # 保存图

    graphfile = graph_path + 'graph.nx'
    pickle.dump(G, open(graphfile, 'wb'))
    print("save graph")

def gen_ui_history(refine_path,path,user_history_path):

    refinefolder = refine_path
    ui_relation = refinefolder + 'user_item.relation'
    user_item_relation = pd.read_csv(ui_relation, header=None, sep=',')
    # print(user_item_relation)
    # 获取用户id
    users = set(user_item_relation[0])
    # print(users)
    user_history_file = user_history_path + 'user_history.txt'
    with open(user_history_file, 'w') as f:
        for user in users:
            # 是从名为 user_item_relation 的数据框中
            # 选择特定用户的历史记录，并按照第三列的值（即时间）进行排序。
            this_user = user_item_relation[user_item_relation[0] == user].sort_values(2)
            path = [user] + this_user[1].tolist()

            for s in path:
                f.write(str(s) + ' ')
            f.write('\n')

 
def gen_ui_history_little(refine_path,path,user_history_path):
    refinefolder = refine_path
    ui_relation = refinefolder + 'this_user_item.relation'
    user_item_relation = pd.read_csv(ui_relation, header=None, sep=',')
    # print(user_item_relation)
    # 获取用户id
    users = set(user_item_relation[0])
    # print(users)
    user_history_file = user_history_path + 'this_user_history.txt'
    with open(user_history_file, 'w') as f:
        for user in users:
            # 是从名为 user_item_relation 的数据框中
            # 选择特定用户的历史记录，并按照第三列的值（即时间）进行排序。
            this_user = user_item_relation[user_item_relation[0] == user].sort_values(2)
            path = [user] + this_user[1].tolist()

            for s in path:
                f.write(str(s) + ' ')
            f.write('\n')
    # 创建边

# 生成新的用户历史记录
    
def refine_data_history(user_history_path,user_additem,user_all_len):
    data_user_history = user_history_path + 'data_user_history.txt'
    this_all_user_history_file = user_history_path + 'user_history.txt'
    with open(this_all_user_history_file, 'r') as all_f:
        for line in all_f:
            s = line.split()
            uid = int(s[0])
            item_history = [int(x) for x in s[1:]]
            if user_additem[uid] == 0:
                with open(data_user_history, 'a+') as f:
                    path = [uid] +item_history
                    for s in path:
                        f.write(str(s) + ' ')
                    f.write('\n')
            else:
                with open(data_user_history, 'a+') as f:
                    user_len = user_all_len[uid]
                    path = [uid] +item_history[:user_len]
                    for s in path:
                        f.write(str(s) + ' ')
                    f.write('\n')






def split_train_test(user_num,user_history_path,user_all_len,last_len,user_additem):
    train = 0.8
    test = 0.2
    user_this_len = [0]*user_num
    training = []
    this_user_history_file = user_history_path + 'this_user_history.txt'
    this_all_user_history_file = user_history_path + 'user_history.txt'

    # 计算训练
    with open(this_all_user_history_file, 'r') as all_f:
        for line in all_f:
            s = line.split()
            uid = int(s[0])
            item_history = [int(x) for x in s[1:]]
            user_this_len[uid] = len(item_history)
    with open(this_user_history_file, 'r') as f:
        for line in f:
            s = line.split()
            uid = int(s[0])
            if user_additem[uid] == 0:
                # 1.小于
                # 2.等于
                # 3.大于
                if user_this_len[uid] < user_all_len[uid]:
                    item_history = [int(x) for x in s[1:]] 
                    # h = len(item_history)
                    training.append([uid] + item_history)

                elif user_this_len[uid] == user_all_len[uid]:
                    item_history = [int(x) for x in s[1:]] 
                    # h = len(item_history)
                    training.append([uid] + item_history)

                    user_additem[uid] = 1
                else:
                    h = user_all_len[uid]- last_len[uid]
                    item_history = [int(x) for x in s[1:h]] 
                    training.append([uid] + item_history)

                    user_additem[uid] = 1
    refine_data_history(user_history_path,user_additem,user_all_len)

    training_file = user_history_path + 'this_training'
    pickle.dump(training, open(training_file, 'wb'))
    last_len = user_this_len
    return last_len,user_additem

def read_interaction_data(all_user_history_path):
    data = []
    with open(all_user_history_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            user_id = int(parts[0])
            item_ids = [int(item_id) for item_id in parts[1:]]
            data.append((user_id, item_ids))
    return data
def _si_norm_lap(adj):
    # D^{-1}A
    rowsum = np.array(adj.sum(1))
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    norm_adj = d_mat_inv.dot(adj)
    return norm_adj.tocoo()

def neg_sample(user_history_path,negs_path):
    # 加载训练集和测试集
    training_file = user_history_path + 'this_training'
    # testing_file = user_history_path + 'this_testing'
    # testing = pickle.load(open(testing_file, 'rb'))
    training = pickle.load(open(training_file, 'rb'))  # uid, item1, item2,

    # 创建了一个字典，键为用户id，属性为行为记录
    # 这是为了选出负样本，所以采用的是user_history而不是this_user_history

    training_negs = []
    for user_record in training:
        uid = user_record[0]
        positive = [[uid, item, 1] for item in user_record[1:]]
        training_negs = training_negs + positive 
    # 保存生成训练集的样本数据，其中包括正样本和负样本。
    training_negs_tf = pd.DataFrame(training_negs)
    training_negs_file = negs_path +'training.links'
    training_negs_tf.to_csv(training_negs_file, header=None, index=None, sep=',')
    # # 测试集\
    # test_negs = []
    # for user_record in testing:
    #     uid = user_record[0]
    #     positive = [[uid, item, 1] for item in user_record[1:]]
    #     test_negs = test_negs + positive 
    # test_negs_tf = pd.DataFrame(test_negs)
    # testing_negs_file = negs_path + 'testing.links'
    # test_negs_tf.to_csv(testing_negs_file, header=None, index=None, sep=',')
    print(f'save neg 0 sampled links ... finish')


if __name__ == '__main__':
    
    # please remember to change most_common(2200)

    args = parse_args()
    model_name = args.model_name
    data_name = "music"

    # fit_item_csv = "./"+str(model_name)+"/data_pre/"+str(data_name)+"/old/filtered_output.csv"
    # 用于指定数据库文件夹的路径。    
    databasefolder = "./"+str(model_name)+"/data/"+str(data_name)+"/"
    # 用于指定要处理的初始评分AmazonCSV文件的路径。    
    metafile = "./"+str(model_name)+"/data/"+str(data_name)+"/old/"    
    # 原始的交互记录
    ratings_csv = "./"+str(model_name)+"/data/"+str(data_name)+"/old/"+str(data_name)+".csv"
    # 存放处理后的交互记录
    output_ratings_csv = "./"+str(model_name)+"/data/"+str(data_name)+"/old/user_rate_item.csv"
    # 存放原始筛选后meta文件
    outputoldfolder = "./"+str(model_name)+"/data/"+str(data_name)+"/old/"
    # 存放处理后meta文件
    oldrefinefolder = "./"+str(model_name)+"/data/"+str(data_name)+"/refine/"




    Path(oldrefinefolder).mkdir(parents=True, exist_ok=True)
    Path(databasefolder +"data_list/").mkdir(parents=True, exist_ok=True)

    start_gen_data = time()
    # 1. 预处理之后的用户评分记录
    # 并且生成了user_rate_item.csv文件
    # process_initial_ratings(ratings_csv,output_ratings_csv)


    # 2. 按照时间划分用户记录，
    # 生成不同时间段的信息
    user_rate_item_df = pd.read_csv(output_ratings_csv, header=None, sep=',')
    time_count= int(max(user_rate_item_df[3])/300)+1

    # data_list = 0
    # start = 0
    # for index,row in user_rate_item_df.iterrows():
    #     if row[3] ==0:
    #         continue
    #     if row[3] % 300 ==0 and row[4] == 1:
    #         now = user_rate_item_df.head(index)
    #         Path(databasefolder +'data_list/'+str(data_list)).mkdir(parents=True, exist_ok=True)
    #         Path(databasefolder +'data_list/'+str(data_list)+'/refine').mkdir(parents=True, exist_ok=True)
    #         file_path = databasefolder +'data_list/'+str(data_list)+'/user_rate_item_df'+str(data_list)+'.csv'
    #         now.to_csv(file_path,header=None,index=None)
    #         # 当前数据集
    #         file_path = databasefolder +'data_list/'+str(data_list)+'/user_rate_item_df_little.csv'
    #         little = user_rate_item_df.iloc[start:index]
    #         little.to_csv(file_path,header=None,index=None)
    #         start = index
    #         data_list+=1
    #     if row[3] ==(time_count-1)*300:
    #         file_path = databasefolder +'data_list/'+str(data_list)+'/user_rate_item_df'+str(data_list)+'.csv'
    #         Path(databasefolder +'data_list/'+str(data_list)+'/refine').mkdir(parents=True, exist_ok=True)
    #         user_rate_item_df.to_csv(file_path, header=None,index=None)
    #         file_path = databasefolder +'data_list/'+str(data_list)+'/user_rate_item_df_little.csv'
    #         little = user_rate_item_df.iloc[start:]
    #         little.to_csv(file_path,header=None,index=None)

    # 3. 从json文件中获取到需要的属性
    # 并且生成了item_brand,item_category,item_item文件
    get_item_meta(metafile,outputoldfolder)




    # 4. 生成每个时间段节点连接情况和映射情况
    # 最终生成了图
    ic_filename = outputoldfolder + 'item_category.csv'
    ug_filename = outputoldfolder + 'user_gender.csv'
    # ua_filename = outputoldfolder + 'user_age.csv'
    # uo_filename = outputoldfolder + 'user_occupation.csv'

    item_category_df = pd.read_csv(ic_filename, header=None, sep=',')
    user_gender_df = pd.read_csv(ug_filename, header=None, sep=',')
    # user_age_df = pd.read_csv(ua_filename, header=None, sep=',')
    # user_occupation_df = pd.read_csv(uo_filename, header=None, sep=',')

    for index in range(time_count):
        file_path = "./"+str(model_name)+"/data/"+str(data_name)+"/data_list/"+str(index)+"/user_rate_item_df"+str(index)+".csv"
        refine_path = "./"+str(model_name)+"/data/"+str(data_name)+"/data_list/"+str(index)+"/refine/"
        little_path = "./"+str(model_name)+"/data/"+str(data_name)+"/data_list/"+str(index)+"/user_rate_item_df_little.csv"
        graph_path = "./"+str(model_name)+"/data/"+str(data_name)+"/data_list/"+str(index)+"/"  
        all_allnodes = form_ids(oldrefinefolder,file_path,refine_path,little_path,index,item_category_df,user_gender_df)
        gen_graph(all_allnodes,refine_path,graph_path)
        print("success:"+str(index))

  

    type_path = "./"+str(model_name)+"/data/"+str(data_name)+"/data_list/"+str(time_count-1)+"/refine/map.type2id"
    type2id = pickle.load(open(type_path, 'rb'))
    users_list = type2id['user']
    user_num = int(len(users_list))
    item_list = type2id['item']
    item_num = int(len(item_list))


    # 5. 生成每个时间段的用户行为记录
    for index in range(time_count):
        refine_path = "./"+str(model_name)+"/data/"+str(data_name)+"/data_list/"+str(index)+"/refine/"
        path = "./"+str(model_name)+"/data/"+str(data_name)+"/data_list/"+str(index)+"/path/"
        user_history_path = "./"+str(model_name)+"/data/"+str(data_name)+"/data_list/"+str(index)+"/path/user_history/"
        Path(databasefolder +'data_list/'+str(index)+'/path').mkdir(parents=True, exist_ok=True)
        Path(databasefolder +'data_list/'+str(index)+'/path/user_history').mkdir(parents=True, exist_ok=True)
        Path(databasefolder +'data_list/'+str(index)+'/negs').mkdir(parents=True, exist_ok=True)
        gen_ui_history(refine_path,path,user_history_path)
        gen_ui_history_little(refine_path,path,user_history_path)
        print("success:"+str(index))




    # 6. 训练集和测试集
    print("stare generate train and test")
    last_len= [0]*user_num
    user_additem= [0]*user_num
    user_all_len=[]
    testing = []
    all_user_history_file = "./"+str(model_name)+"/data/"+str(data_name)+"/data_list/"+str(time_count-1)+"/path/user_history/user_history.txt"
    
    with open(all_user_history_file, 'r') as all_f:
        for line in all_f:
            s = line.split()
            uid = int(s[0])
            item_history = [int(x) for x in s[1:]]
            user_all_len.append(int(len(item_history)*0.8))
            h = len(item_history)*0.8
            testing.append([uid]+item_history[int(h):])
    # 保存测试
    testing_file = "./"+str(model_name)+"/data/"+str(data_name)+"/data_list/"+str(time_count-1)+"/path/user_history/this_testing"
    pickle.dump(testing, open(testing_file,'wb'))

    # 测试集\
    testing = pickle.load(open(testing_file, 'rb'))
    test_negs = []
    for user_record in testing:
        uid = user_record[0]
        positive = [[uid, item, 1] for item in user_record[1:]]
        test_negs = test_negs + positive 
    test_negs_tf = pd.DataFrame(test_negs)
    testing_negs_file = "./"+str(model_name)+"/data/"+str(data_name)+"/data_list/"+str(time_count-1)+"/negs/testing.links"
    test_negs_tf.to_csv(testing_negs_file, header=None, index=None, sep=',')


    for index in range(time_count):
        refine_path = "./"+str(model_name)+"/data/"+str(data_name)+"/data_list/"+str(index)+"/refine/"
        user_history_path = "./"+str(model_name)+"/data/"+str(data_name)+"/data_list/"+str(index)+"/path/user_history/"
        negs_path = "./"+str(model_name)+"/data/"+str(data_name)+"/data_list/"+str(index)+"/negs/"
        
        # 6. split_train_test
        # 划分了数据集，分为训练集和测试集
        last_len,user_additem = split_train_test(user_num,user_history_path,user_all_len,last_len,user_additem)
        # 7. negative sample
        # 生成训练集和测试集的样本数据，其中包括正样本和负样本。
        neg_sample(user_history_path,negs_path)
        print("success:"+str(index))

    print(time()-start_gen_data)



    # 7. 生成交互系稀疏图

    for index in range(time_count):
        all_user_history_path = "./"+str(model_name)+"/data/"+str(data_name)+"/data_list/"+str(index)+"/path/user_history/data_user_history.txt"
        gen = "./"+str(model_name)+"/data/"+str(data_name)+"/data_list/"+str(index)+"/user_mean_item.npz"
        all_user_history = read_interaction_data(all_user_history_path)
        # 初始化行、列和数据列表
        rows = []
        cols = []
        data_values = []
        #填充行、列和数据列表
        for user_id, items in all_user_history:
            for item_id in items:
                rows.append(user_id)  # 用户ID转换为从0开始的索引
                cols.append(item_id - user_num)  # 项目ID转换为从0开始的索引
                data_values.append(1)  # 存在交互为1

        data_values = np.array(data_values, dtype=np.float64)

        interaction_matrix = sp.coo_matrix((data_values, (rows, cols)), shape=(user_num, item_num))
        mean_mat_list = _si_norm_lap(interaction_matrix)
        sparse.save_npz(gen, mean_mat_list)

        print("success:"+str(index))