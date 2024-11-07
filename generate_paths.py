import sys
sys.path.append('../../')

import numpy as np
import time
import pickle
from pathlib import Path
import pandas as pd
from utils.parser import parse_args




class IIPath:
    def __init__(self, **kargs):
        self.metapath_list = kargs.get('metapath_list')
        # self.metapath_list = ['uibi', 'uibici', 'uici', 'uicibi', 'ibibi', 'ibici', 'ibiui', 'icibi', 'icici', 'iciui', 'iuiui']
        self.outputfolder = kargs.get('outputfolder')
        # 同理完成映射
        self.ui_dict = dict()
        self.iu_dict = dict()
        self.ic_dict = dict()
        self.ci_dict = dict()
        self.ib_dict = dict()
        self.bi_dict = dict()
        self.ii_dict = dict()
        self.ii2_dict = dict()
        # 包含了用户访问项目历史的，各个项目对，都挨着
        self.all_ii_direct = set()

        # 设置了总体的嵌入向量矩阵

        self.embeddings = kargs.get('embeddings')
        print('############################')
        print(self.embeddings.shape)

        print('Begin to load data')
        start = time.time()

        self.load_ui(kargs.get('ui_relation_file'))
        self.load_ic(kargs.get('ic_relation_file'))
        self.load_ib(kargs.get('ib_relation_file'))
        self.load_ii(kargs.get('ii_relation_file'),kargs.get('user_history_file'))
        # 加载用户历史文件，并且把项目对加入到集合里面
        self.load_all_ii_direct(kargs.get('user_history_file'))

        end = time.time()
        print('Load data finished, used time %.2fs' % (end - start))
        self.path_list = list()

        self.metapath_based_randomwalk()


    def load_all_ii_direct(self, user_history_f):
        # 只读方式打开文件
        with open(user_history_f, 'r') as f:  # 2200
            # 遍历每一行
            for line in f:
                # 获取每一行
                s = line.split()
                # 获取用户的项目历史
                item_history = [int(x) for x in s[1:]]
                item_num = len(item_history)
                for index in range(item_num-1):
                    # i1，i2为两个挨着的项目
                    i1 = item_history[index]
                    i2 = item_history[index+1]
                    if i1 not in self.ii_dict.keys():
                        self.ii_dict[i1] = [i2]
                    else:
                        self.ii_dict[i1].append(i2)

                    # 把项目对加入到集合里面
                    self.all_ii_direct.add((i1,i2))
        print(f'Here are {len(self.all_ii_direct)} item-item paths in data.')

    def load_ib(self, ibfile):
        item_brand_data = open(ibfile, 'r').readlines()
        for item_brand_ele in item_brand_data:
            item_brand_ele_list = item_brand_ele.strip().split(',')
            item = int(item_brand_ele_list[0])
            brand = int(item_brand_ele_list[1])
            if item not in self.ib_dict.keys():
                self.ib_dict[item] = [brand]
            else:
                self.ib_dict[item].append(brand)

            if brand not in self.bi_dict.keys():
                self.bi_dict[brand] = [item]
            else:
                self.bi_dict[brand].append(item)

    def load_ic(self, icfile):
        item_category_data = open(icfile, 'r').readlines()
        for item_category_ele in item_category_data:
            item_category_ele_list = item_category_ele.strip().split(',')
            item = int(item_category_ele_list[0])
            category = int(item_category_ele_list[1])
            if item not in self.ic_dict.keys():
                self.ic_dict[item] = [category]
            else:
                self.ic_dict[item].append(category)

            if category not in self.ci_dict.keys():
                self.ci_dict[category] = [item]
            else:
                self.ci_dict[category].append(item)

    def load_ui(self, uifile):
        user_item_data = open(uifile, 'r').readlines()
        for user_item_ele in user_item_data:
            user_item_ele_list = user_item_ele.strip().split(',')
            user = int(user_item_ele_list[0])
            item = int(user_item_ele_list[1])
            if item not in self.iu_dict.keys():
                self.iu_dict[item] = [user]
            else:
                self.iu_dict[item].append(user)

            if user not in self.ui_dict.keys():
                self.ui_dict[user] = [item]
            else:
                self.ui_dict[user].append(item)
    def load_ii(self, iifile,user_history_file):
        item_item_data = open(iifile, 'r').readlines()
        for item_item_ele in item_item_data:
            item_item_ele_list = item_item_ele.strip().split(',')
            item = int(item_item_ele_list[0])
            item2 = int(item_item_ele_list[1])
            if item not in self.ii_dict.keys():
                self.ii_dict[item] = [item2]
            else:
                self.ii_dict[item].append(item2)

            if item2 not in self.ii2_dict.keys():
                self.ii2_dict[item2] = [item]
            else:
                self.ii2_dict[item2].append(item)

    def ifInIIpairs(self, startItem, endItem):
        if (startItem, endItem) in self.all_ii_direct:
            return True
        else:
            return False

    def save_icibi(self, all_item_ids, start_time, outfile):
        limit = 1
        with open(outfile, 'w') as file:

            for i in all_item_ids:
                count = 0
                try:
                    c_list = self.ic_dict[i]
                    c_list = self.get_top_k(c_list, i, 20)

                except KeyError:
                    continue

                for c in c_list:
                    if(count>=limit):
                        continue
                    try:
                        i2_list = self.ci_dict[c][:] 
                        if i in i2_list: i2_list.remove(i)

                    except KeyError:
                        continue
                    for i2 in i2_list:
                        if(count>=limit):
                            continue
                        try:
                            b_list = self.ib_dict[i2]

                        except KeyError:
                            continue
                        for b in b_list:
                            if(count>=limit):
                                continue
                            try:

                                i3_list = self.bi_dict[b][:] 

                                if i in i3_list: i3_list.remove(i)
                                if i2 in i3_list: i3_list.remove(i2)

                            except KeyError:
                                continue
                            for i3 in i3_list:
                                if(count>=limit):
                                    continue

                                if self.ifInIIpairs(i, i3):
                                    
                                    
                                    path = str(i) + ' ' + str(c) + ' ' + str(i2) + ' ' + str(b) + ' ' + str(i3)
                                    path_id = str(i) + ',' + str(i3)
                                    write_content = path_id + '\t' + path + '\n'
                                    outfile = self.outputfolder+'icibi.paths'
                                    file.write(write_content)
                                    count+=1
                if(count>=limit):
                    continue

                if i%100 ==0:
                    this_time = time.time() - start_time
                    print(f'processed item: {i}, time: {this_time}')

    def save_ibici(self, all_item_ids, start_time, outfile):
        limit = 1
        with open(outfile, 'w') as file:
            for i in all_item_ids:
                count = 0
                try:
                    b_list = self.ib_dict[i]
                except KeyError:
                    continue
                # b_list = self.get_top_k(b_list, i, limit)
                for b in b_list:
                    if(count>=limit):
                        continue
                    try:
                        i2_list = self.bi_dict[b]
                    except KeyError:
                        continue
                    i2_list = self.get_top_k(i2_list, b, 20)
                    for i2 in i2_list:
                        if(count>=limit):
                            continue
                        try:
                            c_list = self.ic_dict[i2]
                        except KeyError:
                            continue
                        for c in c_list:
                            if(count>=limit):
                                continue
                            try:
                                i3_list = self.ci_dict[c]
                            except KeyError:
                                continue
                            for i3 in i3_list:
                                if(count>=limit):
                                    continue
                                if self.ifInIIpairs(i, i3):
                                    path = str(i) + ' ' + str(b) + ' ' + str(i2) + ' ' + str(c) + ' ' + str(i3)
                                    path_id = str(i) + ',' + str(i3)
                                    write_content = path_id + '\t' + path + '\n'
                                    outfile = self.outputfolder+'ibici.paths'

                                    file.write(write_content)
                                    count+=1
            
                if i%100 ==0:
                    this_time = time.time() - start_time
                    print(f'processed item: {i}, time: {this_time}')    


    def get_sim(self, u, v):
        #  计算余弦相似度
        return u.dot(v) / ((u.dot(u) ** 0.5) * (v.dot(v) ** 0.5))

    def get_top_k(self, c_list, i, limit):
        i_c1_sim_list = []
        for c1 in c_list:
            # cal sim
            sim = self.get_sim(self.embeddings[c1], self.embeddings[i])
            i_c1_sim_list.append([c1, sim])
        i_c1_sim_list.sort(key=lambda x: x[1], reverse=True)
        i_c1_sim_list = i_c1_sim_list[:min(limit, len(i_c1_sim_list))]
        c_list = [c1 for c1, sim in i_c1_sim_list]
        return c_list

    def save_icici(self, all_item_ids, start_time, outfile):
        limit = 1
        outfile = self.outputfolder+'icici.paths'
        start_time = time.time()
        with open(outfile, 'w') as file:
            for i in all_item_ids:
                count = 0
                try:
                    c_list = self.ic_dict[i]
                except KeyError:
                    continue
                c_list = self.get_top_k(c_list, i, 20)
                for c1 in c_list:
                    if(count>=limit):
                        continue
                    try:
                        i_list2 = self.ci_dict[c1]
                    except KeyError:
                        continue
                    i_list2 = self.get_top_k(i_list2, c1, 20)
                    for i2 in i_list2:
                        if(count>=limit):
                            continue
                        try:
                            c_list2 = self.ic_dict[i2]
                        except KeyError:
                            continue
                        c_list2 = self.get_top_k(c_list2, i2, 20)
                        for c2 in c_list2:
                            if(count>=limit):
                                continue
                            try:
                                i_list3 = self.ci_dict[c2]
                            except KeyError:
                                continue
                            for i3 in i_list3:
                                if(count>=limit):
                                    continue
                                if self.ifInIIpairs(i, i3):
                                    path = str(i) + ' ' + str(c1) + ' ' + str(i2) + ' ' + str(c2) + ' ' + str(i3)
                                    path_id = str(i) + ',' + str(i3)
                                    write_content = path_id + '\t' + path + '\n'

                                    

                                    file.write(write_content)

                                    count+=1

                if i%100 ==0:
                    this_time = time.time() - start_time
                    print(f'processed item: {i}, time: {this_time}')





    def save_ici(self, all_item_ids, start_time, outfile):
        limit = 1
        outfile = self.outputfolder+'ici.paths'
        with open(outfile, 'w') as file:
            for i in all_item_ids:
                count = 0
                try:
                    c_list = self.ic_dict[i]
                except KeyError:
                    continue
                # c -> i
                for c1 in c_list:
                    if(count>=limit):
                        continue
                    try:
                        i2_list = self.ci_dict[c1][:] 
                        if i in i2_list: i2_list.remove(i)
                    except KeyError:
                        continue
                    for i2 in i2_list:
                        if(count>=limit):
                            continue
                        if self.ifInIIpairs(i, i2):
                            path = str(i) + ' ' + str(c1) + ' ' + str(i2)
                            path_id = str(i) + ',' + str(i2)
                            write_content = path_id + '\t' + path + '\n'
                            

                            file.write(write_content)
                            count+=1
                if i%100 ==0:
                    this_time = time.time() - start_time
                    print(f'processed item: {i}, time: {this_time}')
    def save_ibi(self, all_item_ids, start_time, outfile):
        limit = 1
        outfile = self.outputfolder+'ibi.paths'
        with open(outfile, 'w') as file:
            for i in all_item_ids:
                
                count = 0
                try:
                    b_list = self.ib_dict[i]

                except KeyError:
                    continue
                # b -> i
                for b1 in b_list:


                    if(count>=limit):
                        continue
                    try:
                        i2_list = self.bi_dict[b1][:] 
                        if i in i2_list: i2_list.remove(i)
                    except KeyError:
                        continue
                    for i2 in i2_list:

                        if(count>=limit):
                            continue
                        if self.ifInIIpairs(i, i2):
                            path = str(i) + ' ' + str(b1) + ' ' + str(i2)
                            path_id = str(i) + ',' + str(i2)
                            write_content = path_id + '\t' + path + '\n'
                            
                            file.write(write_content)
                            count+=1
                if i%100 ==0:
                    this_time = time.time() - start_time
                    print(f'processed item: {i}, time: {this_time}')
    def save_iii(self, all_item_ids, start_time, outfile):


        limit = 1
        outfile = self.outputfolder+'iii.paths'
        with open(outfile, 'w') as file:

            for i in all_item_ids:

                count = 0
                try:
                    i_list = self.ii_dict[i]

                except KeyError:
                    continue
                # c -> i
                for i1 in i_list:
                    if(count>=limit):
                        continue
                    try:
                        i2_list = self.ii2_dict[i1][:] 
                        if i in i2_list: i2_list.remove(i)
                    except KeyError:
                        continue
                    
                    for i2 in i2_list:
                        if(count>=limit):
                            continue
                        if self.ifInIIpairs(i, i2):

                            path = str(i) + ' ' + str(i1) + ' ' + str(i2)
                            path_id = str(i) + ',' + str(i2)
                            write_content = path_id + '\t' + path + '\n'
                            file.write(write_content)
                            count+=1
                if i%100 ==0:
                    this_time = time.time() - start_time

                    print(f'processed item: {i}, time: {this_time}')

    def save_iui(self, all_item_ids, start_time, outfile):


        limit = 1
        outfile = self.outputfolder+'iui.paths'
        with open(outfile, 'w') as file:

            for i in all_item_ids:

                count = 0
                try:
                    u_list = self.iu_dict[i]

                except KeyError:
                    continue
                # c -> i
                for u1 in u_list:
                    if(count>=limit):
                        continue
                    try:
                        i2_list = self.ui_dict[u1][:] 
                        if i in i2_list: i2_list.remove(i)
                    except KeyError:
                        continue
                    
                    for i2 in i2_list:
                        if(count>=limit):
                            continue
                        if self.ifInIIpairs(i, i2):

                            path = str(i) + ' ' + str(u1) + ' ' + str(i2)
                            path_id = str(i) + ',' + str(i2)
                            write_content = path_id + '\t' + path + '\n'
                            file.write(write_content)
                            count+=1
                if i%100 ==0:
                    this_time = time.time() - start_time

                    print(f'processed item: {i}, time: {this_time}')



    def save_ibibi(self, all_item_ids, start_time, outfile):
        limit = 1
        outfile = self.outputfolder+'ibibi.paths'
        with open(outfile, 'w') as file:
            for i in all_item_ids:
                count = 0
                try:
                    b_list = self.ib_dict[i]
                except KeyError:
                    continue
                # b -> i
                for b1 in b_list:
                    if(count>=limit):
                        continue
                    try:
                        i2_list = self.bi_dict[b1][:] 
                        if i in i2_list: i2_list.remove(i)

                    except KeyError:
                        continue
                    # i -> b
                    for i2 in i2_list:
                        if(count>=limit):
                            continue
                        try:
                            b_list2 = self.ib_dict[i2][:] 
                            # if b1 in b_list2: b_list2.remove(b1)

                        except KeyError:
                            continue
                        # b -> i
                        for b2 in b_list2:
                            if(count>=limit):
                                continue
                            try:
                                i3_list = self.bi_dict[b2][:] 
                                if i in i3_list: i3_list.remove(i)
                                if i2 in i3_list: i3_list.remove(i2)
                            except KeyError:
                                continue
                            for i3 in i3_list:
                                if(count>=limit):
                                    continue
                                if self.ifInIIpairs(i, i3):
                                    
                                    path = str(i) + ' ' + str(b1) + ' ' + str(i2) + ' ' + str(b2) + ' ' + str(i3)
                                    path_id = str(i) + ',' + str(i3)
                                    write_content = path_id + '\t' + path + '\n'

                                    

                                    file.write(write_content)
                                    count+=1
                if i%100 ==0:
                    this_time = time.time() - start_time
                    print(f'processed item: {i}, time: {this_time}')

    def save_iuiui(self, all_item_ids, start_time, outfile):
        limit = 1
        outfile = self.outputfolder+'iuiui.paths'
        with open(outfile, 'w') as file:
        
            for i in all_item_ids:
                count = 0
                try:
                    u_list1 = self.iu_dict[i]
                except KeyError:
                    continue

                for u1 in u_list1:
                    if(count>=limit):
                        continue
                    try:
                        i2_list = self.ui_dict[u1]
                    except KeyError:
                        continue
                    for i2 in i2_list:
                        if(count>=limit):
                            continue
                        try:
                            u_list2 = self.iu_dict[i2]
                        except KeyError:
                            continue
                        for u2 in u_list2:
                            if(count>=limit):
                                continue
                            try:
                                i3_list = self.ui_dict[u2]
                            except KeyError:
                                continue
                            for i3 in i3_list:
                                if(count>=limit):
                                    continue
                                if self.ifInIIpairs(i, i3):
                                    path = str(i) + ' ' + str(u1) + ' ' + str(i2) + ' ' + str(u2) + ' ' + str(i3)
                                    path_id = str(i) + ',' + str(i3)
                                    write_content = path_id + '\t' + path + '\n'
                                    

                                    file.write(write_content)
                                    count+=1

                if i%100 ==0:
                    this_time = time.time() - start_time
                    print(f'processed item: {i}, time: {this_time}')


    def save_iciui(self, all_item_ids, start_time, outfile):
        limit = 1
        outfile = self.outputfolder+'iciui.paths'
        with open(outfile, 'w') as file:
            
            for i in all_item_ids:
                count = 0
                try:
                    c_list = self.ic_dict[i]
                except KeyError:
                    continue

                for c1 in c_list:
                    if(count>=limit):
                        continue
                    try:
                        i2_list = self.ci_dict[c1]
                    except KeyError:
                        continue
                    for i2 in i2_list:
                        if(count>=limit):
                            continue
                        try:
                            u_list = self.iu_dict[i2]
                        except KeyError:
                            continue
                        for u in u_list:
                            if(count>=limit):
                                continue
                            try:
                                i3_list = self.ui_dict[u]
                            except KeyError:
                                continue
                            for i3 in i3_list:
                                if(count>=limit):
                                    continue
                                if self.ifInIIpairs(i, i3):
                                    path = str(i) + ' ' + str(c1) + ' ' + str(i2) + ' ' + str(u) + ' ' + str(i3)
                                    path_id = str(i) + ',' + str(i3)
                                    write_content = path_id + '\t' + path + '\n'
                                    

                                    file.write(write_content)
                                    count+=1
                if i%100 ==0:
                    this_time = time.time() - start_time
                    print(f'processed item: {i}, time: {this_time}')

    def save_ibiui(self, all_item_ids, start_time, outfile):
        limit = 1
        outfile = self.outputfolder+'ibiui.paths'
        with open(outfile, 'w') as file:
            for i in all_item_ids:
                count = 0
                try:
                    b_list = self.ib_dict[i]
                except KeyError:
                    continue

                for b in b_list:
                    if(count>=limit):
                        continue
                    try:
                        i2_list = self.bi_dict[b]
                    except KeyError:
                        continue
                    for i2 in i2_list:
                        if(count>=limit):
                            continue
                        try:
                            u_list = self.iu_dict[i2]
                        except KeyError:
                            continue
                        for u in u_list:
                            if(count>=limit):
                                continue
                            try:
                                i3_list = self.ui_dict[u]
                            except KeyError:
                                continue
                            for i3 in i3_list:
                                if(count>=limit):
                                    continue
                                if self.ifInIIpairs(i, i3):
                                    path = str(i) + ' ' + str(b) + ' ' + str(i2) + ' ' + str(u) + ' ' + str(i3)
                                    path_id = str(i) + ',' + str(i3)
                                    write_content = path_id + '\t' + path + '\n'
                                    

                                    file.write(write_content)
                                    count+=1

                if i%100 ==0:
                    this_time = time.time() - start_time
                    print(f'processed item: {i}, time: {this_time}')



    def metapath_based_randomwalk(self):
        # 获取到了所有的项目id
        all_item_ids = list(self.iu_dict.keys())
        all_item_ids.sort()
        # print(all_item_ids)
        # lsj
        start_time = time.time()
        

        for metapath in self.metapath_list:
            outfile = self.outputfolder + metapath + '.paths'
            print(f'outfile name = {self.outputfolder}{metapath}.paths')
            if metapath == 'ici':
                self.save_ici(all_item_ids, start_time, outfile)
            if metapath == 'ibi':
                self.save_ibi(all_item_ids, start_time, outfile)
            if metapath == 'iii':
                self.save_iii(all_item_ids, start_time, outfile)
            if metapath == 'iui':
                self.save_iui(all_item_ids, start_time, outfile)


            if metapath == 'icibi':
                self.save_icibi(all_item_ids, start_time, outfile)
            if metapath == 'ibici':
                self.save_ibici(all_item_ids, start_time, outfile)
            if metapath == 'icici':
                self.save_icici(all_item_ids, start_time, outfile)
            if metapath == 'ibibi':
                self.save_ibibi(all_item_ids, start_time, outfile)
            if metapath == 'iuiui':
                self.save_iuiui(all_item_ids, start_time, outfile)
            if metapath == 'iciui':
                self.save_iciui(all_item_ids, start_time, outfile)
            if metapath == 'ibiui':
                self.save_ibiui(all_item_ids, start_time, outfile)



def gen_instances(embeddings,refine_path,user_history_path,out_path):
    ic_relation_file = refine_path + 'item_category.relation'
    ib_relation_file = refine_path + 'item_brand.relation'
    ui_relation_file = refine_path + 'user_item.relation'
    ii_relation_file = refine_path + 'item_item.relation'
    # ii_metapahts_list = ['ici','ibi','iii','iui']
    ii_metapahts_list = ['icici','icibi','iciui', 'ibiui','ibici','ibibi','ici','ibi','iii','iui'
                           'iuiui']
    IIPath(ib_relation_file=ib_relation_file, ic_relation_file=ic_relation_file,ui_relation_file=ui_relation_file,ii_relation_file = ii_relation_file,
            user_history_file=user_history_path,
            metapath_list=ii_metapahts_list,
           outputfolder=out_path,embeddings = embeddings)
   
def count_nodes(time_count):
    type_path = "./"+str(model_name)+"/data/"+str(data_name)+"/data_list/"+str(time_count-1)+"/refine/map.type2id"
    type2id = pickle.load(open(type_path, 'rb'))
    users_list = type2id['user']
    user_num = len(users_list)
    item_list = type2id['item']
    item_num = len(item_list)
    category_list = type2id['category']
    category_num = len(category_list)
    brand_list = type2id['brand']
    brand_num = len(brand_list)
    all_nodes = user_num + item_num + category_num + brand_num
    tensor_shape = (int(all_nodes), 100)
    embeddings = np.random.rand(*tensor_shape)

    return embeddings
if __name__ == '__main__':
    args = parse_args()
    model_name = args.model_name
    data_name = args.dataset
    # 获取总的时间段数和总节点数
    user_rate_item_df = pd.read_csv("./"+str(model_name)+"/data/"+str(data_name)+"/old/user_rate_item.csv", header=None, sep=',')
    time_count= int(max(user_rate_item_df[3])/100)+1
    embeddings= count_nodes(time_count)

    for index in range(time_count):
        refine_path = "./"+str(model_name)+"/data/"+str(data_name)+"/data_list copy/"+str(index)+"/refine/"
        user_history_path = "./"+str(model_name)+"/data/"+str(data_name)+"/data_list copy/"+str(index)+"/path/user_history/data_user_history.txt"
        type_path = "./"+str(model_name)+"/data/"+str(data_name)+"/data_list copy/"+str(index)+"/refine/map.type2name"
        out_path = "./"+str(model_name)+"/data/"+str(data_name)+"/data_list copy/"+str(index)+"/path/all_ii_paths/"
        Path("./"+str(model_name)+"/data/"+str(data_name)+"/data_list copy/"+str(index)+"/path/all_ii_paths/").mkdir(parents=True, exist_ok=True)
        # 从文件加载数据
        
        gen_instances(embeddings,refine_path,user_history_path,out_path)

