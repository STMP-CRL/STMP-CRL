'''
Created on Sep 4, 2023

'''
import torch
import numpy as np
import math
import heapq
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Helper(object):

    def __init__(self):
        self.timber = True

    def gen_group_member_dict(self, path):
        g_m_d = {}
        with open(path, 'r') as f:
            line = f.readline().strip()
            while line != None and line != "":
                a = line.split(' ')
                g = int(a[0])
                g_m_d[g] = []
                for m in a[1].split(','):
                    g_m_d[g].append(int(m))
                line = f.readline().strip()
        return g_m_d

    def gen_member_event_dict(self, path):

        f = open(path, 'r')
        m_e_d = {}
        for line in f.readlines():
            line = line.strip('\n')
            temp = line.split(' ')
            user_id = int(temp[0])
            item_id = int(temp[1])
            if user_id in m_e_d:
                m_e_d[user_id].append(item_id)
            else:
                m_e_d[user_id] = [item_id]

        return m_e_d

    def evaluate_model(self, model, testRatings, testNegatives, K, type_m):
        hr5_list, hr10_list = [], []
        ndcg5_list, ndcg10_list = [], []
        for idx in range(len(testRatings)):
            hr5, hr10, ndcg5, ndcg10 = self.eval_one_rating(model, testRatings, testNegatives, K, type_m, idx)
            hr5_list.append(hr5)
            hr10_list.append(hr10)
            ndcg5_list.append(ndcg5)
            ndcg10_list.append(ndcg10)

        HR5, HR10, NDCG5, NDCG10 = np.array(hr5_list).mean(), np.array(hr10_list).mean(), np.array(ndcg5_list).mean(), np.array(ndcg10_list).mean()
        return round(HR5, 5), round(HR10, 5), round(NDCG5, 5), round(NDCG10, 5)


    def eval_one_rating(self, model, testRatings, testNegatives, K, type_m, idx):
        rating = testRatings[idx]
        items = testNegatives[idx]
        u = rating[0]
        gtItem = rating[1]
        items.append(gtItem)

        map_item_score = {}
        users = np.full(len(items), u)

        users_var = torch.from_numpy(users)
        users_var = users_var.long().to(device)
        items_var = torch.LongTensor(items).to(device)
        if type_m == 'group':
            predictions = model('group', users_var, items_var)
        elif type_m == 'user':
            predictions, _Contrastive_loss = model('user', users_var, items_var)
        for i in range(len(items)):
            item = items[i]
            map_item_score[item] = predictions.data.cpu().numpy()[i]
        items.pop()

        ranklist = heapq.nlargest(5, map_item_score, key=map_item_score.get)
        hr5 = self.getHitRatio(ranklist, gtItem)
        ndcg5 = self.getNDCG(ranklist, gtItem)

        ranklist = heapq.nlargest(10, map_item_score, key=map_item_score.get)
        hr10 = self.getHitRatio(ranklist, gtItem)
        ndcg10 = self.getNDCG(ranklist, gtItem)
        return hr5, hr10, ndcg5, ndcg10

    def getHitRatio(self, ranklist, gtItem):
        for item in ranklist:
            if item == gtItem:
                return 1
        return 0

    def getNDCG(self, ranklist, gtItem):
        for i in range(len(ranklist)):
            item = ranklist[i]
            if item == gtItem:
                return math.log(2) / math.log(i+2)
        return 0