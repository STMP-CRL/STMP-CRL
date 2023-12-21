'''

Modified  on Sep 21, 2023, by
'''

import scipy.sparse as sp
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class GDataset(object):

    def __init__(self, user_path, group_path, num_negatives):
        self.num_negatives = num_negatives

        self.user_trainMatrix = self.load_rating_file_as_matrix(user_path + "Train.txt")

        self.user_testRatings = self.load_rating_file_as_list(user_path + "Test.txt")

        self.user_testNegatives = self.load_negative_file(user_path + "Negative.txt")

        self.num_users, self.num_items = self.user_trainMatrix.shape

        self.group_trainMatrix = self.load_rating_file_as_matrix(group_path + "Train.txt")

        self.group_testRatings = self.load_rating_file_as_list(group_path + "Test.txt")

        self.group_testNegatives = self.load_negative_file(group_path + "Negative.txt")

        self.graph = self.get_graph()
    def get_graph(self):

        adj_mat = sp.dok_matrix((self.num_users + self.num_items, self.num_users + self.num_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.user_trainMatrix.tolil()
        adj_mat[:self.num_users, self.num_users:] = R
        adj_mat[self.num_users:, :self.num_users] = R.T
        adj_mat = adj_mat.todok()

        rowsum = np.array(adj_mat.sum(axis=1))

        d_inv = np.power(rowsum, -0.5).flatten()

        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        Graph = Graph.coalesce().to(device)

        return Graph

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                negatives = []
                for x in arr[1:]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_rating_file_as_matrix(self, filename):
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                if len(arr) > 2:
                    user, item, rating = int(arr[0]), int(arr[1]), int(arr[2])
                    if (rating > 0):
                        mat[user, item] = 1.0
                else:
                    user, item = int(arr[0]), int(arr[1])
                    mat[user, item] = 1.0
                line = f.readline()
        return mat


    def get_train_instances(self, train):
        users, pos_items, neg_items = [], [], []

        num_users, num_items = train.shape[0], train.shape[1]

        for (u, i) in train.keys():
            for _ in range(self.num_negatives):
                users.append(u)
                pos_items.append(i)

                j = np.random.randint(num_items)
                while (u, j) in train:
                    j = np.random.randint(num_items)
                neg_items.append(j)
        pos_neg_items = [[pos_item, neg_item] for pos_item, neg_item in zip(pos_items, neg_items)]
        return users, pos_neg_items

    def get_user_dataloader(self, batch_size):
        users, pos_neg_items = self.get_train_instances(self.user_trainMatrix)
        train_data = TensorDataset(torch.LongTensor(users), torch.LongTensor(pos_neg_items))
        return DataLoader(train_data, batch_size=batch_size, shuffle=True)

    def get_group_dataloader(self, batch_size):
        groups, pos_neg_items = self.get_train_instances(self.group_trainMatrix)
        train_data = TensorDataset(torch.LongTensor(groups), torch.LongTensor(pos_neg_items))
        return DataLoader(train_data, batch_size=batch_size, shuffle=True)





