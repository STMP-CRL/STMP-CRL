'''
Created on Sep 4, 2023
Main function
'''
from datetime import datetime

from model.STMP_CRL import STMP_CRL
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from time import time
from config import Config
from utils.util import Helper
from dataset import GDataset
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state = {}
def training(model, train_loader, epoch_id, config, type_m):
    model = model.to(device)
    learning_rates = config.lr
    lr = learning_rates[0]
    if epoch_id >= 5 and epoch_id < 15:
        lr = learning_rates[1]
    elif epoch_id >= 15:
        lr = learning_rates[2]
    if epoch_id % 5 == 0:
        lr /= 2
    st_time = time()
    optimizer = optim.RMSprop(model.parameters(), lr)

    losses = []
    print('%s train_loader length: %d' % (type_m, len(train_loader)))
    for batch_id, (u, pi_ni) in tqdm(enumerate(train_loader)):
        # Data Load
        input = u.to(device) #
        pos_item_input = pi_ni[:, 0].to(device) # (2345,2345)
        neg_item_input = pi_ni[:, 1].to(device) # ()
        # Forward
        if type_m == 'user':
            pos_prediction, contrast_loss_pos = model("user", input, pos_item_input)
            neg_prediction, _contrast_loss_neg = model("user", input, neg_item_input)
        elif type_m == 'group':
            pos_prediction = model("group", input, pos_item_input)
            neg_prediction = model("group", input, neg_item_input)
        # Zero_grad
        model.zero_grad()
        # BPR Loss
        if type_m == 'user':
            loss = torch.mean(torch.nn.functional.softplus(neg_prediction - pos_prediction)) + contrast_loss_pos
        elif type_m == 'group':
            loss = torch.mean(torch.nn.functional.softplus(neg_prediction - pos_prediction))
        # record loss history
        losses.append(loss)
        # Backward
        loss.backward()
        optimizer.step()

    print(
        f'Epoch {epoch+1}, {type_m} loss: {torch.mean(torch.stack(losses)):.5f}, Cost time: {time() - st_time:4.2f}s')

    return f'Epoch {epoch+1}, {type_m} loss: {torch.mean(torch.stack(losses)):.5f}, Cost time: {time() - st_time:4.2f}s'

def evaluation(model, helper, testRatings, testNegatives, K, type_m):
    model.eval()
    HR5, HR10, NDCG5, NDCG10 = helper.evaluate_model(model, testRatings, testNegatives, K, type_m)
    return HR5, HR10, NDCG5, NDCG10

if __name__ == '__main__':

    config = Config()
    helper = Helper()
    g_m_d = helper.gen_group_member_dict(config.user_in_group_path)
    dataset = GDataset(config.user_dataset, config.group_dataset, config.num_negatives)
    user_item_serialization = helper.gen_member_event_dict(config.user_dataset + "Train.txt")
    num_group = len(g_m_d)
    num_users, num_items, graph = dataset.num_users, dataset.num_items, dataset.graph

    # build AGREE model
    n_layers = config.gcn_layers
    is_split = config.is_split

    agree = STMP_CRL(num_users, num_items, num_group, config.embedding_size, config.input_dim, config.hidden_size,
                  config.num_layers, g_m_d, user_item_serialization,
                  graph, n_layers, is_split, config.drop_ratio,config.temperature,config.lambda1)
    agree.to(device)
    maxEpoch = 0
    max_userHr5, max_userHr10, max_userNDCG5, max_userNDCG10 = 0, 0, 0, 0
    max_groupHr5, max_groupHr10, max_groupNDCG5, max_groupNDCG10 = 0, 0, 0, 0
    f = open(config.save_path, "w+")
    for epoch in range(config.epoch):
        agree.train()
        t1 = time()
        loss_user = training(agree, dataset.get_user_dataloader(config.batch_size), epoch, config, 'user')

        loss_group = training(agree, dataset.get_group_dataloader(config.batch_size), epoch, config, 'group')
        print("user and group training time is: [%.1f s]" % (time()-t1))
        t3 = time()
        user_HR5, user_HR10, user_NDCG5, user_NDCG10 = evaluation(agree, helper, dataset.user_testRatings,
                                                                  dataset.user_testNegatives, config.topK, 'user')
        t_user = time() - t3
        print(
            f"[Epoch {epoch+1}] User, Hit@{config.topK}: {user_HR5, user_HR10}, NDCG@{config.topK}: {user_NDCG5, user_NDCG10}, time:[{round(t_user,1)}]")

        t2 = time()
        group_HR5, group_HR10, group_NDCG5, group_NDCG10 = evaluation(agree, helper, dataset.group_testRatings,
                                                                      dataset.group_testNegatives, config.topK,
                                                                      'group')
        t_group = time() - t2
        print(
            f"[Epoch {epoch+1}] Group, Hit@{config.topK}: {group_HR5, group_HR10}, NDCG@{config.topK}: {group_NDCG5, group_NDCG10}, time:[{round(t_group)}]")

        user_tag = 0
        group_tag = 0
        if max_userHr5 < user_HR5:
            user_tag = user_tag + 1
        else:
            user_tag = user_tag - 1
        if max_userHr10 < user_HR10:
            user_tag = user_tag + 1
        else:
            user_tag = user_tag - 1
        if max_userNDCG5 < user_NDCG5:
            user_tag = user_tag + 1
        else:
            user_tag = user_tag - 1
        if max_userNDCG10 < user_NDCG10:
            user_tag = user_tag + 1
        else:
            user_tag = user_tag - 1

        if max_groupHr5 < group_HR5:
            group_tag = group_tag + 1
        else:
            group_tag = group_tag - 1
        if max_groupHr10 < group_HR10:
            group_tag = group_tag + 1
        else:
            group_tag = group_tag - 1
        if max_groupNDCG5 < group_NDCG5:
            group_tag = group_tag + 1
        else:
            group_tag = group_tag - 1
        if max_groupNDCG10 < group_NDCG10:
            group_tag = group_tag + 1
        else:
            group_tag = group_tag - 1

        tag = user_tag + group_tag
        if tag > 0:
            maxEpoch = epoch
            max_userHr5 = user_HR5
            max_userHr10 = user_HR10
            max_userNDCG5 = user_NDCG5
            max_userNDCG10 = user_NDCG10
            max_groupHr5 = group_HR5
            max_groupHr10 = group_HR10
            max_groupNDCG5 = group_NDCG5
            max_groupNDCG10 = group_NDCG10
        elif tag == 0:
            if group_tag >= user_tag:
                maxEpoch = epoch
                max_userHr5 = user_HR5
                max_userHr10 = user_HR10
                max_userNDCG5 = user_NDCG5
                max_userNDCG10 = user_NDCG10
                max_groupHr5 = group_HR5
                max_groupHr10 = group_HR10
                max_groupNDCG5 = group_NDCG5
                max_groupNDCG10 = group_NDCG10
        if (epoch + 1) % 5 == 0:
            print('= ' * 20)
            print(
                f"[MaxEpoch {maxEpoch+1}] user, Hit@{config.topK}: {max_userHr5, max_userHr10}, NDCG@{config.topK}: {max_userNDCG5, max_userNDCG10}")
            print(
                f"[MaxEpoch {maxEpoch+1}] Group, Hit@{config.topK}: {max_groupHr5, max_groupHr10}, NDCG@{config.topK}: {max_groupNDCG5, max_groupNDCG10}")
            f.write(
                f"[MaxEpoch {maxEpoch+1}] user, Hit@{config.topK}: {max_userHr5, max_userHr10}, NDCG@{config.topK}: {max_userNDCG5, max_userNDCG10}")
            f.write(
                f"[MaxEpoch {maxEpoch+1}] Group, Hit@{config.topK}: {max_groupHr5, max_groupHr10}, NDCG@{config.topK}: {max_groupNDCG5, max_groupNDCG10}")
            f.write("\n")
    f.close()
    print('## Finishing Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print('= ' * 20)
    print("Done!")









