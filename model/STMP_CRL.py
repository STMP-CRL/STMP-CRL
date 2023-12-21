import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class STMP_CRL(nn.Module):
    def __init__(self, num_users, num_items, num_groups, embedding_dim, input_dim, hidden_size, num_layers, g_m_d,  member_event_dict, graph, GCN_layers, is_split, drop_ratio,temperature,lambda1):
        super(STMP_CRL, self).__init__()
        self.userembeds = nn.Embedding(num_users, embedding_dim)
        self.itemembeds = nn.Embedding(num_items, embedding_dim)
        self.groupembeds = nn.Embedding(num_groups, embedding_dim)
        self.eventembeds = EventEmebddingLayer(num_items, embedding_dim)
        self.attention = AttentionLayer(2 * embedding_dim, drop_ratio)
        self.predictlayer = PredictLayer(embedding_dim, drop_ratio)

        self.group_member_dict = g_m_d
        self.num_users = num_users
        self.num_items = num_items

        # LightGCN Convolution
        self.light_gcn = LightGCN(num_users, num_items, GCN_layers, graph)

        # RNN GRU
        self.member_event_dict = member_event_dict
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru_rnn = GRULayer(input_dim, hidden_size, num_layers)
        self.dropout = drop_ratio
        self.temperature = temperature
        self.lambda1 = lambda1

        # initial model
        nn.init.xavier_uniform_(self.userembeds.weight)
        nn.init.xavier_uniform_(self.itemembeds.weight)
        nn.init.xavier_uniform_(self.groupembeds.weight)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
    def RNN_GRU(self, user_input_id):
        user_finnal_list = []
        user_input_id = user_input_id.tolist()
        for k in user_input_id:
            c_state = torch.zeros(self.num_layers, 1, self.hidden_size).to(device)
            events = self.member_event_dict[k]
            event_embedding = self.eventembeds(Variable(torch.LongTensor(events).to(device))).unsqueeze(1).float()
            event_embedding = event_embedding.to(device)
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            prediction, c_state = self.gru_rnn(event_embedding, c_state)
            b_state = c_state.squeeze(1)
            user_finnal_list.append(b_state.view(-1))
        user_finnal_vec = torch.stack(user_finnal_list, dim=0)
        return user_finnal_vec

    def forward(self, type, inputs, item_inputs):
        # train group
        if (type == "group" ):
            out = self.grp_forward(inputs, item_inputs)
            return out
        # train user
        elif(type == "user"):
            out,contrastive_loss = self.usr_forward(inputs, item_inputs)
            return out,contrastive_loss

    # group forward
    def grp_forward(self, group_inputs, item_inputs):
        group_embeds = torch.Tensor().to(device)
        item_embeds_group = self.itemembeds(item_inputs)
        for i, j in zip(group_inputs, item_inputs):
            members = self.group_member_dict[i.item()]
            user_inputs = torch.LongTensor(members).to(device)
            user_embeds_pure = self.userembeds(user_inputs)

            items_numb = []
            for _ in members:
                items_numb.append(j)
            item_inputs_user = torch.LongTensor(items_numb).to(device)
            item_embeds_pure = self.itemembeds(item_inputs_user)
            user_embeds = user_embeds_pure
            item_embeds = item_embeds_pure
            group_item_embeds = torch.cat((user_embeds, item_embeds), dim=1)
            at_wt = self.attention(group_item_embeds)
            g_embeds_with_attention = torch.matmul(at_wt,user_embeds)
            group_embeds_pure = self.groupembeds(torch.LongTensor([i]).to(device))
            g_embeds = g_embeds_with_attention + group_embeds_pure
            group_embeds = torch.cat((group_embeds, g_embeds))
        y = torch.sigmoid(self.predictlayer(group_embeds * item_embeds_group))  # # Tensor[256,1]
        return y

    # user forward
    def usr_forward(self, user_inputs, item_inputs):
        light_gcn_user_emb, light_gcn_item_emb = self.light_gcn(self.userembeds.weight,
                                                                self.itemembeds.weight)
        gcn_user_emb = light_gcn_user_emb[user_inputs]
        gcn_item_emb = light_gcn_item_emb[item_inputs]
        gru_user_emb = self.RNN_GRU(user_inputs)

        user_embeds_pure = self.userembeds(user_inputs)
        item_embeds_pure = self.itemembeds(item_inputs)
        con_loss1 = self._create_distance_correlation(gcn_user_emb,gru_user_emb)
        Contrastive_loss = con_loss1

        item_embeds = gcn_item_emb + item_embeds_pure

        gcn = gcn_user_emb * item_embeds
        gru = gru_user_emb * item_embeds
        pure = user_embeds_pure * item_embeds

        p1 = self.predictlayer(gcn)
        p2 = self.predictlayer(gru)
        p3 = self.predictlayer(pure)

        p = (p1 + p2 + p3) / 3.0
        y = torch.sigmoid(p)
        return y, Contrastive_loss

    def Contrastive_learning1(self,gcn_user_emb,gru_user_emb):
        gru_user_embed_nor = F.normalize(gru_user_emb,p=2,dim=1)
        gcn_user_embed_nor = F.normalize(gcn_user_emb,p=2,dim=1)
        sim_matrix = torch.mm(gru_user_embed_nor, gcn_user_embed_nor.T)
        positive = torch.diag(sim_matrix).unsqueeze(1)
        nominator = torch.exp(positive / self.temperature)
        denominator = torch.sum(torch.exp(sim_matrix / self.temperature), axis=1, keepdim=True)
        Contrastive_loss = self.lambda1 * (torch.sum(-torch.log(nominator / denominator)) / gcn_user_emb.shape[0])

        return Contrastive_loss

    def Contrastive_learning2(self,gcn_user_emb,gru_user_emb):
        gru_user_embed_nor = F.normalize(gru_user_emb, p=2, dim=1)
        gcn_user_embed_nor = F.normalize(gcn_user_emb, p=2, dim=1)
        sim_matrix = torch.mm(gru_user_embed_nor, gcn_user_embed_nor.T)
        positive = torch.diag(sim_matrix).unsqueeze(1)
        loss_matrix = -torch.log(torch.sigmoid(positive - sim_matrix))
        Contrastive_loss = self.lambda1*((torch.sum(loss_matrix) - torch.sum(torch.diag(loss_matrix))) / gcn_user_emb.shape[0])
        return Contrastive_loss

    def Contrastive_learning3(self,gcn_user_emb,gru_user_emb):
        gru_user_embed_nor = F.normalize(gru_user_emb, p=2, dim=1)
        gcn_user_embed_nor = F.normalize(gcn_user_emb, p=2, dim=1)
        sim_matrix = torch.mm(gru_user_embed_nor, gcn_user_embed_nor.T)
        loss_matrix = -torch.log(torch.sigmoid(1 - sim_matrix))
        Contrastive_loss = self.lambda1*((torch.sum(loss_matrix)) / gcn_user_emb.shape[0])
        return Contrastive_loss

    def _create_distance_correlation(self, X1, X2):
        def _create_centered_distance(X):
            r = torch.sum(X * X, dim=1, keepdim=True)
            value = r - 2 * torch.mm(X, X.T + r.T)
            zero_value = torch.zeros_like(value)
            value = torch.where(value > 0.0, value, zero_value)
            D = torch.sqrt(value + 1e-8)
            D = (
                D
                - torch.mean(D, dim=0, keepdim=True)
                - torch.mean(D, dim=1, keepdim=True)
                + torch.mean(D)
            )
            return D

        def _create_distance_covariance(D1, D2):
            n_samples = float(D1.size(0))
            value = torch.sum(D1 * D2) / (n_samples * n_samples)
            zero_value = torch.zeros_like(value)
            value = torch.where(value > 0.0, value, zero_value)
            dcov = torch.sqrt(value + 1e-8)
            return dcov

        D1 = _create_centered_distance(X1)
        D2 = _create_centered_distance(X2)

        dcov_12 = _create_distance_covariance(D1, D2)
        dcov_11 = _create_distance_covariance(D1, D1)
        dcov_22 = _create_distance_covariance(D2, D2)

        value = dcov_11 * dcov_22
        zero_value = torch.zeros_like(value)
        value = torch.where(value > 0.0, value, zero_value)
        dcor = dcov_12 / (torch.sqrt(value) + 1e-10)
        return dcor

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, layers, g):
        super(LightGCN, self).__init__()
        self.num_users, self.num_items = num_users, num_items
        self.layers = layers
        self.graph = g

    def compute(self, users_emb, items_emb):
        all_emb = torch.cat([users_emb, items_emb])
        embeddings = [all_emb]
        for _ in range(self.layers):
            all_emb = torch.sparse.mm(self.graph, all_emb)
            embeddings.append(all_emb)
        embeddings = torch.mean(torch.stack(embeddings, dim=1), dim=1)
        users, items = torch.split(embeddings, [self.num_users, self.num_items])
        return users, items
    def forward(self, groups_emb, items_emb):
        return self.compute(groups_emb, items_emb)

class GRULayer(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers):
        super(GRULayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x, h):
        r_out, h_state = self.gru(x, h)
        outs = []
        for record in range(r_out.size(1)):
            outs.append(self.out(r_out[:, record, :]))
        return torch.stack(outs, dim=1), h_state

class EventEmebddingLayer(nn.Module):
    def __init__(self, num_events, embedding_dim):
        super(EventEmebddingLayer, self).__init__()
        self.eventEmbedding = nn.Embedding(num_events, embedding_dim)

    def forward(self, event_inputs):
        event_embeds = self.eventEmbedding(event_inputs)
        return event_embeds

class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(AttentionLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        out = self.linear(x)
        weight = torch.softmax(out.view(1, -1), dim=1)
        return weight


class PredictLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 8),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        out = self.linear(x)
        return out

