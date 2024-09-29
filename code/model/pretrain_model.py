import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class GraphConv(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 node_dropout,
                 bias=False):

        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = out_dim
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.act = nn.PReLU()
        self.dropout = node_dropout

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_dim))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        # for m in self.modules():
        #     self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.414)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj):
        seq = self.fc(seq)
        out = torch.spmm(adj, seq)
        if self.bias is not None:
            out += self.bias
        out = F.relu(out)
        return out

# 定义一个GCN模型
class GCNlayer(nn.Module):
    def __init__(self, meta_path_patterns, in_features, out_features, sqrt_user, sqrt_item, device):
        super(GCNlayer, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(len(meta_path_patterns)):
            self.layers.append(GraphConv(in_features, out_features, 0.1))
        self.meta_path_patterns = list(tuple(meta_path_pattern) for meta_path_pattern in meta_path_patterns)
        self.sqrt_user = sqrt_user
        self.sqrt_item = sqrt_item

    def forward(self, adj, x):
        embeds = []
        for i,layer in enumerate(self.layers):
            x = layer(x, adj[i])
            embeds.append(x)
        embeds = torch.stack(embeds)
        return embeds

def get_sqrt(sparse_tensors,num,device):
    # 初始化一个空的稀疏张量
    sparse_result = torch.sparse_coo_tensor(
        size=(num, num),
        device=device
    )

    # 遍历列表中的稀疏张量，相加到 sparse_result 中
    for tensor in sparse_tensors:
        sparse_result = sparse_result + tensor

    return sparse_result

class Pretrain(nn.Module):
    def __init__(self, meta_path_patterns, user_key, item_key, user_mps, item_mps, in_size, out_size, num_layer, device):
        super(Pretrain, self).__init__()
        self.initializer = nn.init.xavier_uniform_
        self.userkey = user_key
        self.itemkey = item_key
        self.device = device
        self.num_layer = num_layer
        self.user_mps = [mp.to(device) for mp in user_mps]
        self.item_mps = [mp.to(device) for mp in item_mps]
        self.num_users = user_mps[0].shape[0]
        self.num_items = item_mps[0].shape[0]
        self.sqrt_user = get_sqrt(self.user_mps,self.num_users,device)
        self.sqrt_item = get_sqrt(self.item_mps,self.num_items,device)
        self.meta_path_patterns = meta_path_patterns
        self.user_mps_2 = [mp * self.sqrt_user for mp in self.user_mps]
        self.item_mps_2 = [mp * self.sqrt_item for mp in self.item_mps]
        # no pretrain
        # xavier uniform is a better choice than normal for training model
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=in_size)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=in_size)
        self.initializer(self.user_embedding.weight, gain=1.414)
        self.initializer(self.item_embedding.weight, gain=1.414)
        self.weight_b1 = torch.nn.Parameter(torch.FloatTensor(len(meta_path_patterns[user_key]),1,1), requires_grad=True)
        self.weight_b2 = torch.nn.Parameter(torch.FloatTensor(len(meta_path_patterns[item_key]),1,1), requires_grad=True)
        torch.nn.init.uniform_(self.weight_b1, a=0, b=1)
        torch.nn.init.uniform_(self.weight_b2, a=0, b=1)

        # one HANLayer for user, one HANLayer for item
        self.gcns = nn.ModuleDict({
            key: GCNlayer(value, in_size, out_size, self.sqrt_user, self.sqrt_item, device) for key, value in
            self.meta_path_patterns.items()
        })

        self.user_layer1 = nn.Linear(out_size, out_size, bias=True)
        self.item_layer1 = nn.Linear(out_size, out_size, bias=True)

        self.layernorm = nn.LayerNorm(out_size)


    def forward(self, user_idx, item_idx, neg_item_idx):
        user_key = self.userkey
        item_key = self.itemkey

        # metapath-based aggregation
        h1 = {}
        h2 = {}
        for key in self.meta_path_patterns.keys():
            if key == user_key:
                for i in range(self.num_layer):
                    if i == 0:
                        h1[key] = self.gcns[key](self.user_mps, self.user_embedding.weight)
                        h1[key] = (self.weight_b1 * h1[key]).sum(0)
                    else:
                        h2[key] = self.gcns[key](self.user_mps_2, h1[key])
                        h2[key] = (self.weight_b1 * h2[key]).sum(0)
                # h1[key] = h1[key] + h2[key]
            else:
                for i in range(self.num_layer):
                    if i == 0:
                        h1[key] = self.gcns[key](self.item_mps, self.item_embedding.weight)
                        h1[key] = (self.weight_b2 * h1[key]).sum(0)
                    else:
                        h2[key] = self.gcns[key](self.item_mps_2, h1[key])
                        h2[key] = (self.weight_b2 * h2[key]).sum(0)
                # h1[key] = h1[key] + h2[key]

            # update node embeddings
        user_emb = h1[user_key]
        item_emb = h1[item_key]

        # user_emb = self.user_layer1(user_emb)
        # item_emb = self.item_layer1(item_emb)
        # layer norm
        # user_emb = self.layernorm(user_emb)
        # item_emb = self.layernorm(item_emb)
        # Relu
        user_emb = F.relu(user_emb)
        item_emb = F.relu(item_emb)


        user_feat = user_emb[user_idx]
        item_feat = item_emb[item_idx]
        neg_item_feat = item_emb[neg_item_idx]

        return user_feat, item_feat, neg_item_feat

    def bpr_loss(self, users, pos, neg):
        users_emb, pos_emb, neg_emb = self.forward(users, pos, neg)

        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))

        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def getUsersRating(self, user_idx):
        item_idx = torch.Tensor(np.arange(self.num_items)).long().to(self.device)
        users_emb, all_items, _ = self.forward(user_idx, item_idx, None)
        rating = torch.matmul(users_emb, all_items.t())
        return rating
