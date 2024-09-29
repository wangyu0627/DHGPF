import torch
from torch import nn
import numpy as np
from utility.dataloader import Data
from model.sgat import GAT
import dgl
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
import dgl.function as fn
import scipy.sparse as sp
import os

def get_graph(graph,mu):
    # 找出数据小于阈值的边
    edges_to_keep = graph.data >= mu

    # 提取符合条件的数据点
    # new_data = graph.data[edges_to_keep]
    new_row = graph.row[edges_to_keep]
    new_col = graph.col[edges_to_keep]

    # 创建 DGL 图
    g = dgl.graph((new_row, new_col), num_nodes=graph.shape[0])
    # g.edata['weight'] = new_data
    return g

class GSLRec(nn.Module):
    def __init__(self, config, dataset: Data, uu_graph, ii_graph, mu_u, mu_i, device):
        super(GSLRec, self).__init__()
        self.config = config
        self.dataset = dataset
        self.in_size = config.in_size
        self.device = device
        self.init_weights()

        # create dgl graph
        self.uu_graph = get_graph(uu_graph, mu_u).to(device)
        self.ii_graph = get_graph(ii_graph, mu_i).to(device)
        # one HANLayer for user, one HANLayer for item
        self.user_gats = GAT(self.uu_graph, config.num_gat_layer, config.in_size, config.dim,
                                  config.out_size, [1, config.num_heads], F.elu, config.dropout,
                                  config.dropout, 0.2, 1e-6, False, 1)
        # self.user_gats = GATConv(config.in_size, config.out_size, config.num_gat_layer,
        #                          config.dropout, config.dropout, activation=F.elu, allow_zero_in_degree=True)
        self.item_gats = GAT(self.ii_graph, config.num_gat_layer, config.in_size, config.dim,
                                  config.out_size, [1, config.num_heads], F.elu, config.dropout,
                                  config.dropout, 0.2, 1e-6, False, 1)
        # self.item_gats = GATConv(config.in_size, config.out_size, config.num_gat_layer,
        #                          config.dropout, config.dropout, activation=F.elu, allow_zero_in_degree=True)

    def init_weights(self):
        self.initializer = nn.init.xavier_uniform_
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_users, embedding_dim=self.config.in_size)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_items, embedding_dim=self.config.in_size)
        self.initializer(self.user_embedding.weight, gain=1.414)
        self.initializer(self.item_embedding.weight, gain=1.414)

        self.Graph = self.dataset.sparse_adjacency_matrix()  # sparse matrix
        self.Graph = self.convert_sp_mat_to_sp_tensor(self.Graph)  # sparse tensor
        self.Graph = self.Graph.coalesce().to(self.device)  # Sort the edge index and remove redundancy
        self.activation = nn.Sigmoid()

    def convert_sp_mat_to_sp_tensor(self, sp_mat):
        """
            coo.row: x in user-item graph
            coo.col: y in user-item graph
            coo.data: [value(x,y)]
        """
        coo = sp_mat.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        value = torch.FloatTensor(coo.data)
        # from a sparse matrix to a sparse float tensor
        sp_tensor = torch.sparse.FloatTensor(index, value, torch.Size(coo.shape))
        return sp_tensor

    def aggregate(self):
        # [user + item, emb_dim]
        all_embedding = torch.cat([self.user_embedding.weight, self.item_embedding.weight])

        # no dropout
        embeddings = [all_embedding]

        for layer in range(self.config.GCNLayer):
            all_embedding = torch.sparse.mm(self.Graph, all_embedding)
            embeddings.append(all_embedding)

        final_embeddings = torch.stack(embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)

        users_emb, items_emb = torch.split(final_embeddings, [self.dataset.num_users, self.dataset.num_items])

        return users_emb, items_emb

    def bpr_loss(self, user, positive, negative):
        user_emb0, item_emb0 = self.aggregate()
        # metapath-based aggregation, h2
        user_emb1 = self.user_gats(self.user_embedding.weight)
        item_emb1 = self.item_gats(self.item_embedding.weight)

        # user_emb1 = self.user_gats(self.uu_graph, self.user_embedding.weight).flatten(1)
        # item_emb1 = self.item_gats(self.ii_graph, self.item_embedding.weight).flatten(1)
        # user_emb1 = F.relu(user_emb1)
        # item_emb1 = F.relu(item_emb1)

        all_user_embeddings = user_emb0 + self.config.k1*user_emb1
        all_item_embeddings = item_emb0 + self.config.k2*item_emb1
        # all_user_embeddings = user_emb0
        # all_item_embeddings = item_emb0

        user_embedding = all_user_embeddings[user.long()]
        positive_embedding = all_item_embeddings[positive.long()]
        negative_embedding = all_item_embeddings[negative.long()]

        ego_user_emb = self.user_embedding(user)
        ego_pos_emb = self.item_embedding(positive)
        ego_neg_emb = self.item_embedding(negative)

        reg_loss = (1 / 2) * (ego_user_emb.norm(2).pow(2) + ego_pos_emb.norm(2).pow(2) +
                              ego_neg_emb.norm(2).pow(2)) / float(len(user))

        pos_score = torch.sum(torch.mul(user_embedding, positive_embedding), dim=1)

        neg_score = torch.sum(torch.mul(user_embedding, negative_embedding), dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_score - pos_score))

        # reg_loss = reg_loss * self.config.l2
        # con_loss = self.user_gats.gat_layers[0].loss + self.user_gats.gat_layers[1].loss +\
        #            self.item_gats.gat_layers[0].loss + self.item_gats.gat_layers[1].loss
        return loss, reg_loss#, con_loss

    def getUsersRating(self, user):
        all_user_embeddings, all_item_embeddings = self.aggregate()

        user_embeddings = all_user_embeddings[user.long()]

        rating = self.activation(torch.matmul(user_embeddings, all_item_embeddings.t()))

        return rating