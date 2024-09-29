import torch
from torch import nn
import numpy as np
from utility.dataloader import Data


class LightGCN(nn.Module):
    def __init__(self, config, dataset: Data, device):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self.device = device
        self.init_weights()

    def init_weights(self):
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_users, embedding_dim=self.config.dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_items, embedding_dim=self.config.dim)

        # no pretrain
        # xavier uniform is a better choice than normal for training model
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)

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

    def get_bpr_loss(self, user, positive, negative):
        all_user_embeddings, all_item_embeddings = self.aggregate()

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

        reg_loss = reg_loss * self.config.l2

        return loss, reg_loss

    def getUsersRating(self, user):
        all_user_embeddings, all_item_embeddings = self.aggregate()

        user_embeddings = all_user_embeddings[user.long()]

        rating = self.activation(torch.matmul(user_embeddings, all_item_embeddings.t()))

        return rating