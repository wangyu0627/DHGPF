import torch
import torch.nn as nn

class BPR(nn.Module):
    def __init__(self, n_user, n_item, device, args):
        super(BPR, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.device = device
        self.emb_size = 128
        self.batch_size = args.batch_size

        self.user_embedding = torch.nn.Embedding(self.n_user, self.emb_size)
        self.item_embedding = torch.nn.Embedding(self.n_item, self.emb_size)
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)

    def getUsersRating(self, users):
        users_emb = self.user_embedding.weight[users]
        items_emb = self.item_embedding.weight
        rating = torch.matmul(users_emb, items_emb.t())
        return rating

    def bpr_loss(self, users, pos, neg):
        users_emb = self.user_embedding.weight[users]
        pos_emb = self.item_embedding.weight[pos]
        neg_emb = self.item_embedding.weight[neg]

        # reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
        #                       pos_emb.norm(2).pow(2) +
        #                       neg_emb.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss#, reg_loss