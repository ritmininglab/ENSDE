import torch
import torch.nn as nn

class emb_network(nn.Module):
    def __init__(self,n_users,m_items,latent_dim,eps=1e-6):
        super(emb_network, self).__init__()
        self.num_users = n_users
        self.num_items = m_items
        self.latent_dim = latent_dim
        self.mse = nn.MSELoss()
        self.eps = eps
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

    def forward(self, users,items):
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)

        return  users_emb,items_emb

    def item(self, items):
        return self.embedding_item(items)

    def user(self, users):
        return self.embedding_user(users)