import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LightGCN
from torch_geometric.utils import degree

class SpotifyGNN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super(SpotifyGNN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # User and Item embeddings (Initialized)
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        
        # Initialize embeddings with normal distribution
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        
        self.num_layers = num_layers

    def forward(self, edge_index):
        # We need to create a unified embedding Matrix for message passing
        # Concatenate [Users, Items]
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0) # (U+I, dim)
        
        # Simplified LightGCN message passing (Adjacency normalization)
        # Create bipartite graph adjacency (row = users, col = items)
        u, i = edge_index
        # Offset item indices
        i = i + self.num_users
        
        # Build symmetric adjacency matrix for GCN
        adj_row = torch.cat([u, i], dim=0)
        adj_col = torch.cat([i, u], dim=0)
        full_edge_index = torch.stack([adj_row, adj_col], dim=0)
        
        # Degree normalization (1 / sqrt(di * dj))
        row, col = full_edge_index
        deg = degree(row, all_emb.size(0), dtype=all_emb.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Message passing layers
        embs_list = [all_emb]
        for _ in range(self.num_layers):
            all_emb = self.gcn_step(all_emb, full_edge_index, norm)
            embs_list.append(all_emb)
            
        # LightGCN final embedding: average of all layers
        final_emb = torch.stack(embs_list, dim=0).mean(dim=0)
        
        user_final = final_emb[:self.num_users]
        item_final = final_emb[self.num_users:]
        
        return user_final, item_final

    def gcn_step(self, x, edge_index, norm):
        # Message passing: x_i = sum (norm * x_j for j in neighbors(i))
        row, col = edge_index
        out = torch.zeros_like(x)
        msg = x[col] * norm.view(-1, 1)
        out.index_add_(0, row, msg)
        return out

if __name__ == "__main__":
    # Test model
    model = SpotifyGNN(num_users=100, num_items=1000)
    # Dummy edges: user 0 likes item 1
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    u_vecs, i_vecs = model(edge_index)
    print(f"User embeddings shape: {u_vecs.shape}")
    print(f"Item embeddings shape: {i_vecs.shape}")
