import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import random
import os
from retrieval.gnn_model import SpotifyGNN

def bpr_loss(users_emb, pos_items_emb, neg_items_emb):
    # Bayesian Personalized Ranking loss
    pos_scores = torch.mul(users_emb, pos_items_emb).sum(dim=1)
    neg_scores = torch.mul(users_emb, neg_items_emb).sum(dim=1)
    
    loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
    return loss

def train_gnn():
    print("Initializing GNN training (LightGCN) on co-listening graph...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load graph data
    with open("data/processed/gnn_data.pkl", 'rb') as f:
        data = pickle.load(f)
        
    edge_index = data['edge_index'].to(device)
    num_users = data['num_users']
    num_items = data['num_items']
    
    model = SpotifyGNN(num_users=num_users, num_items=num_items).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train loops
    epochs = 10
    batch_size = 1024
    num_edges = edge_index.shape[1]
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        # Sampling positive and negative items for BPR
        # Randomly shuffle positive edges
        perm = torch.randperm(num_edges)
        u_pos = edge_index[0, perm]
        i_pos = edge_index[1, perm]
        
        for i in range(0, num_edges, batch_size):
            u_batch = u_pos[i:i+batch_size]
            i_batch = i_pos[i:i+batch_size]
            # Sample random negatives (one per positive)
            neg_batch = torch.randint(0, num_items, (len(u_batch),)).to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: Aggregate graph-level embeddings
            users_final, items_final = model(edge_index)
            
            u_emb = users_final[u_batch]
            i_emb = items_final[i_batch]
            n_emb = items_final[neg_batch]
            
            loss = bpr_loss(u_emb, i_emb, n_emb)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/max(1, (num_edges//batch_size)):.4f}")

    # Save GNN model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/gnn_lightgcn.pth")
    print("GNN Model trained and saved.")

if __name__ == "__main__":
    train_gnn()
