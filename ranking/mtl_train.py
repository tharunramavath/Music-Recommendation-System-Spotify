import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from ranking.mtl_model import MultiTaskRanker
from retrieval.two_tower_model import UserTower # To generate user embeddings

class RankingDataset(Dataset):
    def __init__(self, interaction_csv, item_vec_path, user_tower_path, user_count=10000):
        self.df = pd.read_csv(interaction_csv)
        
        # Load item vectors (dim=64)
        with open(item_vec_path, 'rb') as f:
            self.item_vectors = pickle.load(f)
            
        # Initialize UserTower 
        self.user_tower = UserTower(user_vocab_size=user_count)
        state_dict = torch.load(user_tower_path, map_location='cpu')
        # Filter for user_tower
        u_state = {k.replace('user_tower.', ''): v for k, v in state_dict.items() if k.startswith('user_tower.')}
        self.user_tower.load_state_dict(u_state)
        self.user_tower.eval()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        user_id = row['user_id']
        track_id = row['track_id']
        
        # Labels
        y_like = 1.0 if row['interaction_type'] == 'like' else 0.0
        y_skip = 1.0 if row['interaction_type'] == 'skip' else 0.0
        
        # Context (norm)
        ctx = np.zeros(10)
        ctx[0] = row['time_of_day'] / 23.0
        dev_map = {"mobile": 1, "desktop": 2, "car": 3}
        ctx[1] = dev_map.get(row['device'], 0) / 3.0
        
        # User dynamic embedding (forward pass once if heavy, but for training small batches it's fine)
        with torch.no_grad():
            u_id_tensor = torch.tensor([user_id], dtype=torch.long)
            ctx_tensor = torch.tensor([ctx], dtype=torch.float32)
            u_emb = self.user_tower(u_id_tensor, ctx_tensor).squeeze(0).numpy()
            
        # Item embedding (precomputed 64-dim)
        i_emb = self.item_vectors.get(track_id, np.zeros(64))
        
        # Concatenated feature vector: [User, Item, Context]
        # Dim: 64 + 64 + 10 = 138
        x = np.concatenate([u_emb, i_emb, ctx])
        
        return {
            "features": torch.tensor(x, dtype=torch.float32),
            "label_like": torch.tensor(y_like, dtype=torch.float32),
            "label_skip": torch.tensor(y_skip, dtype=torch.float32)
        }

def train_ranker():
    print("Training Multi-Task Ranking Model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset
    ds = RankingDataset(
        interaction_csv="data/raw/simulated_interactions.csv",
        item_vec_path="data/processed/two_tower_item_vectors.pkl",
        user_tower_path="models/two_tower_epoch5.pth"
    )
    # Smaller batch for faster forward passes of user tower
    dl = DataLoader(ds, batch_size=512, shuffle=True)
    
    # Model
    model = MultiTaskRanker(input_dim=138).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    for epoch in range(5):
        epoch_like_loss = 0
        epoch_skip_loss = 0
        
        for batch in dl:
            x = batch['features'].to(device)
            y_like = batch['label_like'].to(device).unsqueeze(1)
            y_skip = batch['label_skip'].to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            p_like, p_skip = model(x)
            
            loss_like = criterion(p_like, y_like)
            loss_skip = criterion(p_skip, y_skip)
            
            # Weighted task sum (Tune weights as needed)
            total_loss = loss_like + (0.5 * loss_skip) 
            
            total_loss.backward()
            optimizer.step()
            
            epoch_like_loss += loss_like.item()
            epoch_skip_loss += loss_skip.item()
            
        print(f"Epoch {epoch+1}/5 | Like Loss: {epoch_like_loss/len(dl):.4f} | Skip Loss: {epoch_skip_loss/len(dl):.4f}")

    # Save Ranker
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/ranking_mtl.pth")
    print("Multi-Task Ranker trained and saved.")

if __name__ == "__main__":
    train_ranker()
