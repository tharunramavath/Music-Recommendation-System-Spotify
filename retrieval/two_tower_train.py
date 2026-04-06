import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from retrieval.two_tower_model import UserTower, ItemTower, TwoTowerModel

class SpotifyDataset(Dataset):
    def __init__(self, interaction_csv, tracks_csv, genre_encoder_path, vectors_path):
        self.interactions = pd.read_csv(interaction_csv)
        self.tracks = pd.read_csv(tracks_csv, low_memory=False).set_index('id')
        
        with open(genre_encoder_path, 'rb') as f:
            self.genre_encoder = pickle.load(f)
            
        with open(vectors_path, 'rb') as f:
            self.track_vectors = pickle.load(f)

        # Precompute target genre IDs for tracks
        import ast
        def get_primary_genre_id(genres_str):
            if not isinstance(genres_str, str): return 0
            try:
                g_list = ast.literal_eval(genres_str)
                if not g_list: return 0
                return self.genre_encoder.get(g_list[0], 0)
            except:
                return 0
        
        # Only tracks mentioned in interactions to save memory
        int_tracks = self.interactions['track_id'].unique()
        self.track_data = {}
        for tid in int_tracks:
            if tid in self.tracks.index:
                row = self.tracks.loc[tid]
                self.track_data[tid] = {
                    "audio": self.track_vectors.get(tid, np.zeros(11)),
                    "genre": get_primary_genre_id(row['artist_genres'])
                }

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]
        user_id = row['user_id']
        track_id = row['track_id']
        label = 1.0 if row['interaction_type'] == 'like' else 0.0
        
        # Dummy context features (tod, device_enc)
        # tod: 0-23
        ctx = np.zeros(10)
        ctx[0] = row['time_of_day'] / 23.0
        # Simple device encoding 
        dev_map = {"mobile": 1, "desktop": 2, "car": 3}
        ctx[1] = dev_map.get(row['device'], 0) / 3.0
        
        # Fetch track data
        track_info = self.track_data.get(track_id, {"audio": np.zeros(11), "genre": 0})
        
        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "context": torch.tensor(ctx, dtype=torch.float32),
            "audio": torch.tensor(track_info['audio'], dtype=torch.float32),
            "genre": torch.tensor(track_info['genre'], dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float32)
        }

def train_model():
    print("Initializing Two-Tower Model training on simulated data...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset params
    intr_csv = "data/raw/simulated_interactions.csv"
    trks_csv = "data/processed/kaggle_tracks_cleaned.csv"
    genre_enc = "data/processed/genre_encoder.pkl"
    vecs_pkl = "data/processed/track_vectors.pkl"
    
    dataset = SpotifyDataset(intr_csv, trks_csv, genre_enc, vecs_pkl)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # Model configuration
    # Assuming 5000 users as per sim script and 4515 genres as per encoder script
    user_tower = UserTower(user_vocab_size=10000, context_dim=10) # 10k for safety
    item_tower = ItemTower(audio_feature_dim=11, genre_vocab_size=5000) 
    model = TwoTowerModel(user_tower, item_tower).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss() # Using dot-product mapped to 0-1 could need BCE
    # Actually, dot product of 2 normalized vectors is [-1, 1].
    # Standard Two-Tower often uses Contrastive Loss or BCE with a sigmoid.
    
    for epoch in range(5):
        epoch_loss = 0
        for batch in dataloader:
            u_ids = batch['user_id'].to(device)
            ctx = batch['context'].to(device)
            aud = batch['audio'].to(device)
            gen = batch['genre'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # Forward
            scores = model(u_ids, ctx, aud, gen)
            
            # Map [-1, 1] to [0, 1] for BCE
            probs = (scores + 1) / 2
            
            loss = criterion(probs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/5 | Average Loss: {epoch_loss/len(dataloader):.4f}")

    # Save checkpoints
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/two_tower_epoch5.pth")
    print("Model trained and saved to models/two_tower_epoch5.pth")

if __name__ == "__main__":
    train_model()
