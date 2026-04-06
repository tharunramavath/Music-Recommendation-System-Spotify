import torch
import pandas as pd
import pickle
import numpy as np
import os
from retrieval.two_tower_model import ItemTower
import ast

def export_item_embeddings(model_path="models/two_tower_epoch5.pth", output_path="data/processed/two_tower_item_vectors.pkl"):
    print(f"Exporting item embeddings using {model_path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load metadata and encoder
    tracks_df = pd.read_csv("data/processed/kaggle_tracks_cleaned.csv", low_memory=False)
    with open("data/processed/genre_encoder.pkl", 'rb') as f:
        genre_encoder = pickle.load(f)
    with open("data/processed/track_vectors.pkl", 'rb') as f:
        audio_vectors = pickle.load(f)

    # Initialize ItemTower
    item_tower = ItemTower(audio_feature_dim=11, genre_vocab_size=5000)
    state_dict = torch.load(model_path, map_location=device)
    
    # Extract weights for item tower
    # In TwoTowerModel, they are under item_tower.
    item_tower_state = {k.replace('item_tower.', ''): v for k, v in state_dict.items() if k.startswith('item_tower.')}
    item_tower.load_state_dict(item_tower_state)
    item_tower.to(device)
    item_tower.eval()

    def parse_genres(x):
        if not isinstance(x, str): return 0
        try:
            l = ast.literal_eval(x)
            return genre_encoder.get(l[0], 0) if l else 0
        except: return 0

    print("Feeding items through ItemTower (Batching)...")
    track_ids = tracks_df['id'].tolist()
    batch_size = 1000
    all_embeddings = {}

    with torch.no_grad():
        for i in range(0, len(track_ids), batch_size):
            batch_ids = track_ids[i:i+batch_size]
            batch_df = tracks_df.iloc[i:i+batch_size]
            
            # Prepare tensors
            aud_batch = torch.tensor([audio_vectors.get(tid, np.zeros(11)) for tid in batch_ids], dtype=torch.float32).to(device)
            gen_batch = torch.tensor([parse_genres(g) for g in batch_df['artist_genres']], dtype=torch.long).to(device)
            
            # Forward
            embeddings = item_tower(aud_batch, gen_batch)
            embeddings_np = embeddings.cpu().numpy()
            
            for j, tid in enumerate(batch_ids):
                all_embeddings[tid] = embeddings_np[j]
            
            if (i // batch_size) % 50 == 0:
                print(f"Processed {i} / {len(track_ids)} tracks...")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(all_embeddings, f)
        
    print(f"Successfully exported {len(all_embeddings)} item embeddings (dim=64) to {output_path}")

if __name__ == "__main__":
    export_item_embeddings()
