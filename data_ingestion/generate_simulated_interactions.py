import pandas as pd
import numpy as np
import random
import os
import ast

def generate_simulated_data(n_users=5000, n_interactions=50000, output_path="data/raw/simulated_interactions.csv"):
    """
    Faster version: Groups tracks by broad genres once and samples within those groups.
    """
    print(f"Generating optimized simulated interaction data for {n_users} users...")
    
    # Load tracks
    tracks_df = pd.read_csv("data/processed/kaggle_tracks_cleaned.csv", low_memory=False)
    
    # Pre-calculate genre groups
    print("Pre-bucketing tracks by broad genres for faster sampling...")
    def parse_list(x):
        if not isinstance(x, str): return []
        try: return ast.literal_eval(x)
        except: return []

    # Map each track to a simplified genre label
    # E.g., anything with 'rock' or 'pop' in it
    pop_track_ids = tracks_df[tracks_df['artist_genres'].str.contains('pop|rock|dance', na=False, case=False)]['id'].tolist()
    jazz_track_ids = tracks_df[tracks_df['artist_genres'].str.contains('jazz|classical|swing', na=False, case=False)]['id'].tolist()
    all_track_ids = tracks_df['id'].tolist()

    interactions = []
    
    for _ in range(n_interactions):
        user_id = random.randint(0, n_users-1)
        # 50/50 affinity for pop or jazz
        is_pop_user = (user_id % 2 == 0)
        
        # 80% pick from relevant group, 20% random
        if random.random() < 0.8:
            track_id = random.choice(pop_track_ids if is_pop_user else jazz_track_ids)
            outcome = 1 if random.random() > 0.2 else 0 # 80% like
        else:
            track_id = random.choice(all_track_ids)
            outcome = 1 if random.random() > 0.8 else 0 # 20% like
            
        interactions.append({
            "user_id": user_id,
            "track_id": track_id,
            "interaction_type": "like" if outcome == 1 else "skip",
            "timestamp": pd.Timestamp.now() - pd.Timedelta(seconds=random.randint(0, 86400*30)),
            "device": random.choice(["mobile", "desktop", "car"]),
            "time_of_day": random.randint(0, 23)
        })
        
    interactions_df = pd.DataFrame(interactions)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    interactions_df.to_csv(output_path, index=False)
    print(f"Generated {len(interactions_df)} interactions. Saved to {output_path}")

if __name__ == "__main__":
    generate_simulated_data()
