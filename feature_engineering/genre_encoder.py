import pandas as pd
import pickle
import os
import ast

def build_genre_vocab(tracks_csv="data/processed/kaggle_tracks_cleaned.csv", output_path="data/processed/genre_encoder.pkl"):
    print("Building optimized genre vocabulary...")
    df = pd.read_csv(tracks_csv, usecols=['artist_genres'], low_memory=False)
    
    # Optimization: Find unique genre strings first
    unique_genre_strings = df['artist_genres'].unique()
    print(f"Number of unique genre combinations: {len(unique_genre_strings)}")
    
    def parse_list(x):
        if not isinstance(x, str): return []
        try: return ast.literal_eval(x)
        except: return []

    all_genres = set()
    for s in unique_genre_strings:
        for g in parse_list(s):
            all_genres.add(g)
            
    # Sort for deterministic mapping
    sorted_genres = sorted(list(all_genres))
    genre_to_id = {g: i+1 for i, g in enumerate(sorted_genres)} # 0 for padding/unknown
    genre_to_id['<UNK>'] = 0
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(genre_to_id, f)
        
    print(f"Built vocabulary with {len(genre_to_id)} genres. Saved to {output_path}")
    return genre_to_id

if __name__ == "__main__":
    build_genre_vocab()
