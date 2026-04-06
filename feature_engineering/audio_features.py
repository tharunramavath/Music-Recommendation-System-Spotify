import pandas as pd
import numpy as np
import os
import pickle

def create_audio_vectors(input_csv="data/processed/kaggle_tracks_cleaned.csv"):
    if not os.path.exists(input_csv):
        # Fallback to seed_tracks.csv if bootstrap was run but kaggle dataset isn't present
        input_csv = "data/raw/seed_tracks.csv"
        if not os.path.exists(input_csv):
            print(f"Error: {input_csv} not found. Please run ingest_kaggle.py or spotipy_bootstrap.py first.")
            return

    print(f"Processing audio features from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Select audio features to vectorise
    features = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 
        'speechiness', 'acousticness', 'instrumentalness', 
        'liveness', 'valence', 'tempo'
    ]
    
    # Check if features exist (some may be normalised in earlier step)
    feature_cols = []
    for f in features:
        if f in df.columns:
            feature_cols.append(f)
        elif f"{f}_norm" in df.columns:
            feature_cols.append(f"{f}_norm")
            
    # Final vector composition
    X = df[feature_cols].values
    
    # Standardize if not normalization was done in ingest_kaggle
    # Basic Z-score to bring it to a similar scale
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Map track_id (id column in CSV) to vector
    track_to_vector = {row['id']: X_scaled[i] for i, row in df.iterrows()}
    
    # Save the mapping and the scaler
    output_vec_path = "data/processed/track_vectors.pkl"
    output_scaler_path = "data/processed/scaler.pkl"
    
    os.makedirs(os.path.dirname(output_vec_path), exist_ok=True)
    
    with open(output_vec_path, 'wb') as f:
        pickle.dump(track_to_vector, f)
    
    with open(output_scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
        
    print(f"Created {len(track_to_vector)} vectors (dim={len(feature_cols)}). Saved to {output_vec_path}")

if __name__ == "__main__":
    create_audio_vectors()
