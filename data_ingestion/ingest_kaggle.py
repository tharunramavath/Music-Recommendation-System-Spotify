import pandas as pd
import os
import ast

def ingest_kaggle(tracks_path="data/raw/tracks.csv", artists_path="data/raw/artists.csv"):
    if not os.path.exists(tracks_path):
        print(f"Dataset not found at {tracks_path}. Please place it there.")
        return
    
    print(f"Ingesting Kaggle tracks dataset from {tracks_path}...")
    # Using low_memory=False to avoid DtypeWarning
    tracks_df = pd.read_csv(tracks_path, low_memory=False)
    
    # Process artists metadata if available
    artists_df = None
    if os.path.exists(artists_path):
        print(f"Ingesting Kaggle artists metadata from {artists_path}...")
        artists_df = pd.read_csv(artists_path)
        artists_df = artists_df.rename(columns={
            'id': 'artist_id',
            'popularity': 'artist_popularity',
            'followers': 'artist_followers',
            'genres': 'artist_genres',
            'name': 'artist_name'
        })
        # Clean artist_genres (some are empty strings or [])
        artists_df['artist_genres'] = artists_df['artist_genres'].fillna('[]')
    
    # Check for core features
    core_features = ['acousticness', 'danceability', 'energy', 'valence', 'instrumentalness', 'liveness', 'speechiness', 'tempo', 'loudness', 'key', 'mode']
    missing_features = [f for f in core_features if f not in tracks_df.columns]
    
    if missing_features:
        print(f"Warning: tracks_df is missing some core audio features: {missing_features}")
    
    # Helper to clean list columns
    def parse_list(x):
        if not isinstance(x, str):
            return []
        try:
            return ast.literal_eval(x)
        except:
            return []

    # Process id_artists to get the primary artist ID
    if 'id_artists' in tracks_df.columns:
        print("Extracting primary artist IDs...")
        #id_artists in tracks.csv looks like "['45tIt06XoI0Iio4LBEVpls']"
        tracks_df['artist_ids_list'] = tracks_df['id_artists'].apply(parse_list)
        tracks_df['primary_artist_id'] = tracks_df['artist_ids_list'].apply(lambda x: x[0] if len(x) > 0 else None)
    
    # Merge with artists_df to get genres
    if artists_df is not None and 'primary_artist_id' in tracks_df.columns:
        print("Merging tracks with artist metadata (genres, etc.)...")
        tracks_df = tracks_df.merge(artists_df, left_on='primary_artist_id', right_on='artist_id', how='left')
    
    # Basic cleaning
    tracks_df = tracks_df.dropna(subset=['id', 'name'])
    tracks_df = tracks_df.drop_duplicates(subset=['id'])
    
    # Normalize features (Min-Max) for similarity calculations
    # Already normalized in Kaggle 0-1 range for audio features, but others need it
    norm_cols = ['tempo', 'loudness', 'popularity', 'duration_ms']
    if artists_df is not None:
        norm_cols += ['artist_popularity', 'artist_followers']

    for col in norm_cols:
        if col in tracks_df.columns:
            min_val = tracks_df[col].min()
            max_val = tracks_df[col].max()
            if max_val > min_val:
                tracks_df[f'{col}_norm'] = (tracks_df[col] - min_val) / (max_val - min_val)
            else:
                tracks_df[f'{col}_norm'] = 0.0

    output_path = "data/processed/kaggle_tracks_cleaned.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tracks_df.to_csv(output_path, index=False)
    
    print(f"Enriched & cleaned {len(tracks_df)} tracks. Saved to {output_path}")

if __name__ == "__main__":
    ingest_kaggle()
