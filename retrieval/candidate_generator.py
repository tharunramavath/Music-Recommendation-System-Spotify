import os
import pickle
import pandas as pd
import numpy as np
from vector_store.faiss_manager import FAISSManager

class CandidateGenerator:
    def __init__(self, metadata_path="data/processed/kaggle_tracks_cleaned.csv", vector_path="data/processed/two_tower_item_vectors.pkl"):
        self.metadata_path = metadata_path
        self.vector_path = vector_path
        
        # Load metadata for enrichment
        if os.path.exists(metadata_path):
            print(f"Loading metadata from {metadata_path}...")
            self.df = pd.read_csv(metadata_path, low_memory=False)
            # Create a quick lookup for name/artist
            self.tracks_info = self.df.set_index('id')[['name', 'artist_name', 'artist_genres']].to_dict('index')
        else:
            print(f"Warning: Metadata not found at {metadata_path}")
            self.df = None
            self.tracks_info = {}

        # Load vectors for query lookup
        if os.path.exists(vector_path):
            print(f"Loading track vectors from {vector_path}...")
            with open(vector_path, 'rb') as f:
                self.track_to_vector = pickle.load(f)
        else:
            print(f"Warning: Vectors not found at {vector_path}")
            self.track_to_vector = {}

        # Initialize FAISS Manager
        # It will automatically load the local index from .env paths
        self.faiss_manager = FAISSManager()

    def get_similar_tracks(self, track_id, k=50):
        """Find tracks similar to a given target track ID."""
        if track_id not in self.track_to_vector:
            print(f"Track ID {track_id} not found in vectors.")
            return []

        query_vector = self.track_to_vector[track_id]
        results = self.faiss_manager.search(query_vector, k=k+1)
        
        # Filter out the track itself
        candidates = [res for res in results if res['track_id'] != track_id]
        
        # Enrich with metadata
        for cand in candidates:
            info = self.tracks_info.get(cand['track_id'], {})
            cand.update(info)
            
        return candidates[:k]

    def get_user_candidate_pool(self, user_id, seed_track_ids=None, k=500):
        """
        Generate a broad pool of candidates for a user.
        For now (Milestone 2/3), it uses seed tracks (liked tracks).
        In the future, this will use Two-Tower and GNN outputs.
        """
        if not seed_track_ids:
            # Fallback: Popular tracks or random
            print("No seed tracks provided. Returning popular tracks as candidates.")
            popular_ids = self.df.sort_values(by='popularity', ascending=False).head(k)['id'].tolist()
            return [{"track_id": tid, **self.tracks_info.get(tid, {})} for tid in popular_ids]

        # Use average vector of seed tracks for retrieval
        seed_vectors = [self.track_to_vector[tid] for tid in seed_track_ids if tid in self.track_to_vector]
        if not seed_vectors:
            return []
            
        avg_vector = np.mean(seed_vectors, axis=0)
        results = self.faiss_manager.search(avg_vector, k=k)
        
        # Enrich
        for res in results:
            info = self.tracks_info.get(res['track_id'], {})
            res.update(info)
            
        return results

if __name__ == "__main__":
    # Test the generator
    gen = CandidateGenerator()
    if gen.df is not None:
        random_track = gen.df.iloc[0]['id']
        print(f"\nFinding similar tracks for: {gen.tracks_info[random_track]['name']} by {gen.tracks_info[random_track]['artist_name']}")
        similar = gen.get_similar_tracks(random_track, k=5)
        for i, s in enumerate(similar):
            print(f"{i+1}. {s['name']} by {s['artist_name']} (Dist: {s['distance']:.4f})")
