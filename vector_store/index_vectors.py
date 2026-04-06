import os
import pickle
import numpy as np
import faiss
from vector_store.faiss_manager import FAISSManager

def index_all_tracks(vector_path=None, fresh=True):
    # Default to two-tower vectors if they exist, otherwise audio vectors
    if vector_path is None:
        two_tower_path = "data/processed/two_tower_item_vectors.pkl"
        if os.path.exists(two_tower_path):
            vector_path = two_tower_path
        else:
            vector_path = "data/processed/track_vectors.pkl"

    if not os.path.exists(vector_path):
        print(f"Vectors not found at {vector_path}. Please run audio_features.py or export_embeddings.py first.")
        return

    # Clear existing index if fresh start requested
    index_file = os.getenv("FAISS_INDEX_PATH", "./data/processed/faiss_index.bin")
    meta_file = os.getenv("FAISS_METADATA_PATH", "./data/processed/track_ids.pkl")
    if fresh:
        for f in [index_file, meta_file]:
            if os.path.exists(f):
                print(f"Removing existing index file: {f}")
                os.remove(f)

    print(f"Loading track vectors from {vector_path}...")
    with open(vector_path, 'rb') as f:
        track_to_vector = pickle.load(f)

    track_ids = list(track_to_vector.keys())
    # FAISS expects float32
    vectors = np.array([track_to_vector[tid] for tid in track_ids]).astype('float32')
    
    dim = vectors.shape[1]
    print(f"Indexing {len(track_ids)} tracks with dimension {dim}...")

    # Initialize manager (Force new dimension if current doesn't match)
    manager = FAISSManager(dimension=dim, index_path=index_file, metadata_path=meta_file)
    
    if manager.index.ntotal > 0 and manager.dimension != dim:
        print(f"Dimension mismatch (Index:{manager.dimension}, Data:{dim}). Creating fresh index.")
        manager.index = faiss.IndexFlatL2(dim)
        manager.dimension = dim
        manager.track_ids = []

    # Add to index
    manager.add_vectors(vectors, track_ids)
    
    print(f"Successfully indexed {len(track_ids)} tracks in FAISS (dim={dim}).")

if __name__ == "__main__":
    index_all_tracks()
