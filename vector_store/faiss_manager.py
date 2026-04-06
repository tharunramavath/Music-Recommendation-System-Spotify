import faiss
import numpy as np
import os
import pickle
from dotenv import load_dotenv

load_dotenv()

class FAISSManager:
    def __init__(self, dimension=64, index_path=None, metadata_path=None):
        self.dimension = dimension
        self.index_path = index_path or os.getenv("FAISS_INDEX_PATH", "./data/processed/faiss_index.bin")
        self.metadata_path = metadata_path or os.getenv("FAISS_METADATA_PATH", "./data/processed/track_ids.pkl")
        
        # Initialize index
        if os.path.exists(self.index_path):
            print(f"Loading existing FAISS index from {self.index_path}...")
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                self.track_ids = pickle.load(f)
        else:
            print(f"Initializing new FAISS L2 index (dim={dimension})...")
            # IndexFlatL2 for Euclidean distance
            self.index = faiss.IndexFlatL2(dimension)
            self.track_ids = []

    def add_vectors(self, vectors, ids):
        # vectors: NumPy array of shape (N, dimension)
        # ids: List of track IDs corresponding to the vectors
        print(f"Adding {len(vectors)} vectors to FAISS...")
        
        # FAISS expects float32
        vectors = vectors.astype('float32')
        self.index.add(vectors)
        self.track_ids.extend(ids)
        
        self._save()

    def _save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.track_ids, f)
        print(f"FAISS index and metadata saved.")

    def search(self, query_vector, k=5):
        # query_vector: NumPy array of shape (1, dimension) or (dimension,)
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
            
        _query_vector = query_vector.astype('float32')
        distances, indices = self.index.search(_query_vector, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1: # -1 indicates no match found (or error)
                results.append({
                    "track_id": self.track_ids[idx],
                    "distance": float(dist)
                })
        return results

if __name__ == "__main__":
    # Quick test (requires dummy data)
    try:
        dim = 12
        test_vectors = np.random.random((10, dim)).astype('float32')
        test_ids = [f"track_{i}" for i in range(10)]
        
        fm = FAISSManager(dimension=dim, index_path="./tmp/test_index.bin", metadata_path="./tmp/test_meta.pkl")
        fm.add_vectors(test_vectors, test_ids)
        
        q_vec = test_vectors[0]
        results = fm.search(q_vec, k=3)
        print(f"Search results for track_0: {results}")
    except Exception as e:
        print(f"Error initializing FAISS: {e}")
