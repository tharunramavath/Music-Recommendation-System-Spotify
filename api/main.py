from fastapi import FastAPI, HTTPException, Body
import torch
import os
import pickle
import numpy as np
import pandas as pd
from typing import List, Optional
from retrieval.candidate_generator import CandidateGenerator
from retrieval.onboarding import OnboardingManager
from ranking.mtl_model import MultiTaskRanker
from retrieval.two_tower_model import UserTower
from reranking.bandit import DiscoveryBandit
from reranking.llm_intent_parser import IntentParser

from fastapi.middleware.cors import CORSMiddleware

# Load Environment and Metadata
app = FastAPI(title="Spotify Music Recommender API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global State / Models ---
class ServiceState:
    def __init__(self):
        print("Initializing Recommender Service Pipeline...")
        # Data / Enrichment
        self.metadata = pd.read_csv("data/processed/kaggle_tracks_cleaned.csv", low_memory=False).set_index('id')
        self.tracks_info = self.metadata[['name', 'artist_name', 'artist_genres']].to_dict('index')
        
        # Retrieval Components
        self.generator = CandidateGenerator()
        self.onboarding = OnboardingManager(generator=self.generator)

        # Ranking Config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Two-Tower for user embedding generation
        self.user_tower = UserTower(user_vocab_size=10000)
        state_dict_tt = torch.load("models/two_tower_epoch5.pth", map_location=self.device)
        u_state = {k.replace('user_tower.', ''): v for k, v in state_dict_tt.items() if k.startswith('user_tower.')}
        self.user_tower.load_state_dict(u_state)
        self.user_tower.to(self.device).eval()

        # Load Heavy Ranker
        self.ranker = MultiTaskRanker(input_dim=138)
        self.ranker.load_state_dict(torch.load("models/ranking_mtl.pth", map_location=self.device))
        self.ranker.to(self.device).eval()

        # Re-Ranking Logic
        self.bandit = DiscoveryBandit(epsilon=0.15)
        self.intent_parser = IntentParser()

        # Simple In-Memory User Profiles (Synthetic for now)
        self.user_profiles = {} # user_id -> {'seeds': list of track_ids}

    def get_user_embedding(self, user_id, context):
        """Generates dynamic user embedding using the User Tower."""
        with torch.no_grad():
            u_id_tensor = torch.tensor([user_id], dtype=torch.long).to(self.device)
            ctx_tensor = torch.tensor([context], dtype=torch.float32).to(self.device)
            u_emb = self.user_tower(u_id_tensor, ctx_tensor)
            return u_emb.squeeze(0).cpu().numpy()

    def rank_candidates(self, user_emb, candidate_ids, context):
        """Scores 500 candidates using the MTL model."""
        features_list = []
        valid_ids = []
        
        # Normally item embeddings are pre-loaded in memory for speed
        item_vectors_path = "data/processed/two_tower_item_vectors.pkl"
        if not hasattr(self, 'item_vecs'):
            with open(item_vectors_path, 'rb') as f:
                self.item_vecs = pickle.load(f)

        for tid in candidate_ids:
            i_emb = self.item_vecs.get(tid)
            if i_emb is not None:
                combined = np.concatenate([user_emb, i_emb, context])
                features_list.append(combined)
                valid_ids.append(tid)

        if not features_list: return []

        # Batch forward pass for speed
        x_tensor = torch.tensor(np.array(features_list), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            like_probs, skip_probs = self.ranker(x_tensor)
            # Final ranking score = LikeProb * (1 - SkipProb)
            final_scores = (like_probs * (1 - skip_probs)).squeeze().cpu().numpy()
            
        ranked = []
        for tid, score in zip(valid_ids, final_scores):
            info = self.tracks_info.get(tid, {})
            ranked.append({
                "track_id": tid,
                "score": float(score),
                **info
            })
        
        # Sort by score descending
        return sorted(ranked, key=lambda x: x['score'], reverse=True)

# --- Instantiation ---
state = ServiceState()

@app.get("/")
def read_root():
    return {"message": "Spotify Music Recommender Service - Online", "active_tracks": len(state.metadata)}

@app.get("/genres")
def get_genres():
    return {"top_genres": state.onboarding.get_top_genres(20)}

@app.post("/onboarding")
def user_onboarding(user_id: int, genres: List[str] = Body(...)):
    profile = state.onboarding.create_user_profile(favorite_genres=genres)
    state.user_profiles[user_id] = {"seeds": [t['track_id'] for t in profile['initial_candidates'][:10]]}
    return {"message": "Onboarding complete", "num_seeds": len(state.user_profiles[user_id]['seeds'])}

@app.get("/recommend/{user_id}")
def get_personalized_recommendations(user_id: int, k: int = 50):
    # Input validation
    if user_id < 0:
        raise HTTPException(status_code=400, detail="user_id must be non-negative")
    if k <= 0:
        raise HTTPException(status_code=400, detail="k must be positive")
    
    # 1. Fetch User Context
    # tod: time of day, dev: device (hardcoded mobile=1 for demo)
    import datetime
    tod = datetime.datetime.now().hour / 23.0
    context = np.zeros(10)
    context[0] = tod
    context[1] = 1.0 / 3.0 # Mobile
    
    # 2. Get User Embedding
    user_emb = state.get_user_embedding(user_id % 10000, context)

    # 3. Candidate Generation (Retrieval)
    # Using User Profile Seeds if available
    profile = state.user_profiles.get(user_id, {"seeds": []})
    candidate_results = state.generator.get_user_candidate_pool(user_id=user_id, seed_track_ids=profile['seeds'], k=300)
    
    # Fallback for empty candidates
    if not candidate_results:
        popular_ids = state.metadata.sort_values(by='popularity', ascending=False).head(k)['id'].tolist()
        candidate_results = [{"track_id": tid, **state.tracks_info.get(tid, {})} for tid in popular_ids]
    
    candidate_ids = [c['track_id'] for c in candidate_results]

    # 4. Heavy Ranking (ML Ranker)
    ranked_tracks = state.rank_candidates(user_emb, candidate_ids, context)
    
    # 5. Re-Ranking / Bandit Exploration
    final_output = state.bandit.re_rank(ranked_tracks, candidate_results, k=k)
    
    return {
        "user_id": user_id,
        "recommendations": final_output,
        "total_served": len(final_output)
    }

@app.get("/search")
def semantic_search(query: str):
    """Semantic search powered by Intent Parser + Retrieval."""
    intent = state.intent_parser.parse_intent(query)
    # Use intent genres as seeds
    profile = state.onboarding.create_user_profile(favorite_genres=intent['genres'] if intent['genres'] else ['pop'])
    # Return best matches from onboarding
    return {
        "query": query,
        "intent_detected": intent,
        "results": profile['initial_candidates'][:20]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
