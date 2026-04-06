import os
import pandas as pd
import numpy as np
import pickle
from retrieval.candidate_generator import CandidateGenerator

class OnboardingManager:
    def __init__(self, generator=None):
        self.generator = generator or CandidateGenerator()
        self.df = self.generator.df
        
        # Precompute genre popularity for suggestion
        if self.df is not None and 'artist_genres' in self.df.columns:
            # Flatten genres and count
            import ast
            def parse_list(x):
                if not isinstance(x, str): return []
                try: return ast.literal_eval(x)
                except: return []

            print("Computing popular genres for onboarding suggestions...")
            genres_series = self.df['artist_genres'].apply(parse_list)
            all_genres = [g for sublist in genres_series for g in sublist]
            self.genre_counts = pd.Series(all_genres).value_counts()
        else:
            self.genre_counts = pd.Series()

    def get_top_genres(self, n=20):
        """Suggest top genres to a new user."""
        return self.genre_counts.head(n).index.tolist()

    def create_user_profile(self, favorite_genres=None, favorite_artists=None):
        """
        Create a mock user profile vector based on selections.
        This is a 'pseudo-label' seeding.
        """
        print(f"Creating user profile for genres: {favorite_genres}...")
        
        # Find tracks matching these genres
        if self.df is None: return None
        
        # Simple string matching for now (efficient enough for onboarding search)
        mask = self.df['artist_genres'].str.contains('|'.join(favorite_genres), na=False, case=False)
        seed_tracks = self.df[mask].sort_values(by='popularity', ascending=False).head(50)
        
        if seed_tracks.empty:
            print("No tracks found for these genres. Falling back to global popular.")
            seed_tracks = self.df.sort_values(by='popularity', ascending=False).head(20)
            
        seed_ids = seed_tracks['id'].tolist()
        
        # Use generator to get candidates from this profile
        candidates = self.generator.get_user_candidate_pool(user_id="new_user", seed_track_ids=seed_ids, k=50)
        
        return {
            "user_id": "new_user",
            "seed_tracks_count": len(seed_ids),
            "initial_candidates": candidates
        }

if __name__ == "__main__":
    onboard = OnboardingManager()
    top_genres = onboard.get_top_genres(10)
    print(f"Suggested genres: {top_genres}")
    
    # Simulate user choosing 'pop' and 'rock' (if they exist)
    profile = onboard.create_user_profile(favorite_genres=['pop', 'rock'])
    print(f"\nRecs for New User based on Onboarding:")
    for i, c in enumerate(profile['initial_candidates'][:5]):
        print(f"{i+1}. {c['name']} by {c['artist_name']} (Genres: {c['artist_genres']})")
