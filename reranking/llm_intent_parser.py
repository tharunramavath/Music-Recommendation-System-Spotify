import os
import re
import json

class IntentParser:
    """
    Translates Natural Language queries into structured recommendation filters.
    In a real app, this would call Gemini / GPT-4.
    """
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
    def parse_intent(self, query):
        """
        Input: 'I want some high energy rock for a rainy drive'
        Output: {'mood': 'energy > 0.7', 'genres': ['rock'], 'discovery': True}
        """
        # --- MOCK LLM (Fallback Logic) ---
        query_low = query.lower()
        intent = {
            "genres": [],
            "mood_pref": None,
            "min_popularity": 0
        }
        
        # Simple keyword matching for demo
        genre_keywords = ['rock', 'pop', 'jazz', 'classical', 'dance', 'techno', 'hip hop']
        for g in genre_keywords:
            if g in query_low:
                intent["genres"].append(g)
                
        if any(word in query_low for word in ['energetic', 'workout', 'gym', 'loud', 'party']):
            intent["mood_pref"] = "high_energy"
        elif any(word in query_low for word in ['sad', 'mellow', 'rainy', 'chill', 'sleep']):
            intent["mood_pref"] = "low_energy"

        if 'famous' in query_low or 'hit' in query_low:
            intent["min_popularity"] = 0.7

        return intent

if __name__ == "__main__":
    parser = IntentParser()
    q = "I want some high energy rock for a workout"
    print(f"Query: {q}")
    print(f"Parsed Intent: {parser.parse_intent(q)}")
