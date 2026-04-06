import random

class DiscoveryBandit:
    def __init__(self, epsilon=0.15):
        """
        epsilon: The probability of taking an 'exploration' action.
        """
        self.epsilon = epsilon
        
    def re_rank(self, ranked_items, pool_candidates, k=20):
        """
        Combines top ranked items with exploratory items.
        ranked_items: list of (track_id, score) from the Ranker.
        pool_candidates: list of all candidates from Retrieval.
        k: output list size.
        """
        n_exploit = int(k * (1 - self.epsilon))
        n_explore = k - n_exploit
        
        # Exploit: Top K items from ranked list
        final_list = ranked_items[:n_exploit]
        
        # Explore: Randomized sampling from the pool, excluding already selected
        selected_ids = {item['track_id'] for item in final_list}
        available_explores = [item for item in pool_candidates if item['track_id'] not in selected_ids]
        
        if len(available_explores) >= n_explore:
            explorations = random.sample(available_explores, n_explore)
        else:
            explorations = available_explores
            
        # Add labels to indicate source (for monitoring)
        for item in final_list: item['discovery_type'] = 'exploit'
        for item in explorations: item['discovery_type'] = 'explore'
        
        # Merge and shuffle (for smoother experience)
        res = final_list + explorations
        random.shuffle(res)
        
        return res

if __name__ == "__main__":
    bandit = DiscoveryBandit(epsilon=0.2)
    ranked = [{"track_id": f"T{i}", "score": 0.9-i*0.01} for i in range(50)]
    pool = [{"track_id": f"T{i}"} for i in range(200)]
    
    final = bandit.re_rank(ranked, pool, k=10)
    print(f"Final Playlist (Size {len(final)}):")
    for item in final:
        print(f"- {item['track_id']} ({item['discovery_type']})")
