import pandas as pd
import torch
import pickle
import os

def prepare_gnn_data(interaction_csv="data/raw/simulated_interactions.csv", output_path="data/processed/gnn_data.pkl"):
    print("Preparing GNN graph data from interactions...")
    df = pd.read_csv(interaction_csv)
    
    # We only care about positive signals for the co-listening graph
    pos_df = df[df['interaction_type'] == 'like']
    print(f"Positive interactions: {len(pos_df)}")

    # Map users and items to contiguous IDs for the bipartite graph
    unique_users = pos_df['user_id'].unique()
    unique_items = pos_df['track_id'].unique()
    
    user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
    # Item indices should be offset by number of users in some GNN libs, 
    # but for bipartite LightGCN we usually handle them as separate sets.
    item_to_idx = {tid: i for i, tid in enumerate(unique_items)}
    
    # Create edge_index (2, NumEdges)
    user_indices = pos_df['user_id'].map(user_to_idx).values
    item_indices = pos_df['track_id'].map(item_to_idx).values
    
    edge_index = torch.tensor([user_indices, item_indices], dtype=torch.long)
    
    data = {
        "edge_index": edge_index,
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
        "num_users": len(unique_users),
        "num_items": len(unique_items)
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
        
    print(f"GNN data prepared. Users: {len(unique_users)}, Items: {len(unique_items)}, Edges: {edge_index.shape[1]}")
    return data

if __name__ == "__main__":
    prepare_gnn_data()
