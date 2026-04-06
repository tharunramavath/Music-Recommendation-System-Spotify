import torch
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    def __init__(self, user_vocab_size, embedding_dim=64, context_dim=10):
        super(UserTower, self).__init__()
        # User ID embedding
        self.user_embedding = nn.Embedding(user_vocab_size, embedding_dim)
        
        # Context features (time, device, location, etc.)
        self.context_fc = nn.Linear(context_dim, 32)
        
        # Combined dense layers
        self.fc1 = nn.Linear(embedding_dim + 32, 128)
        self.fc2 = nn.Linear(128, 64) # This 64 must match ItemTower output
        
    def forward(self, user_ids, context_features):
        u_emb = self.user_embedding(user_ids)
        c_emb = F.relu(self.context_fc(context_features))
        
        x = torch.cat([u_emb, c_emb], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Final embedding space (L2 normalize for cosine similarity)
        return F.normalize(x, p=2, dim=1)

class ItemTower(nn.Module):
    def __init__(self, audio_feature_dim=11, genre_vocab_size=5000, genre_emb_dim=32):
        super(ItemTower, self).__init__()
        # Content features (audio features)
        self.audio_fc = nn.Linear(audio_feature_dim, 64)
        
        # Genre profile (mean of genre embeddings if multiple)
        self.genre_embedding = nn.Embedding(genre_vocab_size, genre_emb_dim)
        
        # Combined dense layers
        self.fc1 = nn.Linear(64 + genre_emb_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        
    def forward(self, audio_features, genre_ids):
        # audio_features: (batch, dim)
        # genre_ids: (batch,) for primary genre, or average of multiple
        
        a_emb = F.relu(self.audio_fc(audio_features))
        g_emb = self.genre_embedding(genre_ids)
        
        x = torch.cat([a_emb, g_emb], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # L2 normalize
        return F.normalize(x, p=2, dim=1)

class TwoTowerModel(nn.Module):
    def __init__(self, user_tower, item_tower):
        super(TwoTowerModel, self).__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower
        
    def forward(self, user_ids, context_features, audio_features, genre_ids):
        user_vector = self.user_tower(user_ids, context_features)
        item_vector = self.item_tower(audio_features, genre_ids)
        
        # Return dot product (cosine similarity since vectors are normalized)
        return torch.sum(user_vector * item_vector, dim=1)

if __name__ == "__main__":
    # Test shapes
    u_tower = UserTower(user_vocab_size=100)
    i_tower = ItemTower(audio_feature_dim=11)
    model = TwoTowerModel(u_tower, i_tower)
    
    # Dummy data
    u_ids = torch.randint(0, 100, (8,))
    ctx = torch.randn(8, 10)
    aud = torch.randn(8, 11)
    gen = torch.randint(0, 1000, (8,))
    
    scores = model(u_ids, ctx, aud, gen)
    print(f"Output scores shape: {scores.shape}") # Expect 8
    print(f"Sample score: {scores[0].item():.4f}")
