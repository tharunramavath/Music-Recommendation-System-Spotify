import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedBottom(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(SharedBottom, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class MultiTaskRanker(nn.Module):
    def __init__(self, input_dim=128):
        super(MultiTaskRanker, self).__init__()
        # Shared representations
        self.shared = SharedBottom(input_dim)
        
        # Like Head (Binary Classification)
        self.like_tower = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Skip Head (Binary Classification)
        self.skip_tower = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x is the combined vector of [User Embedding, Item Embedding, Context]
        shared_out = self.shared(x)
        
        like_prob = self.like_tower(shared_out)
        skip_prob = self.skip_tower(shared_out)
        
        return like_prob, skip_prob

if __name__ == "__main__":
    model = MultiTaskRanker(input_dim=138) # (64+64+10)
    dummy_in = torch.randn(8, 140)
    like_p, skip_p = model(dummy_in)
    print(f"Like prob: {like_p[0].item():.4f}, Skip prob: {skip_p[0].item():.4f}")
