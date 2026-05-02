import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Simplified to a standard policy network for REINFORCE
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1) # REINFORCE needs probabilities
        )

    def forward(self, x):
        return self.network(x)

class REINFORCEAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99): # Increased LR slightly
        self.gamma = gamma
        self.model = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.model(state)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update(self):
        # 1. Calculate discounted returns
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # 2. CRITICAL STEP: Normalize returns
        # This reduces variance significantly, helping the agent learn
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 3. Policy Gradient Update
        policy_loss = []
        for log_prob, Gt in zip(self.log_probs, returns):
            # We want to maximize (log_prob * return), so we minimize the negative
            policy_loss.append(-log_prob * Gt)

        self.optimizer.zero_grad()
        # Summing the losses for the entire episode
        total_loss = torch.cat(policy_loss).sum()
        total_loss.backward()
        self.optimizer.step()

        # Clear memory for the next episode
        self.log_probs.clear()
        self.rewards.clear()