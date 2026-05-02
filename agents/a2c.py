import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= NETWORK =================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.actor = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)

        # Orthogonal init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)


# ================= HELPER =================
def safe_normalize(x):
    if x.numel() <= 1:
        return x
    std = x.std(unbiased=False)
    if std < 1e-8:
        return x
    return (x - x.mean()) / (std + 1e-8)


# ================= AGENT =================
class A2CAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        rollout_len=20,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
    ):
        self.gamma = gamma
        self.rollout_len = rollout_len
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.model = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.reset_buffer()

    def reset_buffer(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []

    # ================= ACTION =================
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, _ = self.model(state)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

        return action.item()

    # ================= STEP =================
    def step(self, state, action, reward, next_state, done):
        # scale reward
        reward = reward / 100.0

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(float(done))
        self.next_states.append(next_state)

        # only update on full rollout
        if len(self.states) >= self.rollout_len:
            self.update()

        # force update at episode end
        if done:
            self.update()

    # ================= UPDATE =================
    def update(self):
        if len(self.states) < 2:
            self.reset_buffer()
            return

        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.LongTensor(self.actions).to(device)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)

        # ===== Bootstrap =====
        with torch.no_grad():
            next_state = torch.FloatTensor(self.next_states[-1]).unsqueeze(0).to(device)
            _, next_value = self.model(next_state)
            R = next_value.item() * (1.0 - dones[-1])

        # ===== Returns =====
        returns = []
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1.0 - d)
            returns.insert(0, R)

        returns = torch.FloatTensor(returns).to(device)
        returns = safe_normalize(returns)

        # ===== Forward =====
        logits, values = self.model(states)
        values = values.squeeze(-1)

        advantages = returns - values.detach()
        advantages = safe_normalize(advantages)

        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy = dist.entropy().mean()

        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

        # ===== NaN protection =====
        if torch.isnan(loss):
            print("⚠️ NaN detected, skipping update")
            self.reset_buffer()
            return

        # ===== Optimize =====
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.reset_buffer()