import os
import pickle
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque

from agents.a2c import A2CAgent


# ================= CONFIG =================
ENV_NAME = "LunarLander-v3"
SEED = 42
EPISODES = 1500
CHECKPOINT_DIR = "checkpoints"
HISTORY_DIR = "histories"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= TRAIN =================
def train():
    env = gym.make(ENV_NAME)

    agent = A2CAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
    )

    rewards = []
    avg_rewards = []
    rolling = deque(maxlen=100)

    best_avg = -1e9

    print("🚀 Training A2C...")

    for ep in range(EPISODES):
        state, _ = env.reset(seed=SEED + ep)
        ep_reward = 0

        while True:
            action = agent.select_action(state)

            next_state, reward, done, trunc, _ = env.step(action)

            # CLIP reward
            reward = np.clip(reward, -1, 1)

            terminal = done and not trunc

            agent.step(state, action, reward, next_state, terminal)

            state = next_state
            ep_reward += reward

            if done or trunc:
                break

        rewards.append(ep_reward)
        rolling.append(ep_reward)

        avg = np.mean(rolling)
        avg_rewards.append(avg)

        if avg > best_avg and ep > 100:
            best_avg = avg
            torch.save(agent.model.state_dict(), f"{CHECKPOINT_DIR}/A2C.pth")

        if ep % 50 == 0:
            print(f"Ep {ep:4d} | Reward: {ep_reward:7.2f} | Avg100: {avg:7.2f}")

    env.close()

    with open(f"{HISTORY_DIR}/A2C.pkl", "wb") as f:
        pickle.dump((rewards, avg_rewards), f)

    print(f"\n✅ Done | Best Avg100: {best_avg:.2f}")

    return rewards, avg_rewards, agent


# ================= EVAL =================
def evaluate(agent, runs=10):
    env = gym.make(ENV_NAME)
    scores = []

    for i in range(runs):
        state, _ = env.reset(seed=SEED + 1000 + i)
        total = 0

        while True:
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).to(device)
                logits, _ = agent.model(s)
                action = torch.argmax(logits).item()

            state, reward, done, trunc, _ = env.step(action)
            total += reward

            if done or trunc:
                break

        scores.append(total)

    env.close()

    print(f"\n🧪 Eval: {np.mean(scores):.2f} ± {np.std(scores):.2f}")


# ================= PLOT =================
def plot(rewards, avg_rewards):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, alpha=0.3, label="Episode Reward")
    plt.plot(avg_rewards, linewidth=2, label="Avg(100)")
    plt.axhline(200, linestyle="--", label="Solved")
    plt.legend()
    plt.grid()
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("A2C - LunarLander")
    plt.savefig("a2c.png")
    plt.show()


# ================= MAIN =================
if __name__ == "__main__":
    rewards, avg_rewards, agent = train()
    evaluate(agent)
    plot(rewards, avg_rewards)