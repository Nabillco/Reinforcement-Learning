import os
import gymnasium as gym
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
from agents.dqn import DQNAgent


def train_dqn(
    n_episodes: int = 1000,
    max_t: int = 8000,
    eps_start: float = 1.0,
    eps_end: float = 0.01,
    eps_decay: float = 0.995,
    solve_threshold: float = 200.0,
) -> list:
    env = gym.make("LunarLander-v3")
    state_size  = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size=state_size, action_size=action_size, seed=0)

    scores         = []
    scores_window  = deque(maxlen=100)
    eps            = eps_start

    print("🚀 Training DQN on LunarLander-v3")
    print("-" * 50)

    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0

        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                break

        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)

        print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}", end="")

        if i_episode % 100 == 0:
            print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}")

        if np.mean(scores_window) >= solve_threshold:
            print(
                f"\n🎉 Environment solved in {i_episode - 100:d} episodes!"
                f"\tAverage Score: {np.mean(scores_window):.2f}"
            )
            # BUG FIX: save into the checkpoints/ directory (was saving to root)
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(agent.qnetwork_local.state_dict(), "checkpoints/checkpoint_dqn.pth")
            break

    env.close()
    return scores


if __name__ == "__main__":
    print("Starting DQN training...")
    scores = train_dqn()

    os.makedirs("pictures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(len(scores)), scores, alpha=0.4, color="gray", label="Raw Score")
    # Rolling average
    window = 100
    rolling = [np.mean(scores[max(0, i - window):i + 1]) for i in range(len(scores))]
    ax.plot(rolling, color="steelblue", linewidth=2, label="100-Ep Rolling Avg")
    ax.axhline(200, color="green", linestyle="--", alpha=0.7, label="Solve Threshold (200)")
    ax.set_ylabel("Score")
    ax.set_xlabel("Episode #")
    ax.set_title("DQN Training on LunarLander-v3")
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("pictures/dqn_graph.png", dpi=150)
    plt.show()
    print("📊 Saved to pictures/dqn_graph.png")