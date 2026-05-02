"""
Trains (or loads) DQN, A2C, and REINFORCE on LunarLander-v3, then
compares them with a final evaluation, a learning-curve plot, and
side-by-side visualisation videos for each agent.

Run:  python compare.py
"""

import os
import pickle

import gymnasium as gym
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")          # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

from agents.dqn       import DQNAgent
from agents.a2c       import A2CAgent
from agents.reinforce import REINFORCEAgent


# ================= CONFIG =================
ENV_NAME       = "LunarLander-v3"
SEED           = 42
EPISODES       = 1200
EVAL_RUNS      = 20

CHECKPOINT_DIR = "checkpoints"
HISTORY_DIR    = "histories"

# ---- Output path for plots and videos ----
OUTPUT_DIR = r"C:\Users\asus\Downloads\reinforce"
os.makedirs(OUTPUT_DIR,    exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR,    exist_ok=True)


# ================= EPSILON SCHEDULE =================
def epsilon(ep, eps_start=1.0, eps_end=0.01, decay=500):
    return max(eps_end, eps_start - ep / decay)


# ================= TRAIN =================
def train_agent(agent, name):
    env = gym.make(ENV_NAME)

    rewards, avg_rewards = [], []
    rolling  = deque(maxlen=100)
    best_avg = -float("inf")

    print(f"\n🚀 Training {name}...")

    for ep in range(EPISODES):
        state, _ = env.reset(seed=SEED + ep)
        total_reward = 0

        while True:
            if name == "DQN":
                action = agent.act(state, epsilon(ep))
            else:
                action = agent.select_action(state)

            next_state, reward, done, trunc, _ = env.step(action)
            terminal = done and not trunc

            if name == "REINFORCE":
                agent.store_reward(reward)
            elif name == "DQN":
                agent.step(state, action, reward, next_state, terminal)
            else:
                agent.step(state, action, reward, next_state, terminal)

            state         = next_state
            total_reward += reward

            if done or trunc:
                break

        if name == "REINFORCE":
            agent.update()

        rewards.append(total_reward)
        rolling.append(total_reward)
        avg = np.mean(rolling)
        avg_rewards.append(avg)

        if avg > best_avg and ep >= 100:
            best_avg = avg
            _save_model(agent, name)

        if ep % 100 == 0:
            print(f"  {name:<10} Ep {ep:>4d} | Avg(100): {avg:>8.2f}")

    with open(f"{HISTORY_DIR}/{name}.pkl", "wb") as f:
        pickle.dump((rewards, avg_rewards), f)

    env.close()
    return rewards, avg_rewards


# ================= SAVE / LOAD HELPERS =================
def _model_weights(agent, name):
    if name == "DQN":
        return agent.qnetwork_local.state_dict()
    return agent.model.state_dict()


def _load_weights(agent, name, path):
    sd = torch.load(path, map_location="cpu")
    if name == "DQN":
        agent.qnetwork_local.load_state_dict(sd)
    else:
        agent.model.load_state_dict(sd)


def _save_model(agent, name):
    torch.save(_model_weights(agent, name), f"{CHECKPOINT_DIR}/{name}.pth")


def try_load(agent, name):
    ckpt = f"{CHECKPOINT_DIR}/{name}.pth"
    hist = f"{HISTORY_DIR}/{name}.pkl"

    if not (os.path.exists(ckpt) and os.path.exists(hist)):
        return None

    print(f"📦 Loading {name} from checkpoint...")
    _load_weights(agent, name, ckpt)

    with open(hist, "rb") as f:
        return pickle.load(f)


# ================= EVALUATE =================
def get_greedy_action(agent, state_t, name):
    if name == "DQN":
        logits = agent.qnetwork_local(state_t)
    else:
        output = agent.model(state_t)
        logits = output[0] if isinstance(output, tuple) else output

    return torch.argmax(logits).item()


def evaluate(agents):
    print("\n🧪 Final Evaluation (greedy policy):")
    env = gym.make(ENV_NAME)

    results = {}
    for name, agent in agents.items():
        scores = []

        for i in range(EVAL_RUNS):
            state, _ = env.reset(seed=SEED + i)
            total    = 0

            while True:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    action  = get_greedy_action(agent, state_t, name)

                state, reward, done, trunc, _ = env.step(action)
                total += reward

                if done or trunc:
                    break

            scores.append(total)

        mean, std = np.mean(scores), np.std(scores)
        results[name] = (mean, std)
        solved = "✅" if mean >= 200 else "❌"
        print(f"  {solved} {name:<10}  {mean:>8.2f} ± {std:.2f}")

    env.close()
    return results


# ================= PLOT =================
def plot(histories, results):
    """
    Two-panel figure:
      Top:    learning curves (raw + smoothed) for all agents
      Bottom: bar chart of final evaluation scores with ±1 std error bars
    Saved to OUTPUT_DIR/comparison_plot.png
    """
    colors = {"DQN": "#2196F3", "A2C": "#FF9800", "REINFORCE": "#4CAF50"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 12))

    # ---- Panel 1: Learning curves ----
    for name, (raw, avg) in histories.items():
        c = colors.get(name, None)
        ax1.plot(raw, alpha=0.15, color=c)
        ax1.plot(avg, label=name, color=c, linewidth=2)

    ax1.axhline(200, linestyle="--", color="gray", linewidth=1.2, label="Solved (200)")
    ax1.set_xlabel("Episodes",          fontsize=13)
    ax1.set_ylabel("Avg Reward (100)",  fontsize=13)
    ax1.set_title("RL Comparison — LunarLander-v3 (Training)",  fontsize=15)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    # ---- Panel 2: Bar chart of final evaluation ----
    names  = list(results.keys())
    means  = [results[n][0] for n in names]
    stds   = [results[n][1] for n in names]
    bars   = ax2.bar(
        names, means, yerr=stds,
        color=[colors.get(n, "#999") for n in names],
        capsize=8, edgecolor="black", linewidth=0.8, alpha=0.85,
    )
    ax2.axhline(200, linestyle="--", color="gray", linewidth=1.2, label="Solved (200)")
    ax2.set_ylabel("Mean Reward (eval)", fontsize=13)
    ax2.set_title("Final Evaluation Performance",  fontsize=15)
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3, axis="y")

    # Annotate bar values
    for bar, m, s in zip(bars, means, stds):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + s + 5,
            f"{m:.1f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    fig.tight_layout(pad=3)
    out_path = os.path.join(OUTPUT_DIR, "comparison_plot.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"📊 Plot saved → {out_path}")


# ================= VIDEO RECORDING =================
def _record_episode_frames(agent, name, seed=0):
    """
    Run one greedy episode and collect RGB frames.
    Returns list[np.ndarray] of shape (H, W, 3).
    """
    env    = gym.make(ENV_NAME, render_mode="rgb_array")
    state, _ = env.reset(seed=seed)
    frames = [env.render()]

    while True:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action  = get_greedy_action(agent, state_t, name)

        state, _, done, trunc, _ = env.step(action)
        frames.append(env.render())

        if done or trunc:
            break

    env.close()
    return frames


def _save_individual_video(frames, name, fps=30):
    """
    Save a single agent's episode as an MP4.
    Requires matplotlib (uses ArtistAnimation — no extra codec needed).
    Falls back gracefully if ffmpeg is absent.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.axis("off")
    ax.set_title(name, fontsize=14, fontweight="bold", pad=6)

    patches = [[ax.imshow(f, animated=True)] for f in frames]
    ani = animation.ArtistAnimation(fig, patches, interval=1000 // fps, blit=True)

    out_path = os.path.join(OUTPUT_DIR, f"{name}_episode.mp4")
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        ani.save(out_path, writer=writer)
        print(f"🎬 {name} video saved → {out_path}")
    except Exception as e:
        # ffmpeg not available — save as gif instead
        out_path = out_path.replace(".mp4", ".gif")
        ani.save(out_path, writer="pillow", fps=fps)
        print(f"🎬 {name} GIF saved (ffmpeg unavailable: {e}) → {out_path}")

    plt.close(fig)
    return out_path


def _save_combined_video(all_frames, names, fps=30):
    """
    Side-by-side video of all three agents running simultaneously.
    Pads shorter episodes with their last frame so they stay in sync.
    """
    colors_hex = {"DQN": "#2196F3", "A2C": "#FF9800", "REINFORCE": "#4CAF50"}
    n = len(names)

    # Pad to equal length
    max_len = max(len(f) for f in all_frames)
    padded  = [
        frames + [frames[-1]] * (max_len - len(frames))
        for frames in all_frames
    ]

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5))
    fig.suptitle("LunarLander-v3 — Agent Comparison", fontsize=14, fontweight="bold", y=1.01)

    for ax, name in zip(axes, names):
        ax.axis("off")
        color = colors_hex.get(name, "black")
        ax.set_title(name, fontsize=13, fontweight="bold", color=color, pad=4)

    patches = []
    for t in range(max_len):
        frame_artists = [
            axes[i].imshow(padded[i][t], animated=True)
            for i in range(n)
        ]
        patches.append(frame_artists)

    ani = animation.ArtistAnimation(fig, patches, interval=1000 // fps, blit=True)

    out_path = os.path.join(OUTPUT_DIR, "all_agents_comparison.mp4")
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
        ani.save(out_path, writer=writer, bbox_inches="tight")
        print(f"🎬 Combined video saved → {out_path}")
    except Exception as e:
        out_path = out_path.replace(".mp4", ".gif")
        ani.save(out_path, writer="pillow", fps=fps)
        print(f"🎬 Combined GIF saved (ffmpeg unavailable: {e}) → {out_path}")

    plt.close(fig)


def record_videos(agents):
    """Record one episode per agent and save individual + combined videos."""
    print("\n🎥 Recording agent episodes...")
    all_frames = []
    names      = list(agents.keys())

    for name, agent in agents.items():
        print(f"  Recording {name}...")
        frames = _record_episode_frames(agent, name, seed=SEED)
        all_frames.append(frames)
        _save_individual_video(frames, name)

    _save_combined_video(all_frames, names)
    print("✅ All videos saved!")


# ================= MAIN =================
if __name__ == "__main__":
    env   = gym.make(ENV_NAME)
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n
    env.close()

    agents = {
        "DQN":       DQNAgent(s_dim, a_dim, seed=SEED),
        "A2C":       A2CAgent(s_dim, a_dim),
        "REINFORCE": REINFORCEAgent(s_dim, a_dim),
    }

    histories = {}

    for name, agent in agents.items():
        loaded = try_load(agent, name)

        if loaded is not None:
            print(f"✅ Using saved {name}")
            histories[name] = loaded
        else:
            print(f"⚠️  Training {name} from scratch")
            histories[name] = train_agent(agent, name)

    results = evaluate(agents)
    plot(histories, results)
    record_videos(agents)