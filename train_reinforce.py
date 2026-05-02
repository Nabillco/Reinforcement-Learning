import gymnasium as gym
import numpy as np
from collections import deque
import torch
# Ensure this import matches your file structure
from reinforce import REINFORCEAgent 

def train_reinforce(env_name="LunarLander-v3", episodes=2000):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCEAgent(state_dim, action_dim)

    all_rewards = []
    rolling_rewards = deque(maxlen=100)

    for ep in range(episodes):
        state, _ = env.reset()
        ep_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done, trunc, _ = env.step(action)

            agent.store_reward(reward)
            ep_reward += reward
            state = next_state

            if done or trunc:
                break

        # Update happens only at the end of the episode in REINFORCE
        agent.update()

        all_rewards.append(ep_reward)
        rolling_rewards.append(ep_reward)
        avg = np.mean(rolling_rewards)

        if ep % 50 == 0:
            print(f"Ep {ep} | Ep Reward: {ep_reward:.1f} | 100-Ep Avg: {avg:.1f}")

        # Solved condition for LunarLander-v3 is usually ~200
        if avg >= 200:
            print(f"Solved in {ep} episodes!")
            break

    env.close()
    return all_rewards

if __name__ == "__main__":
    train_reinforce()