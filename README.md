# Reinforcement Learning Agents (DQN, REINFORCE, A2C)

## Project Overview

This project implements and compares multiple Reinforcement Learning (RL) algorithms using Gymnasium environments.
The objective is to train agents that learn optimal behavior through interaction with an environment and maximize cumulative rewards.

The project includes implementation, training, and evaluation of value-based and policy-based methods, along with a comparison of their performance.

---

## Implemented Algorithms

### Deep Q-Network (DQN)

* Experience Replay
* Target Network
* Epsilon-Greedy Exploration

### REINFORCE (Policy Gradient)

* Monte Carlo Policy Gradient
* Episodic training
* Stochastic policy updates

### Advantage Actor-Critic (A2C)

* Actor-Critic architecture
* Advantage estimation
* Improved stability compared to vanilla policy gradients

---

## Environment

* Gymnasium environments (e.g., LunarLander-v3)
* Supports:

  * Discrete action spaces
  * Vector-based observations

---

## Results

### DQN Agent

<p align="center">
  <img src="https://raw.githubusercontent.com/Nabillco/Reinforcement-Learning/main/DQN_episode.gif" width="600"/>
</p>

### A2C Agent

<p align="center">
  <img src="https://raw.githubusercontent.com/Nabillco/Reinforcement-Learning/main/A2C_episode.gif" width="600"/>
</p>

### REINFORCE Agent

<p align="center">
  <img src="https://raw.githubusercontent.com/Nabillco/Reinforcement-Learning/main/REINFORCE_episode.gif" width="600"/>
</p>

---

## Comparison

### Agents Behavior Comparison

<p align="center">
  <img src="https://raw.githubusercontent.com/Nabillco/Reinforcement-Learning/main/all_agents_comparison.gif" width="700"/>
</p>

### Reward Curves

<p align="center">
  <img src="https://raw.githubusercontent.com/Nabillco/Reinforcement-Learning/main/comparison_plot.png" width="700"/>
</p>

---

## Project Structure

```bash
Reinforcement-Learning/
│
├── agents/           
├── utils/            
├── train_dqn.py      
├── train_a2c.py      
├── train_pg.py       
├── compare.py        
├── requirements.txt  
├── README.md
```

---

## Installation

```bash
git clone https://github.com/Nabillco/Reinforcement-Learning.git
cd Reinforcement-Learning

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

---

## Usage

### Train DQN

```bash
python train_dqn.py
```

### Train A2C

```bash
python train_a2c.py
```

### Train REINFORCE

```bash
python train_pg.py
```

### Compare Algorithms

```bash
python compare.py
```

---

## Output

* Reward curves per episode
* Performance comparison plots
* Training logs

---

## Key Concepts

* Markov Decision Process (MDP)
* Policy Optimization
* Value Functions
* Exploration vs Exploitation
* Advantage Estimation

---

## Future Improvements

* Add PPO (Proximal Policy Optimization)
* Support continuous control environments
* Hyperparameter tuning
* TensorBoard integration

---

## License

This project is intended for educational purposes.
