# Reinforcement Learning Agents (DQN, REINFORCE, A2C)

## Project Overview

This project implements and compares multiple Reinforcement Learning (RL) algorithms using Gymnasium environments.
The objective is to train agents that learn optimal behavior through interaction with an environment and maximize cumulative rewards.

This project is based on the following requirements: 

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

## Results and Comparison

* Training rewards are recorded per episode
* Performance comparison between:

  * DQN
  * REINFORCE
  * A2C

Evaluation focuses on:

* Learning speed
* Stability
* Final performance

---

## Project Structure

```
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

