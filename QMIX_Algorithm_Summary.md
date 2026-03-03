# QMIX Algorithm Summary

> Source: *Multi-Agent Reinforcement Learning: Foundations and Modern Approaches*, Chapter 9

---

## 1. Background & Motivation

QMIX (Rashid et al., 2018) is a prominent **value decomposition** algorithm in multi-agent deep reinforcement learning, building upon VDN (Value Decomposition Networks).

### Limitations of VDN

VDN assumes the centralized action-value function can be linearly decomposed into the sum of individual agent utility functions:

```
Q(h, a; θ) = Σᵢ Q(hᵢ, aᵢ; θᵢ)
```

This linear assumption is overly restrictive and fails to capture non-linear interactions between agents.

---

## 2. Core Idea of QMIX

### The IGM Property (Individual-Global-Max)

QMIX aims to satisfy the IGM property: the joint action that is optimal for the centralized value function must also be individually optimal for each agent's utility function:

```
arg max_a Q(h, z, a; θ) = (arg max_{a₁} Q(h₁, a₁; θ₁), ..., arg max_{aₙ} Q(hₙ, aₙ; θₙ))
```

### Monotonicity Constraint

QMIX's key innovation is reformulating the IGM condition as a **monotonicity constraint**:

```
∀i ∈ I, ∀a ∈ A :  ∂Q(h, z, a; θ) / ∂Q(hᵢ, aᵢ; θᵢ) > 0
```

The partial derivative of the centralized value function with respect to each agent's utility must be strictly positive. Intuitively, increasing any agent's utility for its chosen action must increase the centralized value function.

**Monotonicity ⊇ Linearity**: VDN's linear decomposition is a special case of QMIX's monotonic decomposition. Therefore, the set of value functions representable by QMIX is a strict superset of those representable by VDN.

---

## 3. Network Architecture

QMIX consists of three core components:

### 3.1 Individual Utility Networks

- Each agent *i* has its own deep Q-network (similar to DQN)
- **Input**: Agent *i*'s observation history *hᵢ*
- **Output**: Utility values Q(hᵢ, aᵢ; θᵢ) for all available actions
- In the original implementation, agents **share parameters** (θᵢ = θⱼ), distinguished by a one-hot encoded agent ID
- Typically modeled as **recurrent neural networks (GRU)** to handle partial observability

### 3.2 Hypernetwork

- Takes centralized information *z* (e.g., global state *s*) as input
- Outputs the parameters θₘᵢₓ of the mixing network
- Applies an **absolute value activation** to weight outputs to ensure positive weights (and thus monotonicity)

```
θₘᵢₓ = f_hyper(z; θ_hyper)
```

### 3.3 Mixing Network

- Receives all agents' utility values and the parameters from the hypernetwork
- Aggregates utilities through a **positive-weight feedforward network**
- Outputs the centralized action-value function

```
Q(h, z, a; θ) = f_mix(Q(h₁, a₁; θ₁), ..., Q(hₙ, aₙ; θₙ); θₘᵢₓ)
```

---

## 4. Training Objective

QMIX follows the **Centralized Training with Decentralized Execution (CTDE)** paradigm. The training loss is:

```
L(θ) = (1/B) Σ_{B} [rᵗ + γ max_a Q(hᵗ⁺¹, zᵗ⁺¹, a; θ̄) - Q(hᵗ, zᵗ, aᵗ; θ)]²
```

- *B* is a mini-batch sampled from the experience replay buffer
- *θ̄* are the target network parameters (periodically updated)
- Mixing network parameters are **not directly optimized**; they are always obtained as outputs from the hypernetwork

---

## 5. Pseudocode (Algorithm 22)

```
QMIX Algorithm:

1.  Initialize n utility networks with random parameters θ₁, ..., θₙ
2.  Initialize n target networks with parameters θ̄₁ = θ₁, ..., θ̄ₙ = θₙ
3.  Initialize hypernetwork with random parameters θ_hyper
4.  Initialize shared replay buffer D

5.  For each time step t = 0, 1, 2, ...:
6.      Collect centralized information zᵗ and observations o₁ᵗ, ..., oₙᵗ
7.      For each agent i = 1, ..., n:
8.          With probability ε: select random action aᵢᵗ  (exploration)
9.          Otherwise:         aᵢᵗ = arg max_{aᵢ} Q(hᵢᵗ, aᵢ; θᵢ)  (exploitation)
10.     Execute joint action; collect shared reward rᵗ,
        next centralized info zᵗ⁺¹ and observations o₁ᵗ⁺¹, ..., oₙᵗ⁺¹
11.     Store transition (hᵗ, zᵗ, aᵗ, rᵗ, hᵗ⁺¹, zᵗ⁺¹) in replay buffer D

12.     Sample mini-batch of B transitions (hᵏ, zᵏ, aᵏ, rᵏ, hᵏ⁺¹, zᵏ⁺¹) from D
13.     If sᵏ⁺¹ is terminal:
14.         Target yᵏ ← rᵏ
15.     Else:
16.         θ_mix^{k+1} ← f_hyper(zᵏ⁺¹; θ_hyper)
17.         yᵏ ← rᵏ + γ · f_mix(max_{a₁'} Q(h₁ᵏ⁺¹, a₁'; θ̄₁), ...,
                                  max_{aₙ'} Q(hₙᵏ⁺¹, aₙ'; θ̄ₙ); θ_mix^{k+1})
18.     θ_mix^k ← f_hyper(zᵏ; θ_hyper)
19.     Compute value estimates:
        Q(hᵏ, zᵏ, aᵏ; θ) ← f_mix(Q(h₁ᵏ, a₁ᵏ; θ₁), ..., Q(hₙᵏ, aₙᵏ; θₙ); θ_mix^k)
20.     Loss L(θ) ← (1/B) Σ [yᵏ - Q(hᵏ, zᵏ, aᵏ; θ)]²
21.     Minimize L(θ) and update all parameters θ
22.     At set intervals, update target network parameters θ̄ᵢ for each agent i
```

---

## 6. PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


# =====================
# 1. Agent Utility Network
# =====================
class AgentNetwork(nn.Module):
    """Per-agent utility network using GRU for partial observability."""
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs, hidden):
        """
        obs:    (batch, obs_dim)
        hidden: (batch, hidden_dim)
        Returns: q_values (batch, action_dim), new_hidden (batch, hidden_dim)
        """
        x = F.relu(self.fc1(obs))
        new_hidden = self.gru(x, hidden)
        q_values = self.fc2(new_hidden)
        return q_values, new_hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim)


# =====================
# 2. Hypernetwork
# =====================
class HyperNetwork(nn.Module):
    """Receives global state and outputs mixing network parameters."""
    def __init__(self, state_dim, n_agents, mixing_hidden_dim=32):
        super().__init__()
        self.n_agents = n_agents
        self.mixing_hidden_dim = mixing_hidden_dim

        # Generate weights for mixing layer 1: (n_agents -> mixing_hidden_dim)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_agents * mixing_hidden_dim)
        )
        self.hyper_b1 = nn.Linear(state_dim, mixing_hidden_dim)

        # Generate weights for mixing layer 2: (mixing_hidden_dim -> 1)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, mixing_hidden_dim)
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, state):
        """
        state: (batch, state_dim)
        Returns: w1, b1, w2, b2 — mixing network parameters
        """
        batch = state.shape[0]
        # Absolute value ensures positive weights → monotonicity
        w1 = torch.abs(self.hyper_w1(state))
        w1 = w1.view(batch, self.n_agents, self.mixing_hidden_dim)
        b1 = self.hyper_b1(state).view(batch, 1, self.mixing_hidden_dim)

        w2 = torch.abs(self.hyper_w2(state))
        w2 = w2.view(batch, self.mixing_hidden_dim, 1)
        b2 = self.hyper_b2(state).view(batch, 1, 1)

        return w1, b1, w2, b2


# =====================
# 3. Mixing Network
# =====================
class MixingNetwork(nn.Module):
    """Aggregates individual utilities into the centralized value function."""
    def forward(self, agent_qs, w1, b1, w2, b2):
        """
        agent_qs: (batch, n_agents) — per-agent utility for chosen actions
        w1, b1, w2, b2: mixing network parameters from hypernetwork
        Returns: q_total (batch, 1)
        """
        x = agent_qs.unsqueeze(1)          # (batch, 1, n_agents)
        x = torch.bmm(x, w1) + b1          # (batch, 1, mixing_hidden_dim)
        x = F.elu(x)
        x = torch.bmm(x, w2) + b2          # (batch, 1, 1)
        return x.squeeze(-1)               # (batch, 1)


# =====================
# 4. Replay Buffer
# =====================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# =====================
# 5. QMIX Agent
# =====================
class QMIX:
    def __init__(self, n_agents, obs_dim, action_dim, state_dim,
                 hidden_dim=64, mixing_hidden_dim=32,
                 lr=3e-4, gamma=0.99, buffer_capacity=100000,
                 batch_size=128, target_update_interval=200):

        self.n_agents = n_agents
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.train_step = 0

        # Shared agent utility network (parameter sharing across agents)
        self.agent_net = AgentNetwork(obs_dim, action_dim, hidden_dim)
        self.target_agent_net = AgentNetwork(obs_dim, action_dim, hidden_dim)
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())

        # Hypernetwork and mixing network
        self.hyper_net = HyperNetwork(state_dim, n_agents, mixing_hidden_dim)
        self.mixing_net = MixingNetwork()
        self.target_hyper_net = HyperNetwork(state_dim, n_agents, mixing_hidden_dim)
        self.target_hyper_net.load_state_dict(self.hyper_net.state_dict())

        # Joint optimizer for all trainable parameters
        self.optimizer = torch.optim.Adam(
            list(self.agent_net.parameters()) + list(self.hyper_net.parameters()),
            lr=lr
        )

        self.buffer = ReplayBuffer(buffer_capacity)

    def select_actions(self, obs_list, hidden_list, epsilon):
        """ε-greedy decentralized action selection."""
        actions, new_hiddens = [], []
        with torch.no_grad():
            for i in range(self.n_agents):
                obs = torch.FloatTensor(obs_list[i]).unsqueeze(0)
                q_vals, new_hidden = self.agent_net(obs, hidden_list[i])
                action = (random.randint(0, self.action_dim - 1)
                          if random.random() < epsilon
                          else q_vals.argmax(dim=-1).item())
                actions.append(action)
                new_hiddens.append(new_hidden)
        return actions, new_hiddens

    def store_transition(self, obs, state, actions, reward, next_obs, next_state, done):
        """Store a transition in the replay buffer (includes centralized state)."""
        self.buffer.push((obs, state, actions, reward, next_obs, next_state, done))

    def train(self):
        """Sample a mini-batch and update all networks."""
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)
        obs_b, state_b, actions_b, reward_b, next_obs_b, next_state_b, done_b = zip(*batch)

        obs_t      = torch.FloatTensor(np.array(obs_b))
        state_t    = torch.FloatTensor(np.array(state_b))
        actions_t  = torch.LongTensor(np.array(actions_b))
        reward_t   = torch.FloatTensor(np.array(reward_b)).unsqueeze(1)
        next_obs_t = torch.FloatTensor(np.array(next_obs_b))
        next_state_t = torch.FloatTensor(np.array(next_state_b))
        done_t     = torch.FloatTensor(np.array(done_b)).unsqueeze(1)

        # ---- Current Q values ----
        agent_qs, hidden = [], self.agent_net.init_hidden(self.batch_size)
        for i in range(self.n_agents):
            q_vals, hidden = self.agent_net(obs_t[:, i, :], hidden)
            q_a = q_vals.gather(1, actions_t[:, i].unsqueeze(1))
            agent_qs.append(q_a)
        agent_qs = torch.cat(agent_qs, dim=1)          # (batch, n_agents)

        w1, b1, w2, b2 = self.hyper_net(state_t)
        q_total = self.mixing_net(agent_qs, w1, b1, w2, b2)

        # ---- Target Q values ----
        with torch.no_grad():
            target_qs, t_hidden = [], self.target_agent_net.init_hidden(self.batch_size)
            for i in range(self.n_agents):
                tq, t_hidden = self.target_agent_net(next_obs_t[:, i, :], t_hidden)
                target_qs.append(tq.max(dim=1, keepdim=True)[0])
            target_qs = torch.cat(target_qs, dim=1)

            tw1, tb1, tw2, tb2 = self.target_hyper_net(next_state_t)
            q_total_target = self.mixing_net(target_qs, tw1, tb1, tw2, tb2)
            y = reward_t + self.gamma * (1 - done_t) * q_total_target

        # ---- Loss and update ----
        loss = F.mse_loss(q_total, y)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.agent_net.parameters()) + list(self.hyper_net.parameters()), 10.0
        )
        self.optimizer.step()

        # Periodic target network update
        self.train_step += 1
        if self.train_step % self.target_update_interval == 0:
            self.target_agent_net.load_state_dict(self.agent_net.state_dict())
            self.target_hyper_net.load_state_dict(self.hyper_net.state_dict())

        return loss.item()


# =====================
# 6. Training Example (dummy environment)
# =====================
def train_example():
    n_agents, obs_dim, action_dim, state_dim = 3, 10, 5, 20
    n_episodes, max_steps = 1000, 50
    epsilon, epsilon_end, epsilon_decay = 1.0, 0.05, 0.995

    qmix = QMIX(n_agents, obs_dim, action_dim, state_dim)
    total_rewards = []

    for episode in range(n_episodes):
        hiddens = [qmix.agent_net.init_hidden(1) for _ in range(n_agents)]
        obs   = [np.random.randn(obs_dim) for _ in range(n_agents)]
        state = np.random.randn(state_dim)
        ep_reward = 0.0

        for step in range(max_steps):
            actions, hiddens = qmix.select_actions(obs, hiddens, epsilon)
            next_obs   = [np.random.randn(obs_dim) for _ in range(n_agents)]
            next_state = np.random.randn(state_dim)
            reward     = np.random.randn()          # shared reward
            done       = (step == max_steps - 1)

            qmix.store_transition(obs, state, actions, reward, next_obs, next_state, done)
            qmix.train()

            obs, state, ep_reward = next_obs, next_state, ep_reward + reward
            if done:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        total_rewards.append(ep_reward)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1:4d} | "
                  f"Avg Reward: {np.mean(total_rewards[-100:]):6.2f} | "
                  f"Epsilon: {epsilon:.3f}")

    return total_rewards


if __name__ == "__main__":
    train_example()
```

---

## 7. Experimental Results

### 7.1 Matrix Game Analysis

| Scenario | VDN | QMIX |
|----------|-----|------|
| **Linearly decomposable game** | ✅ Learns accurately | ✅ Learns accurately |
| **Monotonically decomposable game** | ❌ Cannot represent non-linear values | ✅ Learns accurately |
| **Two-step stochastic game** | ❌ Underestimates optimal policy; selects suboptimal path | ✅ Learns optimal policy |
| **Climbing game** | ❌ Converges to suboptimal (C,C) | ❌ Converges to suboptimal (C,B) |

### 7.2 Level-Based Foraging Environment

In a complex multi-agent cooperative task, QMIX **significantly outperforms** VDN and IDQN:
- Achieves the highest average returns
- Learns faster
- Shows lower variance across multiple runs

### 7.3 Limitations

- The Climbing game reveals that the monotonicity constraint can **still be too restrictive** to represent all centralized value functions
- Monotonicity is a **sufficient but not necessary** condition for the IGM property
- Subsequent work (QTRAN, QPLEX) proposes less restrictive decomposition conditions

---

## 8. Key Implementation Details (from the original paper)

| Detail | Description |
|--------|-------------|
| **Parameter sharing** | All agents share the utility network; agent identity passed via one-hot ID |
| **Recurrent networks** | GRU-based utility networks to handle partial observability |
| **Last action as input** | Each agent's observation includes its previous action |
| **Episodic replay buffer** | Stores and samples full episodes rather than individual transitions |
| **Centralized state storage** | Replay buffer stores global state *z* for the hypernetwork |
| **ε-greedy exploration** | ε linearly annealed from 1.0 to 0.05 throughout training |

---

## 9. Algorithm Comparison

| Property | IDQN | VDN | QMIX |
|----------|------|-----|------|
| Value decomposition | None | Linear | Monotonic non-linear |
| Centralized training | ✗ | ✓ | ✓ |
| Decentralized execution | ✓ | ✓ | ✓ |
| Uses global state | ✗ | ✗ | ✓ (via hypernetwork) |
| Expressive power | Low | Medium | High |
| Computational cost | Low | Low | Medium |
