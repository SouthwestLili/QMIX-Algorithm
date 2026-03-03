# QMIX 算法总结

> 来源：《Multi-Agent Reinforcement Learning: Foundations and Modern Approaches》第9章

---

## 1. 背景与动机

QMIX（Rashid et al., 2018）是多智能体深度强化学习中**基于值分解（Value Decomposition）**的代表性算法，建立在 VDN（Value Decomposition Networks）之上。

### VDN 的局限性

VDN 假设集中式动作值函数可以被线性分解为各智能体的效用函数之和：

```
Q(h, a; θ) = Σᵢ Q(hᵢ, aᵢ; θᵢ)
```

这种线性假设过于简单，无法表示智能体之间的非线性交互关系。

---

## 2. QMIX 的核心思想

### IGM 属性（Individual-Global-Max）

QMIX 的目标是满足 IGM 属性：对集中式值函数最优的联合动作，对每个智能体的个体效用函数也是最优的。即：

```
arg max_a Q(h, z, a; θ) = (arg max_{a₁} Q(h₁, a₁; θ₁), ..., arg max_{aₙ} Q(hₙ, aₙ; θₙ))
```

### 单调性条件（Monotonicity Constraint）

QMIX 的关键创新是将 IGM 条件转化为**单调性约束**：

```
∀i ∈ I, ∀a ∈ A :  ∂Q(h, z, a; θ) / ∂Q(hᵢ, aᵢ; θᵢ) > 0
```

即集中式值函数关于每个智能体效用的偏导数必须为正。直觉上，任何一个智能体效用的增加，都必须导致集中式值函数的增加。

**单调性 ⊇ 线性性**：VDN 的线性分解是 QMIX 单调分解的特例，因此 QMIX 能表示的值函数空间是 VDN 的超集。

---

## 3. 网络架构

QMIX 包含三个核心组件：

### 3.1 个体效用网络（Individual Utility Networks）

- 每个智能体 i 拥有一个独立的深度 Q 网络（类似 DQN）
- 输入：智能体 i 的观测历史 hᵢ
- 输出：对所有可选动作 aᵢ 的效用值 Q(hᵢ, aᵢ; θᵢ)
- 在原始实现中，各智能体**共享参数**（θᵢ = θⱼ），并用独热编码的智能体 ID 区分不同智能体
- 通常使用**循环神经网络（RNN）**以处理部分可观测性

### 3.2 超网络（Hypernetwork）

- 接收集中式信息 z（如全局状态 s）作为输入
- 输出混合网络的参数 θₘᵢₓ
- 为保证正权重，超网络对权重矩阵输出使用**绝对值激活函数**

```
θₘᵢₓ = f_hyper(z; θ_hyper)
```

### 3.3 混合网络（Mixing Network）

- 接收所有智能体的效用值和超网络输出的参数
- 通过**正权重前馈网络**实现单调聚合
- 输出集中式动作值函数

```
Q(h, z, a; θ) = f_mix(Q(h₁, a₁; θ₁), ..., Q(hₙ, aₙ; θₙ); θₘᵢₓ)
```

---

## 4. 训练目标

QMIX 使用集中式训练、分散式执行（CTDE）范式，训练损失为：

```
L(θ) = (1/B) Σ_{B} [rᵗ + γ max_a Q(hᵗ⁺¹, zᵗ⁺¹, a; θ̄) - Q(hᵗ, zᵗ, aᵗ; θ)]²
```

- B 是从经验回放缓冲区采样的 mini-batch
- θ̄ 是目标网络参数（定期更新）
- 混合网络参数不直接优化，始终由超网络输出得到

---

## 5. 伪代码（Algorithm 22）

```
QMIX 算法伪代码：

1. 初始化 n 个效用网络，参数为 θ₁, ..., θₙ
2. 初始化 n 个目标网络，参数为 θ̄₁ = θ₁, ..., θ̄ₙ = θₙ
3. 初始化超网络，参数为 θ_hyper
4. 初始化共享经验回放缓冲区 D

5. 对每个时间步 t = 0, 1, 2, ...:
   6. 收集集中式信息 zᵗ 和各智能体观测 o₁ᵗ, ..., oₙᵗ
   7. 对每个智能体 i = 1, ..., n:
      8. 以概率 ε 随机选择动作 aᵢᵗ（探索）
      9. 否则选择 aᵢᵗ = arg max_{aᵢ} Q(hᵢᵗ, aᵢ; θᵢ)（利用）
   10. 执行联合动作；收集共享奖励 rᵗ，下一时刻集中式信息 zᵗ⁺¹ 和观测
   11. 将转移 (hᵗ, zᵗ, aᵗ, rᵗ, hᵗ⁺¹, zᵗ⁺¹) 存入共享经验回放缓冲区 D
   
   12. 从 D 中采样 B 个转移 (hᵏ, zᵏ, aᵏ, rᵏ, hᵏ⁺¹, zᵏ⁺¹)
   13. 如果 sᵏ⁺¹ 是终止状态：
       14.   目标值 yᵏ ← rᵏ
   15. 否则：
       16.   θ_mix^{k+1} ← f_hyper(zᵏ⁺¹; θ_hyper)
       17.   yᵏ ← rᵏ + γ · f_mix(max_{a₁'} Q(h₁ᵏ⁺¹, a₁'; θ̄₁), ...,
                                   max_{aₙ'} Q(hₙᵏ⁺¹, aₙ'; θ̄ₙ); θ_mix^{k+1})
   18. θ_mix^k ← f_hyper(zᵏ; θ_hyper)
   19. 计算当前值估计：
       Q(hᵏ, zᵏ, aᵏ; θ) ← f_mix(Q(h₁ᵏ, a₁ᵏ; θ₁), ..., Q(hₙᵏ, aₙᵏ; θₙ); θ_mix^k)
   20. 损失 L(θ) ← (1/B) Σ [yᵏ - Q(hᵏ, zᵏ, aᵏ; θ)]²
   21. 最小化 L(θ)，更新所有参数 θ
   22. 每隔固定步数，更新目标网络参数 θ̄ᵢ
```

---

## 6. PyTorch 实现代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

# =====================
# 1. 个体效用网络（Agent Utility Network）
# =====================
class AgentNetwork(nn.Module):
    """每个智能体的效用网络（使用 GRU 处理部分可观测性）"""
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, obs, hidden):
        """
        obs: (batch, obs_dim)
        hidden: (batch, hidden_dim)
        returns: q_values (batch, action_dim), new_hidden (batch, hidden_dim)
        """
        x = F.relu(self.fc1(obs))
        new_hidden = self.gru(x, hidden)
        q_values = self.fc2(new_hidden)
        return q_values, new_hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim)


# =====================
# 2. 超网络（Hypernetwork）
# =====================
class HyperNetwork(nn.Module):
    """接收全局状态，输出混合网络的参数"""
    def __init__(self, state_dim, n_agents, mixing_hidden_dim=32):
        super().__init__()
        self.n_agents = n_agents
        self.mixing_hidden_dim = mixing_hidden_dim
        
        # 为混合网络第一层生成权重 (n_agents -> mixing_hidden_dim)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_agents * mixing_hidden_dim)
        )
        # 为混合网络第一层生成偏置
        self.hyper_b1 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden_dim)
        )
        # 为混合网络第二层生成权重 (mixing_hidden_dim -> 1)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, mixing_hidden_dim)
        )
        # 为混合网络第二层生成偏置（可为负，无需约束）
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, state):
        """
        state: (batch, state_dim)
        returns: w1, b1, w2, b2（混合网络参数）
        """
        batch = state.shape[0]
        # 绝对值确保权重为正，满足单调性
        w1 = torch.abs(self.hyper_w1(state))  # (batch, n_agents * mixing_hidden_dim)
        w1 = w1.view(batch, self.n_agents, self.mixing_hidden_dim)
        b1 = self.hyper_b1(state).view(batch, 1, self.mixing_hidden_dim)
        
        w2 = torch.abs(self.hyper_w2(state))  # (batch, mixing_hidden_dim)
        w2 = w2.view(batch, self.mixing_hidden_dim, 1)
        b2 = self.hyper_b2(state).view(batch, 1, 1)
        
        return w1, b1, w2, b2


# =====================
# 3. 混合网络（Mixing Network）
# =====================
class MixingNetwork(nn.Module):
    """将个体效用聚合为集中式值函数"""
    def __init__(self):
        super().__init__()
    
    def forward(self, agent_qs, w1, b1, w2, b2):
        """
        agent_qs: (batch, n_agents) - 各智能体选择动作的效用值
        w1, b1, w2, b2: 超网络生成的混合网络参数
        returns: q_total (batch, 1) - 集中式动作值
        """
        batch = agent_qs.shape[0]
        x = agent_qs.unsqueeze(1)  # (batch, 1, n_agents)
        
        # 第一层：正权重线性变换 + ELU激活
        x = torch.bmm(x, w1) + b1  # (batch, 1, mixing_hidden_dim)
        x = F.elu(x)
        
        # 第二层：正权重线性变换（输出标量）
        x = torch.bmm(x, w2) + b2  # (batch, 1, 1)
        return x.squeeze(-1)  # (batch, 1)


# =====================
# 4. 经验回放缓冲区
# =====================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch
    
    def __len__(self):
        return len(self.buffer)


# =====================
# 5. QMIX 主体
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
        
        # 共享参数的智能体效用网络（参数共享）
        self.agent_net = AgentNetwork(obs_dim, action_dim, hidden_dim)
        self.target_agent_net = AgentNetwork(obs_dim, action_dim, hidden_dim)
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        
        # 超网络与混合网络
        self.hyper_net = HyperNetwork(state_dim, n_agents, mixing_hidden_dim)
        self.mixing_net = MixingNetwork()
        self.target_hyper_net = HyperNetwork(state_dim, n_agents, mixing_hidden_dim)
        self.target_hyper_net.load_state_dict(self.hyper_net.state_dict())
        
        # 优化器（联合优化所有参数）
        self.optimizer = torch.optim.Adam(
            list(self.agent_net.parameters()) + list(self.hyper_net.parameters()),
            lr=lr
        )
        
        # 经验回放缓冲区
        self.buffer = ReplayBuffer(buffer_capacity)
    
    def select_actions(self, obs_list, hidden_list, epsilon):
        """
        ε-贪心策略选择动作（分散式执行）
        obs_list: [obs_agent_0, obs_agent_1, ...], 每个 obs 形状 (obs_dim,)
        hidden_list: [hidden_0, hidden_1, ...], 每个 hidden 形状 (hidden_dim,)
        """
        actions = []
        new_hiddens = []
        with torch.no_grad():
            for i in range(self.n_agents):
                obs = torch.FloatTensor(obs_list[i]).unsqueeze(0)
                hidden = hidden_list[i]
                q_vals, new_hidden = self.agent_net(obs, hidden)
                
                if random.random() < epsilon:
                    action = random.randint(0, self.action_dim - 1)
                else:
                    action = q_vals.argmax(dim=-1).item()
                
                actions.append(action)
                new_hiddens.append(new_hidden)
        
        return actions, new_hiddens
    
    def store_transition(self, obs, state, actions, reward, next_obs, next_state, done):
        """存储转移到回放缓冲区（包含集中式状态信息）"""
        self.buffer.push((obs, state, actions, reward, next_obs, next_state, done))
    
    def train(self):
        """从回放缓冲区采样并更新网络"""
        if len(self.buffer) < self.batch_size:
            return None
        
        batch = self.buffer.sample(self.batch_size)
        obs_b, state_b, actions_b, reward_b, next_obs_b, next_state_b, done_b = zip(*batch)
        
        # 转换为张量
        # obs_b: (batch, n_agents, obs_dim)
        obs_t = torch.FloatTensor(np.array(obs_b))
        state_t = torch.FloatTensor(np.array(state_b))
        actions_t = torch.LongTensor(np.array(actions_b))  # (batch, n_agents)
        reward_t = torch.FloatTensor(np.array(reward_b)).unsqueeze(1)  # (batch, 1)
        next_obs_t = torch.FloatTensor(np.array(next_obs_b))
        next_state_t = torch.FloatTensor(np.array(next_state_b))
        done_t = torch.FloatTensor(np.array(done_b)).unsqueeze(1)  # (batch, 1)
        
        # ---- 计算当前 Q 值 ----
        agent_qs = []
        hidden = self.agent_net.init_hidden(self.batch_size)
        for i in range(self.n_agents):
            obs_i = obs_t[:, i, :]  # (batch, obs_dim)
            q_vals, hidden = self.agent_net(obs_i, hidden)
            # 选择实际执行的动作的 Q 值
            q_a = q_vals.gather(1, actions_t[:, i].unsqueeze(1))  # (batch, 1)
            agent_qs.append(q_a)
        
        agent_qs = torch.cat(agent_qs, dim=1)  # (batch, n_agents)
        
        # 通过超网络获取混合参数并混合
        w1, b1, w2, b2 = self.hyper_net(state_t)
        q_total = self.mixing_net(agent_qs, w1, b1, w2, b2)  # (batch, 1)
        
        # ---- 计算目标 Q 值 ----
        with torch.no_grad():
            target_agent_qs = []
            target_hidden = self.target_agent_net.init_hidden(self.batch_size)
            for i in range(self.n_agents):
                next_obs_i = next_obs_t[:, i, :]
                target_q_vals, target_hidden = self.target_agent_net(next_obs_i, target_hidden)
                max_q = target_q_vals.max(dim=1, keepdim=True)[0]  # (batch, 1)
                target_agent_qs.append(max_q)
            
            target_agent_qs = torch.cat(target_agent_qs, dim=1)  # (batch, n_agents)
            
            tw1, tb1, tw2, tb2 = self.target_hyper_net(next_state_t)
            q_total_target = self.mixing_net(target_agent_qs, tw1, tb1, tw2, tb2)
            
            # Bellman 目标
            y = reward_t + self.gamma * (1 - done_t) * q_total_target  # (batch, 1)
        
        # ---- 计算损失并更新 ----
        loss = F.mse_loss(q_total, y)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.agent_net.parameters()) + list(self.hyper_net.parameters()), 10.0
        )
        self.optimizer.step()
        
        # 定期更新目标网络
        self.train_step += 1
        if self.train_step % self.target_update_interval == 0:
            self.target_agent_net.load_state_dict(self.agent_net.state_dict())
            self.target_hyper_net.load_state_dict(self.hyper_net.state_dict())
        
        return loss.item()


# =====================
# 6. 训练示例（使用伪环境）
# =====================
def train_example():
    """演示 QMIX 训练流程"""
    # 超参数
    n_agents = 3
    obs_dim = 10
    action_dim = 5
    state_dim = 20
    n_episodes = 1000
    max_steps = 50
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.995

    qmix = QMIX(
        n_agents=n_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        state_dim=state_dim,
        hidden_dim=64,
        mixing_hidden_dim=32,
        lr=3e-4,
        gamma=0.99
    )

    epsilon = epsilon_start
    total_rewards = []

    for episode in range(n_episodes):
        # 初始化隐藏状态
        hiddens = [qmix.agent_net.init_hidden(1) for _ in range(n_agents)]
        
        # 随机生成初始观测和状态（实际应由环境提供）
        obs = [np.random.randn(obs_dim) for _ in range(n_agents)]
        state = np.random.randn(state_dim)
        
        episode_reward = 0.0
        
        for step in range(max_steps):
            # 选择动作
            actions, hiddens = qmix.select_actions(obs, hiddens, epsilon)
            
            # 环境交互（此处模拟）
            next_obs = [np.random.randn(obs_dim) for _ in range(n_agents)]
            next_state = np.random.randn(state_dim)
            reward = np.random.randn()  # 共享奖励
            done = (step == max_steps - 1)
            
            # 存储转移
            qmix.store_transition(obs, state, actions, reward, next_obs, next_state, done)
            
            # 训练
            loss = qmix.train()
            
            obs = next_obs
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        total_rewards.append(episode_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(total_rewards[-100:])
            print(f"Episode {episode+1}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")

    return total_rewards


if __name__ == "__main__":
    rewards = train_example()
    print("Training completed!")
```

---

## 7. 实验结论

### 7.1 矩阵博弈分析

| 场景 | VDN | QMIX |
|------|-----|------|
| **线性可分解博弈** | ✅ 准确学习 | ✅ 准确学习 |
| **单调可分解博弈** | ❌ 无法准确表示非线性 | ✅ 准确学习 |
| **两步随机博弈** | ❌ 低估最优策略，选择次优路径 | ✅ 准确学习，选择最优路径 |
| **Climbing 博弈** | ❌ 收敛到次优策略 (C,C) | ❌ 收敛到次优策略 (C,B) |

### 7.2 关卡觅食环境（Level-Based Foraging）

在复杂多智能体环境中，QMIX **显著优于** VDN 和 IDQN：
- 获得最高平均回报
- 学习速度更快
- 多次运行方差更小

### 7.3 局限性

- Climbing 博弈表明单调性约束**仍然可能过于严格**，无法表示所有集中式值函数
- 单调性是保证 IGM 属性的充分条件，但**非必要条件**
- 后续工作（如 QPLEX、QTRAN）提出了更宽松的分解条件

---

## 8. 重要实现细节（来自原始论文）

1. **参数共享**：各智能体效用网络共享参数（θᵢ = θⱼ），通过独热编码的智能体 ID 区分
2. **循环网络**：效用网络使用 GRU 处理部分可观测问题
3. **额外观测**：智能体观测包含上一时刻的动作
4. **情节式回放缓冲区**：存储和采样完整情节（而非单步转移）
5. **集中式信息存储**：回放缓冲区需要额外存储集中式信息 z（如全局状态）
6. **ε-贪心探索**：探索率 ε 从 1 线性衰减到 0.05

---

## 9. 方法对比

| 特性 | IDQN | VDN | QMIX |
|------|------|-----|------|
| 值分解 | 无 | 线性 | 单调非线性 |
| 集中式训练 | 否 | 是 | 是 |
| 分散式执行 | 是 | 是 | 是 |
| 全局状态利用 | 否 | 否 | 是（通过超网络） |
| 表达能力 | 低 | 中 | 高 |
| 计算复杂度 | 低 | 低 | 中 |
