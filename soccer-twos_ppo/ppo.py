import numpy as np
from torch.distributions import Normal
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F

# 代码用于离散环境的模型

# ----------------------------------- #
# 构建策略网络--actor
# ----------------------------------- #
 
class PolicyNet(nn.Module):
    def __init__(self, state_size, n_hiddens, action_size):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
 
# ----------------------------------- #
# 构建价值网络--critic
# ----------------------------------- #
 
class ValueNet(nn.Module):
    def __init__(self, n_states, n_hiddens):
        super(ValueNet, self).__init__()
        self.fc_val = nn.Sequential(
            nn.Linear(n_states, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        x = self.fc_val(x) 
        return x
 
# ----------------------------------- #
# 构建模型
# ----------------------------------- #
 
class PPO:
    def __init__(self, n_states, n_hiddens, n_actions,
                 actor_lr, critic_lr, lmbda, epochs, eps, gamma, device):
        # 实例化策略网络
        self.actor = PolicyNet(n_states, n_hiddens, n_actions).to(device)
        # 实例化价值网络
        self.critic = ValueNet(n_states, n_hiddens).to(device)
        # 策略网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr,  weight_decay=0.01)
        # 价值网络的优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = critic_lr,  weight_decay=0.01)
 
        self.gamma = gamma  # 折扣因子
        self.lmbda = lmbda  # GAE优势函数的缩放系数
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device
 
    # 动作选择
    def take_action(self, state):
        # 维度变换 [n_state]-->tensor[1,n_states]
        # state = torch.tensor(state[np.newaxis, :]).to(self.device)
        states = torch.FloatTensor(np.array([state])).to(self.device)
        # 当前状态下，每个动作的概率分布 [1,n_states]
        probs = self.actor(states)
        # 创建以probs为标准的概率分布
        action_list = torch.distributions.Categorical(probs)
        # 依据其概率随机挑选一个动作
        action = action_list.sample().item()
        return action
 
    # 训练
    def learn(self, transition_dict):
        # 提取数据集
        states = torch.tensor(np.array(transition_dict['states']).reshape(-1,294), dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).to(self.device).view(-1,1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device).view(-1,1)
        # rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']).reshape(-1,294), dtype=torch.float).to(self.device)
        # next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(self.device).view(-1,1)
 
        # 目标，下一个状态的state_value  [b,1]
        next_q_target = self.critic(next_states)
        # 目标，当前状态的state_value  [b,1]
        td_target = rewards + self.gamma * next_q_target * (1-dones)
        # 预测，当前状态的state_value  [b,1]
        td_value = self.critic(states)
        # 目标值和预测值state_value之差  [b,1]
        td_delta = td_target - td_value
 
        # 时序差分值 tensor-->numpy  [b,1]
        td_delta = td_delta.cpu().detach().numpy()
        advantage = 0  # 优势函数初始化
        advantage_list = []
 
        # 计算优势函数
        for delta in td_delta[::-1]:  # 逆序时序差分值 axis=1轴上倒着取 [], [], []
            # 优势函数GAE的公式
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        # 正序
        advantage_list.reverse()
        # numpy --> tensor [b,1]
        advantage = torch.tensor(advantage_list, dtype=torch.float).to(self.device)
 
        # 策略网络给出每个动作的概率，根据action得到当前时刻下该动作的概率
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
 
        # 一组数据训练 epochs 轮
        for _ in range(self.epochs):
            # 每一轮更新一次策略网络预测的状态
            log_probs = torch.log(self.actor(states).gather(1, actions))
            # 新旧策略之间的比例
            ratio = torch.exp(log_probs - old_log_probs)
            # 近端策略优化裁剪目标函数公式的左侧项
            surr1 = ratio * advantage
            # 公式的右侧项，ratio小于1-eps就输出1-eps，大于1+eps就输出1+eps
            surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage
 
            # 策略网络的损失函数
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            # 价值网络的损失函数，当前时刻的state_value - 下一时刻的state_value
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
 
            # 梯度清0
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            # 反向传播
            actor_loss.backward()
            critic_loss.backward()
            # 梯度更新
            self.actor_optimizer.step()
            self.critic_optimizer.step()