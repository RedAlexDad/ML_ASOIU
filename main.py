"""
LR5: DQN (Deep Q-Network) - Обучение с подкреплением
Задача: CartPole-v1 (балансировка шеста)

Вариант 2
Студент: Папин А.В., ИУ5Ц-21М
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import gymnasium as gym
import matplotlib.pyplot as plt
import os


# ==================== CONFIG ====================
EPISODES = 50
BATCH_SIZE = 32
HIDDEN_DIM = 128
LEARNING_RATE = 0.003
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.98
EPSILON_MIN = 0.01
TARGET_UPDATE = 50
BUFFER_CAPACITY = 10000
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


# ==================== Q-NETWORK ====================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ==================== REPLAY BUFFER ====================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


# ==================== DQN AGENT ====================
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN
        self.action_dim = action_dim
        self.steps = 0

        self.q_net = QNetwork(state_dim, action_dim, HIDDEN_DIM)
        self.target_net = QNetwork(state_dim, action_dim, HIDDEN_DIM)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        self.buffer = ReplayBuffer(BUFFER_CAPACITY)

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            return self.q_net(torch.FloatTensor(state).unsqueeze(0)).argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self, batch_size):
        if len(self.buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)


# ==================== TRAIN ====================
def train(env, agent, episodes):
    rewards_history = []
    losses_history = []
    
    # Warmup
    print("Warmup: заполнение буфера...")
    for _ in range(10):
        state, _ = env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
    print(f"Буфер: {len(agent.buffer)} переходов")

    print(f"\nОбучение: {episodes} эпизодов")
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_loss = []
        
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store(state, action, reward, next_state, done)
            
            loss = agent.train_step(BATCH_SIZE)
            if loss:
                episode_loss.append(loss)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        rewards_history.append(total_reward)
        if episode_loss:
            losses_history.append(np.mean(episode_loss))
        
        if (episode + 1) % 20 == 0:
            avg = np.mean(rewards_history[-20:])
            print(f"Эпизод {episode+1}/{episodes} | Награда: {total_reward:.1f} | Средняя(20): {avg:.1f} | Epsilon: {agent.epsilon:.3f}")
    
    return rewards_history, losses_history


# ==================== EVALUATE ====================
def evaluate(env, agent, episodes=10):
    rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        while True:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state
            if done:
                break
        rewards.append(total_reward)
    return rewards


# ==================== VISUALIZATION ====================
def plot_results(rewards, losses):
    os.makedirs('plots', exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Reward
    axes[0].plot(rewards)
    window = 20
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(rewards)), smoothed, 'r-', label='Сглаженная')
    axes[0].set_xlabel('Эпизод')
    axes[0].set_ylabel('Награда')
    axes[0].set_title('Награда за эпизод')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(losses)
    axes[1].set_xlabel('Эпизод')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Потери при обучении')
    axes[1].grid(True)
    
    # Distribution
    axes[2].hist(rewards, bins=20, edgecolor='black')
    axes[2].set_xlabel('Награда')
    axes[2].set_ylabel('Частота')
    axes[2].set_title('Распределение наград')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/training_results.png', dpi=150)
    plt.close()
    print("Графики сохранены в plots/")


# ==================== MAIN ====================
def main():
    env = gym.make('CartPole-v1')
    print(f"Состояние: {env.observation_space.shape}, Действий: {env.action_space.n}")
    
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )
    
    rewards, losses = train(env, agent, EPISODES)
    
    agent.save('dqn_model.pth')
    print("\nМодель сохранена: dqn_model.pth")
    
    plot_results(rewards, losses)
    
    print("\nОценка агента...")
    eval_rewards = evaluate(env, agent)
    print(f"Средняя награда: {np.mean(eval_rewards):.1f}")
    print(f"Максимум: {max(eval_rewards)}")
    
    env.close()
    print("\nГотово!")


if __name__ == '__main__':
    main()