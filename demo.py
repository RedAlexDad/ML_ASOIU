"""
Демонстрация обученного агента (с визуализацией)
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from dqn import DQNAgent


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


def load_agent(path='dqn_model.pth', state_dim=4, action_dim=2):
    agent = QNetwork(state_dim, action_dim)
    agent.load_state_dict(torch.load(path))
    agent.eval()
    return agent


def play(agent, env, episodes=3, delay=0.02):
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\nЭпизод {episode + 1}/{episodes}")
        
        while True:
            with torch.no_grad():
                action = agent(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            state = next_state
            
            env.render()
            time.sleep(delay)
            
            if done:
                break
        
        print(f"  Награда: {total_reward}, Шагов: {steps}")
        if total_reward >= 500:
            print("  🎉 МАКСИМАЛЬНАЯ НАГРАДА!")
    
    env.close()
    print("\nГотово!")


if __name__ == '__main__':
    print("Загрузка модели...")
    agent = load_agent()
    
    print("Создание окружения с визуализацией...")
    env = gym.make('CartPole-v1', render_mode='human')
    
    print("\nДемонстрация игры:")
    play(agent, env, episodes=3, delay=0.02)