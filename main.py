"""
LR5: DQN (Deep Q-Network) - Обучение с подкреплением
Задача: CartPole-v1 (балансировка шеста)

Вариант 2
Студент: Папин А.В., ИУ5Ц-21М
"""

import numpy as np
import torch
import gymnasium as gym
import os

from dqn import DQNAgent
from src.visualization import plot_results
from src.train import train as train_agent
from src.evaluate import evaluate as evaluate_agent

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


# ==================== MAIN ====================
def main():
    env = gym.make('CartPole-v1')
    print(f"Состояние: {env.observation_space.shape}, Действий: {env.action_space.n}")
    
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=HIDDEN_DIM,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        epsilon_decay=EPSILON_DECAY,
        epsilon_min=EPSILON_MIN,
        target_update=TARGET_UPDATE,
        buffer_capacity=BUFFER_CAPACITY
    )
    
    # Warmup
    print("Warmup: заполнение буфера...")
    for _ in range(10):
        state, _ = env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
    print(f"Буфер: {len(agent.buffer)} переходов")
    
    rewards, losses, q_values = train_agent(
        env=env,
        agent=agent,
        episodes=EPISODES,
        batch_size=BATCH_SIZE,
        seed=SEED,
        warmup_steps=1000
    )
    
    agent.save('dqn_model.pth')
    print("\nМодель сохранена: dqn_model.pth")
    
    plot_results(rewards, losses, q_values, save_path='plots')
    
    print("\nОценка агента...")
    results = evaluate_agent(env, agent, episodes=10)
    print(f"Средняя награда: {results['mean_reward']:.1f}")
    print(f"Максимум: {results['max_reward']}")
    print(f"Минимум: {results['min_reward']}")
    
    env.close()
    print("\nГотово!")


if __name__ == '__main__':
    main()