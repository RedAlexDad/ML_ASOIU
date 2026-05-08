"""
LR5: DQN (Deep Q-Network) - Обучение с подкреплением

Задача: CartPole-v1 (балансировка шеста)

Вариант 2

Студент: Папин А.В., ИУ5Ц-21М
"""

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from dqn import DQNAgent
import os


# Гиперпараметры
EPISODES = 500
BATCH_SIZE = 64
HIDDEN_DIM = 128
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
TARGET_UPDATE = 100
BUFFER_CAPACITY = 10000
SEED = 42

np.random.seed(SEED)


def train(env, agent, episodes=EPISODES):
    rewards_history = []
    losses = []
    
    for episode in range(episodes):
        state, _ = env.reset(seed=SEED + episode)
        total_reward = 0
        episode_losses = []
        
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, next_state, done)
            
            loss = agent.train_step(BATCH_SIZE)
            if loss is not None:
                episode_losses.append(loss)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        rewards_history.append(total_reward)
        if episode_losses:
            losses.append(np.mean(episode_losses))
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(f"Episode {episode+1}/{episodes} | Avg Reward: {avg_reward:.1f} | Epsilon: {agent.epsilon:.3f}")
    
    return rewards_history, losses


def evaluate(env, agent, episodes=10):
    eval_rewards = []
    
    for episode in range(episodes):
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
        
        eval_rewards.append(total_reward)
    
    return eval_rewards


def plot_results(rewards, losses, save_path='plots'):
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward per Episode')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/training_results.png', dpi=150)
    plt.close()
    
    plt.figure(figsize=(8, 5))
    window = 50
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(smoothed)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title(f'Smoothed Reward (window={window})')
    plt.grid(True)
    plt.savefig(f'{save_path}/smoothed_reward.png', dpi=150)
    plt.close()
    
    print(f"Графики сохранены в {save_path}/")


def main():
    env = gym.make('CartPole-v1')
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=HIDDEN_DIM,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        epsilon_decay=EPSILON_DECAY,
        epsilon_min=EPSILON_MIN,
        target_update=TARGET_UPDATE,
        buffer_capacity=BUFFER_CAPACITY
    )
    
    print("Начало обучения...")
    rewards, losses = train(env, agent)
    
    agent.save('dqn_model.pth')
    print("Модель сохранена: dqn_model.pth")
    
    plot_results(rewards, losses)
    
    print("\nОценка обученного агента...")
    eval_rewards = evaluate(env, agent)
    print(f"Средняя награда на тесте: {np.mean(eval_rewards):.1f}")
    print(f"Максимальная награда: {max(eval_rewards)}")
    
    env.close()


if __name__ == '__main__':
    main()