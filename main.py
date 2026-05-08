"""
LR5: DQN (Deep Q-Network) - Обучение с подкреплением
Задача: CartPole-v1 (балансировка шеста)

Вариант 2
Студент: Папин А.В., ИУ5Ц-21М
"""

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import os
import datetime
import torch
import torch.nn as nn

import mlflow
import mlflow.pytorch
from dqn import DQNAgent


os.environ['MLFLOW_TRACKING_URI'] = './mlruns'
os.environ['MLFLOW_EXPERIMENT_NAME'] = 'DQN_CartPole'


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
torch.manual_seed(SEED)


def get_env_info(env: gym.Env) -> dict:
    return {
        "state_space": env.observation_space.shape,
        "action_space": env.action_space.n,
        "state_high": env.observation_space.high.tolist(),
        "state_low": env.observation_space.low.tolist(),
    }


def train(env: gym.Env, agent: DQNAgent, episodes: int = EPISODES) -> tuple:
    rewards_history = []
    losses = []
    q_values_history = []
    
    start_time = datetime.datetime.now()
    
    for episode in range(episodes):
        state, _ = env.reset(seed=SEED + episode)
        total_reward = 0
        episode_losses = []
        episode_q_values = []
        
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, next_state, done)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_vals = agent.q_network(state_tensor).numpy()[0]
                episode_q_values.append(q_vals.max())
            
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
        if episode_q_values:
            q_values_history.append(np.mean(episode_q_values))
        
        avg_reward = np.mean(rewards_history[-10:]) if len(rewards_history) >= 10 else total_reward
        current_loss = np.mean(episode_losses) if episode_losses else 0
        current_q = np.mean(episode_q_values) if episode_q_values else 0
        
        agent.log_metrics(
            episode=episode,
            reward=total_reward,
            loss=current_loss,
            epsilon=agent.epsilon,
            q_values_mean=current_q
        )
        
        if (episode + 1) % 50 == 0:
            mlflow.log_metric("avg_reward_50", avg_reward, step=episode)
            print(f"Episode {episode+1}/{episodes} | Avg Reward: {avg_reward:.1f} | Epsilon: {agent.epsilon:.3f}")
    
    total_time = (datetime.datetime.now() - start_time).total_seconds()
    mlflow.log_metric("total_training_time", total_time)
    mlflow.log_metric("final_avg_reward", np.mean(rewards_history[-50:]))
    mlflow.log_metric("max_reward", max(rewards_history))
    mlflow.log_metric("min_reward", min(rewards_history))
    
    return rewards_history, losses, q_values_history


def evaluate(env: gym.Env, agent: DQNAgent, episodes: int = 20) -> dict:
    eval_rewards = []
    eval_q_values = []
    episode_lengths = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        episode_q = []
        
        while True:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_vals = agent.q_network(state_tensor).numpy()[0]
                episode_q.append(q_vals.max())
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        eval_rewards.append(total_reward)
        episode_lengths.append(steps)
        eval_q_values.append(np.mean(episode_q))
    
    results = {
        "mean_reward": np.mean(eval_rewards),
        "std_reward": np.std(eval_rewards),
        "max_reward": max(eval_rewards),
        "min_reward": min(eval_rewards),
        "mean_q_value": np.mean(eval_q_values),
        "mean_episode_length": np.mean(episode_lengths),
    }
    
    for key, value in results.items():
        mlflow.log_metric(f"eval_{key}", value)
    
    return results


def plot_results(rewards: list, losses: list, q_values: list, save_path: str = 'plots') -> None:
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(rewards, alpha=0.6, label='Raw')
    window = 20
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    axes[0, 0].plot(range(window-1, len(rewards)), smoothed, 'r-', label=f'Smoothed (w={window})')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Total Reward per Episode')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(losses)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training Loss')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(q_values)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Mean Q-Value')
    axes[1, 0].set_title('Q-Values during Training')
    axes[1, 0].grid(True)
    
    axes[1, 1].hist(rewards, bins=30, edgecolor='black')
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/training_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.fill_between(range(len(rewards)), 
                     np.array(rewards) - np.std(rewards[-50:]), 
                     np.array(rewards) + np.std(rewards[-50:]), 
                     alpha=0.3)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Training Progress with Confidence Interval')
    plt.grid(True)
    plt.savefig(f'{save_path}/training_progress.png', dpi=150)
    plt.close()
    
    print(f"Графики сохранены в {save_path}/")


def main():
    mlflow.set_experiment("DQN_CartPole_Experiment")
    
    with mlflow.start_run(run_name=f"DQN_CartPole_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        env = gym.make('CartPole-v1')
        
        env_info = get_env_info(env)
        mlflow.log_params(env_info)
        
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
            buffer_capacity=BUFFER_CAPACITY,
            experiment_name="DQN_CartPole"
        )
        
        print("Начало обучения...")
        rewards, losses, q_values = train(env, agent)
        
        agent.save('dqn_model.pth')
        agent.log_model(artifact_path="dqn_model")
        print("Модель сохранена: dqn_model.pth")
        
        plot_results(rewards, losses, q_values)
        
        print("\nОценка обученного агента...")
        eval_results = evaluate(env, agent)
        
        print("\n" + "="*50)
        print("РЕЗУЛЬТАТЫ ОЦЕНКИ:")
        print("="*50)
        for key, value in eval_results.items():
            print(f"  {key}: {value:.2f}")
        
        mlflow.log_params({
            "eval_mean_reward": eval_results["mean_reward"],
            "eval_max_reward": eval_results["max_reward"],
            "episodes_trained": EPISODES,
        })
        
        env.close()
        
        print("\n" + "="*50)
        print("MLflow tracking запущен!")
        print("Запусти: mlflow ui --backend-store-uri ./mlruns")
        print("="*50)


if __name__ == '__main__':
    main()