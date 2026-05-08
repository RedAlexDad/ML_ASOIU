import numpy as np
import torch
import datetime
import mlflow
from dqn.agent import DQNAgent
import gymnasium as gym
from typing import Tuple, List


def train(
    env: gym.Env,
    agent: DQNAgent,
    episodes: int,
    batch_size: int,
    seed: int,
    warmup_steps: int = 200
) -> Tuple[List[float], List[float], List[float]]:
    rewards_history = []
    losses = []
    q_values_history = []
    
    start_time = datetime.datetime.now()
    
    print(f"Начало обучения: {episodes} эпизодов, warmup={warmup_steps} шагов")
    
    total_steps = 0
    
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0
        episode_losses = []
        episode_q_values = []
        steps = 0
        max_steps = 500
        
        while steps < max_steps:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, next_state, done)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_vals = agent.q_network(state_tensor).numpy()[0]
                episode_q_values.append(q_vals.max())
            
            # Обучаем после warmup или когда достаточно данных
            if total_steps >= warmup_steps or len(agent.buffer) >= batch_size:
                loss = agent.train_step(batch_size)
                if loss is not None:
                    episode_losses.append(loss)
            
            total_reward += reward
            state = next_state
            steps += 1
            total_steps += 1
            
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
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{episodes} | Reward: {total_reward:.1f} | Steps: {steps} | Epsilon: {agent.epsilon:.3f}")
    
    total_time = (datetime.datetime.now() - start_time).total_seconds()
    mlflow.log_metric("total_training_time", total_time)
    mlflow.log_metric("final_avg_reward", np.mean(rewards_history[-50:]))
    mlflow.log_metric("max_reward", max(rewards_history))
    mlflow.log_metric("min_reward", min(rewards_history))
    
    return rewards_history, losses, q_values_history