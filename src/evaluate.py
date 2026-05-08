import numpy as np
import torch
import mlflow
import gymnasium as gym
from typing import Dict
from dqn.agent import DQNAgent


def evaluate(env: gym.Env, agent: DQNAgent, episodes: int = 20) -> Dict[str, float]:
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