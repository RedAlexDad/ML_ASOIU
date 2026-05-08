import numpy as np
import torch
import gymnasium as gym
from dqn import DQNAgent
import mlflow

np.random.seed(42)
torch.manual_seed(42)

env = gym.make('CartPole-v1')

agent = DQNAgent(
    state_dim=4, 
    action_dim=2, 
    hidden_dim=128, 
    lr=0.003, 
    gamma=0.99, 
    epsilon_decay=0.98,
    epsilon_min=0.01,
    target_update=50
)

print("Warmup: 500 шагов")
for _ in range(10):
    state, _ = env.reset()
    for _ in range(50):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.store_transition(state, action, reward, next_state, done)
        state = next_state
        if done:
            break

print(f"Buffer: {len(agent.buffer)}")

print("Обучение: 100 эпизодов")
for episode in range(100):
    state, _ = env.reset()
    total_reward = 0
    
    while total_reward < 500:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        agent.store_transition(state, action, reward, next_state, done)
        
        if len(agent.buffer) >= 32:
            agent.train_step(32)
        
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    if (episode + 1) % 20 == 0:
        print(f"Episode {episode+1}: reward={total_reward}, epsilon={agent.epsilon:.3f}")

print("\nОценка:")
eval_rewards = []
for _ in range(5):
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

print(f"Среднее: {np.mean(eval_rewards):.1f}, Макс: {max(eval_rewards)}")