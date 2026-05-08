"""
Демонстрация обученного агента (с визуализацией)
"""

import gymnasium as gym
from dqn import DQNAgent
import time


def play(agent, env, episodes=3, delay=0.02):
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\nЭпизод {episode + 1}/{episodes}")
        
        while True:
            action = agent.select_action(state, training=False)
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
    agent = DQNAgent(state_dim=4, action_dim=2, hidden_dim=128)
    agent.load('dqn_model.pth')
    
    print("Создание окружения с визуализацией...")
    env = gym.make('CartPole-v1', render_mode='human')
    
    print("\nДемонстрация игры:")
    play(agent, env, episodes=3, delay=0.02)