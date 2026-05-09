"""
Демонстрация обученного агента (с визуализацией)
"""

import gymnasium as gym
from dqn import DQNAgent
import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser(description='DQN Demo - визуализация агента')
    parser.add_argument('--model', type=str, default='dqn_model.pth', help='Путь к модели')
    parser.add_argument('--episodes', type=int, default=3, help='Количество эпизодов')
    parser.add_argument('--delay', type=float, default=0.02, help='Задержка между шагами (сек)')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Размер скрытого слоя')
    return parser.parse_args()


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
    args = parse_args()
    
    print(f"Загрузка модели: {args.model}")
    agent = DQNAgent(state_dim=4, action_dim=2, hidden_dim=args.hidden_dim)
    agent.load(args.model)
    
    print("Создание окружения с визуализацией...")
    env = gym.make('CartPole-v1', render_mode='human')
    
    print(f"\nДемонстрация игры ({args.episodes} эпизодов):")
    play(agent, env, episodes=args.episodes, delay=args.delay)