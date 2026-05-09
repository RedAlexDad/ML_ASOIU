"""
LR5: DQN (Deep Q-Network) - Обучение с подкреплением
Задача: CartPole-v1 (балансировка шеста)

Вариант 2
Студент: Папин А.В., ИУ5Ц-21М
"""

import numpy as np
import torch
import gymnasium as gym
import argparse
import os

from dqn import DQNAgent
from src.visualization import plot_results
from src.train import train as train_agent
from src.evaluate import evaluate as evaluate_agent


def parse_args():
    parser = argparse.ArgumentParser(description='DQN для CartPole-v1')
    parser.add_argument('--episodes', type=int, default=50, help='Количество эпизодов')
    parser.add_argument('--batch-size', type=int, default=32, help='Размер батча')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Размер скрытого слоя')
    parser.add_argument('--lr', type=float, default=0.003, help='Скорость обучения')
    parser.add_argument('--gamma', type=float, default=0.99, help='Коэффициент дисконтирования')
    parser.add_argument('--epsilon-decay', type=float, default=0.98, help='Затухание epsilon')
    parser.add_argument('--epsilon-min', type=float, default=0.01, help='Минимальный epsilon')
    parser.add_argument('--target-update', type=int, default=50, help='Частота обновления целевой сети')
    parser.add_argument('--buffer-capacity', type=int, default=10000, help='Размер буфера')
    parser.add_argument('--seed', type=int, default=42, help='Seed для воспроизводимости')
    parser.add_argument('--warmup-steps', type=int, default=1000, help='Шаги warmup')
    parser.add_argument('--eval-episodes', type=int, default=10, help='Эпизоды для оценки')
    parser.add_argument('--model-path', type=str, default='dqn_model.pth', help='Путь для сохранения модели')
    return parser.parse_args()


def main():
    args = parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    env = gym.make('CartPole-v1')
    print(f"Состояние: {env.observation_space.shape}, Действий: {env.action_space.n}")
    
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        target_update=args.target_update,
        buffer_capacity=args.buffer_capacity
    )
    
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
    
    print(f"\nПараметры: episodes={args.episodes}, batch={args.batch_size}, lr={args.lr}")
    
    rewards, losses, q_values = train_agent(
        env=env,
        agent=agent,
        episodes=args.episodes,
        batch_size=args.batch_size,
        seed=args.seed,
        warmup_steps=args.warmup_steps
    )
    
    agent.save(args.model_path)
    print(f"\nМодель сохранена: {args.model_path}")
    
    plot_results(rewards, losses, q_values, save_path='plots')
    
    print("\nОценка агента...")
    results = evaluate_agent(env, agent, episodes=args.eval_episodes)
    print(f"Средняя награда: {results['mean_reward']:.1f}")
    print(f"Максимум: {results['max_reward']}")
    print(f"Минимум: {results['min_reward']}")
    
    env.close()
    print("\nГотово!")


if __name__ == '__main__':
    main()