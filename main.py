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
import datetime

from dqn import DQNAgent
from src.visualization import plot_results
from src.train import train as train_agent
from src.evaluate import evaluate as evaluate_agent

try:
    import mlflow
    import mlflow.pytorch  # type: ignore
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


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
    parser.add_argument('--mlflow', action='store_true', default=True, help='Включить MLflow логгирование')
    parser.add_argument('--no-mlflow', action='store_true', help='Отключить MLflow логгирование')
    return parser.parse_args()


def log_params(args):
    mlflow.log_params({
        'episodes': args.episodes,
        'batch_size': args.batch_size,
        'hidden_dim': args.hidden_dim,
        'lr': args.lr,
        'gamma': args.gamma,
        'epsilon_decay': args.epsilon_decay,
        'epsilon_min': args.epsilon_min,
        'target_update': args.target_update,
        'buffer_capacity': args.buffer_capacity,
        'seed': args.seed,
        'warmup_steps': args.warmup_steps,
    })


def log_metrics(rewards, losses, q_values, results):
    mlflow.log_metric('train_mean_reward', np.mean(rewards))
    mlflow.log_metric('train_max_reward', max(rewards))
    mlflow.log_metric('train_min_reward', min(rewards))
    
    if losses:
        mlflow.log_metric('train_mean_loss', np.mean(losses))
    
    if q_values:
        mlflow.log_metric('train_mean_q', np.mean(q_values))
    
    mlflow.log_metric('eval_mean_reward', results['mean_reward'])
    mlflow.log_metric('eval_max_reward', results['max_reward'])
    mlflow.log_metric('eval_min_reward', results['min_reward'])
    mlflow.log_metric('eval_std_reward', results['std_reward'])


def main():
    args = parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    use_mlflow = args.mlflow and not args.no_mlflow and MLFLOW_AVAILABLE
    
    if use_mlflow:
        mlflow.set_experiment('DQN_CartPole_Experiment')
        mlflow.start_run(run_name=f"dqn_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}")
        log_params(args)
        print(f"{'='*50}")
        run = mlflow.active_run()
        print(f"MLflow включен: {run.info.run_id if run else 'N/A'}")
        print(f"{'='*50}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    if use_mlflow:
        mlflow.log_param('device', str(device))
    
    env = gym.make('CartPole-v1')
    obs_space = env.observation_space
    action_space = env.action_space
    obs_shape = obs_space.shape
    action_dim = int(action_space.n)  # type: ignore
    print(f"Состояние: {obs_shape}, Действий: {action_dim}")
    
    agent = DQNAgent(
        state_dim=int(obs_shape[0]),
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        target_update=args.target_update,
        buffer_capacity=args.buffer_capacity,
        device=device
    )
    
    print("Warmup: заполнение буфера...")
    for _ in range(10):
        state, _ = env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, float(reward), next_state, done)
            state = next_state
            if done:
                break
    print(f"Буфер: {len(agent.buffer)} переходов")
    
    print(f"\nПараметры: episodes={args.episodes}, batch={args.batch_size}, lr={args.lr}")
    
    start_time = datetime.datetime.now()
    
    rewards, losses, q_values = train_agent(
        env=env,
        agent=agent,
        episodes=args.episodes,
        batch_size=args.batch_size,
        seed=args.seed,
        warmup_steps=args.warmup_steps,
        mlflow_logging=use_mlflow
    )
    
    training_time = (datetime.datetime.now() - start_time).total_seconds()
    print(f"\nВремя обучения: {training_time:.1f} сек")
    
    if use_mlflow:
        mlflow.log_metric('training_time', training_time)
    
    agent.save(args.model_path)
    print(f"Модель сохранена: {args.model_path}")
    
    if use_mlflow:
        mlflow.pytorch.log_model(agent.q_network, artifact_path='q_network')  # type: ignore
    
    plot_results(rewards, losses, q_values, save_path='plots')
    
    if use_mlflow:
        mlflow.log_artifact('plots/training_analysis.png')
        mlflow.log_artifact('plots/training_progress.png')
    
    print("\nОценка агента...")
    results = evaluate_agent(env, agent, episodes=args.eval_episodes)
    print(f"Средняя награда: {results['mean_reward']:.1f}")
    print(f"Максимум: {results['max_reward']}")
    print(f"Минимум: {results['min_reward']}")
    
    if use_mlflow:
        log_metrics(rewards, losses, q_values, results)
        mlflow.end_run()
        print("\nMLflow логирование завершено")
    
    env.close()
    print("\nГотово!")


if __name__ == '__main__':
    main()