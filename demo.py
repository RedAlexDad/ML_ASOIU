"""
Демонстрация обученного агента (с визуализацией)
"""

import gymnasium as gym
from dqn import DQNAgent
from dqn.network import QNetwork, DuelingQNetwork, QNetworkBN, QNetworkLSTM
import argparse
import time
import torch
import mlflow


def parse_args():
    parser = argparse.ArgumentParser(description='DQN Demo - визуализация агента')
    parser.add_argument('--model', type=str, default='dqn_model.pth', help='Путь к модели')
    parser.add_argument('--network', type=str, default='qnetwork',
                        choices=['qnetwork', 'dueling', 'bn', 'lstm'],
                        help='Тип архитектуры сети')
    parser.add_argument('--episodes', type=int, default=3, help='Количество эпизодов')
    parser.add_argument('--delay', type=float, default=0.02, help='Задержка между шагами (сек)')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Размер скрытого слоя')
    return parser.parse_args()


def create_network(network_type: str, state_dim: int, action_dim: int, hidden_dim: int):
    """Создать сеть нужного типа."""
    if network_type == 'lstm':
        return QNetworkLSTM(state_dim, action_dim, hidden_dim)
    elif network_type == 'dueling':
        return DuelingQNetwork(state_dim, action_dim, hidden_dim)
    elif network_type == 'bn':
        return QNetworkBN(state_dim, action_dim, hidden_dim)
    else:
        return QNetwork(state_dim, action_dim, hidden_dim)


def detect_network_type(path: str) -> str:
    """Определить тип сети по структуре весов."""
    import os
    import torch
    
    model_path = path
    if 'mlruns' in path or 'mlflow' in path.lower():
        if path.endswith('model.pth'):
            model_path = path
        else:
            model_path = os.path.join(path, 'data', 'model.pth')
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if isinstance(state_dict, dict):
            if 'q_network' in state_dict:
                state_dict = state_dict['q_network']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
        elif hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()
        
        keys = list(state_dict.keys()) if hasattr(state_dict, 'keys') else []
        
        if any('lstm' in k for k in keys):
            return 'lstm'
        elif any('fc_val' in k or 'fc_adv' in k for k in keys):
            return 'dueling'
        elif any('bn' in k or 'batch_norm' in k for k in keys):
            return 'bn'
    
    return 'qnetwork'


def load_model_for_demo(path: str, hidden_dim: int, network_type: str) -> DQNAgent:
    """Загрузить модель - поддерживает как локальные файлы, так и MLflow."""
    state_dim, action_dim = 4, 2
    
    if network_type == 'qnetwork' and ('mlruns' in path or 'mlflow' in path.lower()):
        detected_type = detect_network_type(path)
        if detected_type != 'qnetwork':
            print(f"  Автоопределение типа сети: {detected_type}")
            network_type = detected_type
    
    q_network = create_network(network_type, state_dim, action_dim, hidden_dim)
    target_network = create_network(network_type, state_dim, action_dim, hidden_dim)
    
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    agent.q_network = q_network
    agent.target_network = target_network
    
    optimizer = torch.optim.Adam(agent.q_network.parameters(), lr=0.003)
    agent.optimizer = optimizer
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent.device = device
    
    if 'mlruns' in path or 'mlflow' in path.lower():
        print("  Загрузка через MLflow...")
        mlflow_path = path
        if path.endswith('model.pth'):
            mlflow_path = path.replace('/artifacts/data/model.pth', '/artifacts')
        print(f"  MLflow path: {mlflow_path}")
        model = mlflow.pytorch.load_model(mlflow_path, map_location=device)
        model.to(device)
        model.eval()
        
        if isinstance(model, torch.nn.Module):
            try:
                agent.q_network.load_state_dict(model.state_dict())
                agent.target_network.load_state_dict(model.state_dict())
                agent.q_network.to(device)
                agent.target_network.to(device)
            except RuntimeError as e:
                detected_type = detect_network_type(path)
                if detected_type != network_type:
                    print(f"  Автоопределение типа сети: {detected_type}")
                    network_type = detected_type
                    q_network = create_network(network_type, state_dim, action_dim, hidden_dim)
                    target_network = create_network(network_type, state_dim, action_dim, hidden_dim)
                    agent.q_network = q_network
                    agent.target_network = target_network
                    agent.q_network.to(device)
                    agent.target_network.to(device)
                    agent.q_network.load_state_dict(model.state_dict())
                    agent.target_network.load_state_dict(model.state_dict())
                    agent.q_network.to(device)
                    agent.target_network.to(device)
                else:
                    raise
    else:
        agent.load(path)
    
    return agent


def play(agent, env, episodes=3, delay=0.02):
    agent.q_network.eval()
    
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
    print(f"Тип сети: {args.network}")
    agent = load_model_for_demo(args.model, args.hidden_dim, args.network)
    
    print("Создание окружения с визуализацией...")
    env = gym.make('CartPole-v1', render_mode='human')
    
    print(f"\nДемонстрация игры ({args.episodes} эпизодов):")
    play(agent, env, episodes=args.episodes, delay=args.delay)