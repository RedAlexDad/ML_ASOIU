"""
LR5: DQN (Deep Q-Network) - Обучение с подкреплением
Задача: CartPole-v1 (балансировка шеста)

Вариант 2
Студент: Папин А.В., ИУ5Ц-21М
"""

import numpy as np
import torch
import datetime

import mlflow
import mlflow.pytorch
from dqn import DQNAgent
from src import get_env_info, make_env, train, evaluate, plot_results


EPISODES = 300
BATCH_SIZE = 32
HIDDEN_DIM = 128
LEARNING_RATE = 0.003
GAMMA = 0.99
EPSILON_DECAY = 0.98
EPSILON_MIN = 0.01
TARGET_UPDATE = 50
BUFFER_CAPACITY = 10000
SEED = 42
WARMUP_STEPS = 200

np.random.seed(SEED)
torch.manual_seed(SEED)


def main():
    mlflow.set_experiment("DQN_CartPole_Experiment")
    
    with mlflow.start_run(
        run_name=f"DQN_CartPole_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):
        env = make_env('CartPole-v1', SEED)
        
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
        rewards, losses, q_values = train(env, agent, EPISODES, BATCH_SIZE, SEED, WARMUP_STEPS)
        
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