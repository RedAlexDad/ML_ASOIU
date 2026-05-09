import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    import mlflow
    import mlflow.pytorch

import mlflow as mlflow_module
mlflow = mlflow_module
try:
    import mlflow.pytorch
except Exception:
    mlflow.pytorch = None

from dqn.network import QNetwork
from dqn.buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        target_update: int = 100,
        buffer_capacity: int = 10000,
        device: Optional[torch.device] = None,
        experiment_name: str = "DQN_CartPole"
    ) -> None:
        
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update = target_update
        self.steps = 0
        
        self.experiment_name = experiment_name
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)
        
        self._log_hyperparameters(locals())

    def _log_hyperparameters(self, locals_dict: Dict[str, Any]) -> None:
        hp = {
            "hidden_dim": locals_dict.get("hidden_dim"),
            "lr": locals_dict.get("lr"),
            "gamma": locals_dict.get("gamma"),
            "epsilon_decay": locals_dict.get("epsilon_decay"),
            "epsilon_min": locals_dict.get("epsilon_min"),
            "target_update": locals_dict.get("target_update"),
            "buffer_capacity": locals_dict.get("buffer_capacity"),
            "state_dim": locals_dict.get("state_dim"),
            "action_dim": locals_dict.get("action_dim"),
        }
        mlflow.log_params(hp)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self, batch_size: int) -> Optional[float]:
        if len(self.buffer) < batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.steps += 1
        
        if self.steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()

    def log_metrics(
        self,
        episode: int,
        reward: float,
        loss: Optional[float],
        epsilon: float,
        q_values_mean: Optional[float] = None
    ) -> None:
        metrics = {
            "episode_reward": reward,
            "epsilon": epsilon,
        }
        if loss is not None:
            metrics["loss"] = loss
        if q_values_mean is not None:
            metrics["q_values_mean"] = q_values_mean
        
        mlflow.log_metrics(metrics, step=episode)

    def log_model(self, artifact_path: str = "model") -> None:
        mlflow.pytorch.log_model(
            self.q_network,
            artifact_path,
            registered_model_name=f"{self.experiment_name}_model"
        )

    def save(self, path: str) -> None:
        q_network_cpu = self.q_network.cpu().state_dict()
        target_network_cpu = self.target_network.cpu().state_dict()
        optimizer_cpu = self.optimizer.state_dict()
        
        self.q_network.to(self.device)
        self.target_network.to(self.device)
        
        torch.save({
            'q_network': q_network_cpu,
            'target_network': target_network_cpu,
            'optimizer': optimizer_cpu,
            'epsilon': self.epsilon,
            'steps': self.steps,
        }, path)
        mlflow.log_artifact(path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']