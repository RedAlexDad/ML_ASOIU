"""
Модуль DQN агента.
Реализует основной алгоритм Deep Q-Network для обучения с подкреплением.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, Any

import mlflow  # type: ignore

from dqn.network import QNetwork, DuelingQNetwork, QNetworkBN, QNetworkLSTM
from dqn.buffer import ReplayBuffer


class DQNAgent:
    """
    Агент, реализующий алгоритм DQN (Deep Q-Network).
    
    Основные компоненты:
    1. Q-Network - нейросеть для оценки Q-значений Q(s, a)
    2. Target Network - копия Q-сети для вычисления целевых значений
    3. Replay Buffer - буфер для хранения опыта
    4. Epsilon-greedy - стратегия исследования/эксплуатации
    
    Алгоритм обучения:
    1. Выбрать действие через epsilon-greedy политику
    2. Выполнить действие, получить (s, a, r, s', done)
    3. Сохранить переход в буфер
    4. Если буфер достаточно заполнен - обучать на случайном батче:
       - Q(s, a) - текущая оценка
       - target = r + γ * max_a' Q_target(s', a') - целевое значение (Bellman equation)
       - Минимизировать MSE между Q и target
    5. Периодически обновлять Target Network
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        network_type: str = "qnetwork",
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
        """
        Инициализация DQN агента.
        
        Args:
            state_dim: Размерность пространства состояний
            action_dim: Количество возможных действий
            hidden_dim: Количество нейронов в скрытых слоях
            network_type: Тип сети ('qnetwork', 'dueling', 'bn')
            lr: Скорость обучения (learning rate)
            gamma: Коэффициент дисконтирования награды (0.95-0.99)
            epsilon: Начальная вероятность случайного действия
            epsilon_decay: Коэффициент уменьшения epsilon после каждого шага
            epsilon_min: Минимальное значение epsilon (не прекращаем исследование полностью)
            target_update: Частота обновления целевой сети (каждые N шагов)
            buffer_capacity: Размер буфера опыта
            device: Устройство для вычислений (CPU/CUDA)
            experiment_name: Имя эксперимента для MLflow
        """
        # Выбор архитектуры сети
        network_classes = {
            'qnetwork': QNetwork,
            'dueling': DuelingQNetwork,
            'bn': QNetworkBN,
            'lstm': QNetworkLSTM
        }
        NetworkClass = network_classes.get(network_type, QNetwork)
        
        # Параметры действий и награды
        self.action_dim = action_dim
        self.gamma = gamma  # γ - коэффициент дисконтирования
        self.network_type = network_type
        
        # Параметры epsilon-greedy стратегии
        self.epsilon = epsilon  # Текущее значение epsilon
        self.epsilon_decay = epsilon_decay  # Во сколько раз уменьшаем epsilon
        self.epsilon_min = epsilon_min  # Минимальное значение epsilon
        
        self.target_update = target_update  # Интервал обновления target сети
        self.steps = 0  # Счетчик шагов обучения
        
        self.experiment_name = experiment_name
        
        # Определение устройства (CPU или GPU)
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # Инициализация сетей
        # Q-Network - обучаемая сеть (выбранного типа)
        self.q_network = NetworkClass(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Target Network - фиксированная цель для стабильности
        # Инициализируется весами Q-network
        self.target_network = NetworkClass(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Режим инференса (без обучения)
        
        # Оптимизатор - Adam с настраиваемым lr
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Буфер опыта для Experience Replay
        self.buffer = ReplayBuffer(buffer_capacity)
        
        # Логирование гиперпараметров в MLflow
        self._log_hyperparameters(locals())

    def _log_hyperparameters(self, locals_dict: Dict[str, Any]) -> None:
        """Логирование гиперпараметров в MLflow."""
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
        """
        Выбор действия на основе текущей политики.
        
        Epsilon-greedy стратегия:
        - С вероятностью epsilon: случайное действие (исследование)
        - С вероятностью 1-epsilon: действие с максимальным Q-значением (эксплуатация)
        
        Args:
            state: Текущее состояние
            training: Флаг режима обучения (влияет на epsilon-greedy)
            
        Returns:
            Выбранное действие (целое число)
        """
        # Исследование: случайное действие
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        # Эксплуатация: выбираем действие с максимальным Q-значением
        self.q_network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
        self.q_network.train()
        return action

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Сохранение перехода в буфер опыта.
        
        Args:
            state: Текущее состояние s
            action: Выбранное действие a
            reward: Полученная награда r
            next_state: Следующее состояние s'
            done: Флаг завершения эпизода
        """
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self, batch_size: int) -> Optional[float]:
        """
        Один шаг обучения на случайном батче из буфера.
        
        Алгоритм (Double DQN с MSE loss):
        1. sampling: Выбрать случайный батч из буфера
        2. Q-values: Q(s, a) - текущие оценки для выбранных действий
        3. Target: r + γ * max_a' Q_target(s', a') - целевое значение
           - Используем target_network для устойчивости
           - (1 - done) обнуляет награду для terminal states
        4. Loss: MSE(current_q, target_q)
        5. Backprop: обновить веса Q-network
        6. Update: обновить target network и epsilon
        
        Args:
            batch_size: Размер батча для обучения
            
        Returns:
            Значение функции потерь или None если буфер слишком мал
        """
        # Не обучаем если недостаточно данных в буфере
        if len(self.buffer) < batch_size:
            return None
        
        # 1. Получить случайный батч из буфера
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        # 2. Конвертировать в тензоры PyTorch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 3. Вычислить текущие Q-значения для выбранных действий
        # gather(1, actions.unsqueeze(1)) выбирает Q(s, a) для конкретных действий
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 4. Вычислить целевые Q-значения через Target Network
        # Double DQN: используем Q-network для выбора, target_network для оценки
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            # Уравнение Беллмана: Q(s,a) = r + γ * max Q(s',a')
            # Для terminal states (done=True) последний член обнуляется
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 5. Вычислить и минимизировать потери (MSE Loss)
        loss = F.mse_loss(current_q, target_q)
        
        # 6. Обратное распространение
        self.optimizer.zero_grad()  # Обнулить градиенты
        loss.backward()            # Вычислить градиенты
        
        # Gradient clipping - предотвратить взрыв градиентов
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()  # Обновить веса
        
        # 7. Обновить счетчик шагов
        self.steps += 1
        
        # 8. Периодически обновлять Target Network (политика фиксированных целей)
        if self.steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        # 9. Уменьшать epsilon (меньше исследований со временем)
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
        """
        Логирование метрик обучения в MLflow.
        
        Args:
            episode: Номер эпизода
            reward: Награда за эпизод
            loss: Значение функции потерь
            epsilon: Текущее значение epsilon
            q_values_mean: Среднее Q-значение
        """
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
        """
        Сохранить модель в MLflow.
        
        Args:
            artifact_path: Путь для сохранения артефакта
        """
        mlflow.pytorch.log_model(  # type: ignore
            self.q_network,
            artifact_path,
            registered_model_name=f"{self.experiment_name}_model"
        )

    def save(self, path: str) -> None:
        """
        Сохранить веса модели и состояние агента.
        
        Сохраняемые компоненты:
        - Веса Q-network
        - Веса Target Network
        - Состояние оптимизатора
        - Текущий epsilon (политика)
        - Количество шагов
        
        Args:
            path: Путь для сохранения файла
        """
        # Перенести на CPU для сериализации
        q_network_cpu = self.q_network.cpu().state_dict()
        target_network_cpu = self.target_network.cpu().state_dict()
        optimizer_cpu = self.optimizer.state_dict()
        
        # Вернуть обратно на устройство
        self.q_network.to(self.device)
        self.target_network.to(self.device)
        
        # Сохранить словарь состояния
        torch.save({
            'q_network': q_network_cpu,
            'target_network': target_network_cpu,
            'optimizer': optimizer_cpu,
            'epsilon': self.epsilon,
            'steps': self.steps,
        }, path)
        
        # Логировать в MLflow
        mlflow.log_artifact(path)

    def load(self, path: str) -> None:
        """
        Загрузить веса модели и состояние агента.
        
        Args:
            path: Путь к файлу с сохраненными весами
        """
        checkpoint = torch.load(path, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']