"""
Модуль нейронной сети для DQN агента.
Реализует аппроксимацию Q-функции: Q(s, a) -> оценка качества действия в состоянии.

Доступные архитектуры:
1. QNetwork - базовая DQN сеть (3 слоя, ReLU)
2. DuelingQNetwork - Dueling DQN с разделением V(s) и A(s,a)
3. QNetworkBN - DQN с BatchNorm для стабилизации
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mlflow
import mlflow.pytorch
from datetime import datetime


class QNetwork(nn.Module):
    """
    Нейронная сеть для оценки Q-значений (Value Network).
    
    Архитектура: 3 полносвязных слоя с ReLU активацией.
    Вход: состояние окружения (state_dim,)
    Выход: Q-значения для каждого действия (action_dim,)
    
    Пример для CartPole-v1:
        - state_dim = 4 (позиция, скорость, угол, угловая скорость)
        - action_dim = 2 (влево или вправо)
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super(QNetwork, self).__init__()
        
        # Первый полносвязный слой: state -> hidden
        # Преобразует вектор состояния в скрытое представление
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        
        # Второй полносвязный слой: hidden -> hidden
        # Дополнительная нелинейность для сложных зависимостей
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Выходной слой: hidden -> action
        # Генерирует Q-значения для каждого возможного действия
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через сеть.
        
        Args:
            x: Тензор состояния формы (batch_size, state_dim)
            
        Returns:
            Тензор Q-значений формы (batch_size, action_dim)
        """
        # ReLU: f(x) = max(0, x) - добавляет нелинейность
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DuelingQNetwork(nn.Module):
    """
    Архитектура Dueling DQN - разделяет оценку состояния и преимущества действий.
    
    Основная идея Dueling DQN:
    Q(s, a) = V(s) + A(s, a) - avg(A(s, a))
    
    где:
    - V(s) - ценность состояния (Value)
    - A(s, a) - преимущество действия (Advantage)
    
    Преимущества:
    - Лучше оценивает состояния без действий
    - Сходится быстрее для задач с многими действиями
    - Меньше параметров чем полная Q-сеть
    
    Вдохновлено: "Dueling Network Architectures for Deep Reinforcement Learning" (Wang et al., 2016)
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super(DuelingQNetwork, self).__init__()
        
        # Общие признаки - извлекают представление состояния
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream - оценивает ценность состояния V(s)
        # Отвечает на вопрос: "Насколько хорошо находиться в этом состоянии?"
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Один выход - V(s)
        )
        
        # Advantage stream - оценивает преимущество каждого действия A(s,a)
        # Отвечает на вопрос: "Насколько лучше это действие относительно других?"
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)  # action_dim выходов - A(s,a)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через сеть.
        
        Args:
            x: Тензор состояния формы (batch_size, state_dim)
            
        Returns:
            Тензор Q-значений формы (batch_size, action_dim)
        """
        # Извлечение признаков
        features = self.feature(x)
        
        # Вычисление V(s) и A(s,a)
        value = self.value_stream(features)  # (batch_size, 1)
        advantage = self.advantage_stream(features)  # (batch_size, action_dim)
        
        # Комбинация: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        # Вычитание среднего делает сеть идентифицируемой
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values


class QNetworkBN(nn.Module):
    """
    QNetwork с BatchNorm для стабилизации обучения.
    
    BatchNorm преимущества:
    - Стабилизация градиентов (внутреннее смещение covariate shift)
    - Ускорение сходимости
    - Регуляризация (менее подвержен переобучению)
    - Позволяет использовать более высокий learning rate
    
    Вдохновлено: "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" (Ioffe & Szegedy, 2015)
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super(QNetworkBN, self).__init__()
        
        # BatchNorm нормализует входные данные каждого слоя
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Dropout для дополнительной регуляризации
        # Случайно обнуляет нейроны во время обучения
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через сеть с BatchNorm и Dropout.
        
        Args:
            x: Тензор состояния формы (batch_size, state_dim)
            
        Returns:
            Тензор Q-значений формы (batch_size, action_dim)
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        return self.fc3(x)