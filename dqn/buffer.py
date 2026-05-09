"""
Модуль буфера повторения опыта (Replay Buffer).
Используется для стабилизации обучения DQN - разрывает корреляцию между последовательными переходами.
"""

import numpy as np
import random
from collections import deque
from typing import Tuple, List


class ReplayBuffer:
    """
    Буфер для хранения переходов (s, a, r, s', done).
    
    Принцип работы:
    1. Агент взаимодействует с окружением и сохраняет переходы в буфер
    2. При обучении случайно выбираем батч из буфера (Experience Replay)
    3. Это устраняет корреляцию между последовательными наблюдениями
    
    Преимущества:
    - Повторное использование опыта (эффективнее данных)
    - Случайная выборка стабилизирует градиенты
    - Разрывает временную зависимость между переходами
    """
    
    def __init__(self, capacity: int = 10000) -> None:
        """
        Инициализация буфера.
        
        Args:
            capacity: Максимальное количество хранимых переходов.
                     При достижении лимита старые переходы удаляются.
        """
        # deque - эффективная структура с O(1) добавлением/удалением
        # maxlen=capacity автоматически удаляет старые элементы
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """
        Добавить переход в буфер.
        
        Args:
            state: Текущее состояние s
            action: Выбранное действие a
            reward: Полученная награда r
            next_state: Следующее состояние s'
            done: Флаг завершения эпизода (terminal state)
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Случайно выбрать батч переходов для обучения.
        
        Args:
            batch_size: Размер батча
            
        Returns:
            Кортеж массивов: (states, actions, rewards, next_states, dones)
        """
        # Случайная выборка без повторений
        batch: List = random.sample(self.buffer, batch_size)
        
        # Распаковка батча на отдельные компоненты
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),         # (batch_size, state_dim)
            np.array(actions, dtype=np.int64),          # (batch_size,)
            np.array(rewards, dtype=np.float32),        # (batch_size,)
            np.array(next_states, dtype=np.float32),    # (batch_size, state_dim)
            np.array(dones, dtype=np.float32)           # (batch_size,) - 1.0 если done, иначе 0.0
        )

    def __len__(self) -> int:
        """Возвращает количество переходов в буфере."""
        return len(self.buffer)