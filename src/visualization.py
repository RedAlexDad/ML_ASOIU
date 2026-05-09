import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List


def plot_results(
    rewards: List[float],
    losses: List[float],
    q_values: List[float],
    save_path: str = 'plots'
) -> None:
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(rewards, alpha=0.6, label='Награда')
    window = 20
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(rewards)), smoothed, 'r-', label=f'Скользящее среднее (w={window})')
    axes[0, 0].set_xlabel('Эпизод')
    axes[0, 0].set_ylabel('Награда')
    axes[0, 0].set_title('Награда за эпизод')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(losses)
    axes[0, 1].set_xlabel('Эпизод')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Потери при обучении')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(q_values)
    axes[1, 0].set_xlabel('Эпизод')
    axes[1, 0].set_ylabel('Q-значение')
    axes[1, 0].set_title('Q-значения при обучении')
    axes[1, 0].grid(True)
    
    axes[1, 1].hist(rewards, bins=30, edgecolor='black')
    axes[1, 1].set_xlabel('Награда')
    axes[1, 1].set_ylabel('Частота')
    axes[1, 1].set_title('Распределение наград')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/training_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    if len(rewards) >= 50:
        std_val = float(np.std(rewards[-50:]))
        y1 = [r - std_val for r in rewards]
        y2 = [r + std_val for r in rewards]
        plt.fill_between(range(len(rewards)), y1, y2, alpha=0.3)  # type: ignore
    plt.xlabel('Эпизод')
    plt.ylabel('Награда')
    plt.title('Прогресс обучения DQN с доверительным интервалом')
    plt.grid(True)
    plt.savefig(f'{save_path}/training_progress.png', dpi=150)
    plt.close()


def plot_smoothed(rewards: List[float], window: int = 50, save_path: str = 'plots') -> None:
    plt.figure(figsize=(10, 6))
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(rewards)), smoothed)
    plt.xlabel('Эпизод')
    plt.ylabel('Средняя награда')
    plt.title(f'Сглаженная награда (окно={window})')
    plt.grid(True)
    plt.savefig(f'{save_path}/smoothed_reward.png', dpi=150)
    plt.close()


def plot_detailed_analysis(
    rewards: List[float],
    losses: List[float],
    q_values: List[float],
    save_path: str = 'plots'
) -> None:
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    axes[0, 0].plot(rewards, 'b-', alpha=0.5, label='Награда')
    if len(rewards) >= 10:
        ma = np.convolve(rewards, np.ones(10)/10, mode='valid')
        axes[0, 0].plot(range(9, len(rewards)), ma, 'r-', linewidth=2, label='СС(10)')
    axes[0, 0].set_xlabel('Эпизод')
    axes[0, 0].set_ylabel('Награда')
    axes[0, 0].set_title('Награды со скользящим средним')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(losses, 'g-')
    axes[0, 1].set_xlabel('Эпизод')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Потери по эпизодам')
    axes[0, 1].grid(True)
    
    axes[0, 2].plot(q_values, 'm-')
    axes[0, 2].set_xlabel('Эпизод')
    axes[0, 2].set_ylabel('Q-значение')
    axes[0, 2].set_title('Эволюция Q-значений')
    axes[0, 2].grid(True)
    
    axes[1, 0].scatter(range(len(rewards)), rewards, c=rewards, cmap='viridis', alpha=0.7)
    axes[1, 0].set_xlabel('Эпизод')
    axes[1, 0].set_ylabel('Награда')
    axes[1, 0].set_title('Тепловая карта наград')
    axes[1, 0].grid(True)
    
    axes[1, 1].hist(losses, bins=20, color='orange', edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Loss')
    axes[1, 1].set_ylabel('Частота')
    axes[1, 1].set_title('Распределение потерь')
    axes[1, 1].grid(True)
    
    axes[1, 2].bar(['Мин', 'Макс', 'Сред', 'СТД'], [
        min(rewards), max(rewards), np.mean(rewards), np.std(rewards)
    ], color=['red', 'green', 'blue', 'purple'])
    axes[1, 2].set_ylabel('Значение')
    axes[1, 2].set_title('Статистика наград')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/detailed_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()