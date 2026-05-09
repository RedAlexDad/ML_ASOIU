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
    
    axes[0, 0].plot(rewards, alpha=0.6, label='Raw')
    window = 20
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(rewards)), smoothed, 'r-', label=f'Smoothed (w={window})')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Total Reward per Episode')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(losses)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training Loss')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(q_values)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Mean Q-Value')
    axes[1, 0].set_title('Q-Values during Training')
    axes[1, 0].grid(True)
    
    axes[1, 1].hist(rewards, bins=30, edgecolor='black')
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Reward Distribution')
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
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Training Progress with Confidence Interval')
    plt.grid(True)
    plt.savefig(f'{save_path}/training_progress.png', dpi=150)
    plt.close()


def plot_smoothed(rewards: List[float], window: int = 50, save_path: str = 'plots') -> None:
    plt.figure(figsize=(10, 6))
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(rewards)), smoothed)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title(f'Smoothed Reward (window={window})')
    plt.grid(True)
    plt.savefig(f'{save_path}/smoothed_reward.png', dpi=150)
    plt.close()