import gymnasium as gym
from typing import Dict, Any


def get_env_info(env: gym.Env) -> Dict[str, Any]:
    return {
        "state_space": env.observation_space.shape,
        "action_space": env.action_space.n,
        "state_high": env.observation_space.high.tolist(),
        "state_low": env.observation_space.low.tolist(),
    }


def make_env(env_id: str = 'CartPole-v1', seed: int = 42) -> gym.Env:
    env = gym.make(env_id)
    return env