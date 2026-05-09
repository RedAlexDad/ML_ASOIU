import gymnasium as gym
from typing import Dict, Any


def get_env_info(env: gym.Env) -> Dict[str, Any]:
    obs = env.observation_space
    action_n = int(env.action_space.n)  # type: ignore
    return {
        "state_space": obs.shape,
        "action_space": action_n,
        "state_high": obs.high.tolist() if hasattr(obs, 'high') else [],  # type: ignore
        "state_low": obs.low.tolist() if hasattr(obs, 'low') else [],  # type: ignore
    }


def make_env(env_id: str = 'CartPole-v1', seed: int = 42) -> gym.Env:
    env = gym.make(env_id)
    return env