import abc
from typing import Dict


class Task(abc.ABC):
    """These are often tied to a specific environment type.

    Tasks for a given environment are defined in that environment's file.
    """

    @abc.abstractmethod
    def get_reward(self, env, obs: Dict):
        return 0.0

    def reset(self, env, obs: Dict):
        pass

    @property
    def name(self) -> str:
        return "empty"
