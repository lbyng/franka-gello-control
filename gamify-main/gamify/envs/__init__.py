# If we want to register environments in gym.
# These will be loaded when we import the gamify package.
from gym.envs import register

from .base import EmptyEnv

# Add things you explicitly want exported here.
# Otherwise, all imports are deleted.
__all__ = ["EmptyEnv"]

# from .robomimic import RobomimicEnv
from .deoxys_franka import DeoxysFrankaEnv

del register
