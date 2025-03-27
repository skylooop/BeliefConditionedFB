import gymnasium
from envs.env_utils import EpisodeMonitor
from gymnasium.spaces import Box
import numpy as np

class GymXYWrapper(gymnasium.Wrapper):
    """Wrapper for directional locomotion tasks."""

    def __init__(self, env, resample_interval=100):
        """Initialize the wrapper.

        Args:
            env: Environment.
            resample_interval: Interval at which to resample the direction.
        """
        super().__init__(env)

        self.z = None
        self.num_steps = 0
        self.resample_interval = resample_interval

        ob, _ = self.reset()
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=ob.shape, dtype=np.float64)

    def reset(self, *args, **kwargs):
        ob, info = self.env.reset(*args, **kwargs)
        self.num_steps = 0

        return ob, info

    def step(self, action):
        cur_xy = self.unwrapped.data.qpos[:2].copy()
        ob, reward, terminated, truncated, info = self.env.step(action)
        next_xy = self.unwrapped.data.qpos[:2].copy()
        self.num_steps += 1

        # Reward is the dot product of the direction and the change in xy.
        # reward = (next_xy - cur_xy).dot(self.z)
        info['xy'] = next_xy
        dir = (next_xy - cur_xy)
        info['direction'] = dir / np.linalg.norm(dir)

        return ob, reward, terminated, truncated, info

def make_online_env(env_name, default_ind):
    """Make online environment.

    If the environment name contains the '-xy' suffix, the environment will be wrapped with a directional locomotion
    wrapper. For example, 'online-ant-xy-v0' will return an 'online-ant-v0' environment wrapped with GymXYWrapper.

    Args:
        env_name: Name of the environment.
    """

    # Manually recognize the '-xy' suffix, which indicates that the environment should be wrapped with a directional
    # locomotion wrapper.
    if '-xy' in env_name:
        env_name = env_name.replace('-xy', '')
        apply_xy_wrapper = True
    else:
        apply_xy_wrapper = False

    # Set camera.
    if 'humanoid' in env_name:
        extra_kwargs = dict(camera_id=0)
    else:
        extra_kwargs = dict()

    # Make environment.
    env = gymnasium.make(env_name, render_mode='rgb_array', height=200, width=200, default_ind=default_ind, **extra_kwargs)

    if apply_xy_wrapper:
        # Apply the directional locomotion wrapper.
        from ogbench.online_locomotion.wrappers import DMCHumanoidXYWrapper

        if 'humanoid' in env_name:
            env = DMCHumanoidXYWrapper(env, resample_interval=200)
        else:
            env = GymXYWrapper(env, resample_interval=100)

    env = EpisodeMonitor(env)

    return env
