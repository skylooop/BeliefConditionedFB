import numpy as np
from gymnasium import spaces
from enum import IntEnum

import matplotlib.pyplot as plt
import mediapy
import scienceplots

import minigrid
from minigrid.minigrid_env import MiniGridEnv, MissionSpace, Grid
from minigrid.core.world_object import Wall, WorldObj, Goal, Door
from minigrid.core.constants import COLOR_NAMES
import gymnasium as gym
from minigrid.wrappers import SymbolicObsWrapper

class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env, action_repeat=4, action_repeat_mode="id"):
        super().__init__(env)
        self.action_repeat = action_repeat
        self.action_repeat_mode = action_repeat_mode

    def get_obs(self, *args, **kwargs):
        obs = self.unwrapped._get_obs(*args, **kwargs)
        return obs

    def step(self, action):
        status = self.env.step(action)

        for i in range(1, self.action_repeat):
            if self.action_repeat_mode == "id":
                step_a = action
            elif self.action_repeat_mode == "linear":
                step_a = action - i * (action / self.action_repeat)
            elif self.action_repeat_mode == "null":
                step_a = np.array([0, 0])
            else:
                raise NotImplementedError

            status = self.env.step(step_a)

        return status

class DiscreteActions(IntEnum):
    move_left = 2
    move_right = 3
    move_up = 0
    move_down = 1

class DynamicsGeneralization_Doors(MiniGridEnv):
    def __init__(
        self,
        size=9,
        agent_start_pos=(1, 4),
        agent_start_dir=0,
        max_steps: int | None = None,
        task_num: int = 0,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.task_num = task_num
        
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height, layout_type=None, empty=False):
        # Create an empty grid
        self.grid = Grid(width, height)

        self.grid.wall_rect(0, 0, width, height)

        if not empty:
            for i in range(0, height):
                self.grid.set(5, i, Wall())
                self.grid.set(3, i, Wall())
            
            if layout_type == 1:
                door_3 = 2
                door_5 = 6
            elif layout_type == 0:
                door_3 = 6
                door_5 = 2
            else:
                # If layout_type is provided, use it as a seed; else, random
                if layout_type is not None:
                    rng = np.random.RandomState(layout_type)
                    door_3 = rng.randint(1, height - 1)
                    door_5 = rng.randint(1, height - 1)
                else:
                    door_3 = np.random.randint(1, height - 1)
                    door_5 = np.random.randint(1, height - 1)
                    
            self.grid.set(3, door_3, Door(COLOR_NAMES[0], is_locked=False, is_open=True))
            self.grid.set(5, door_5, Door(COLOR_NAMES[3], is_locked=False, is_open=True))
            
        if self.task_num == 0: # currently hardcoded
            goal_coordinates = (width - 2, 4)
        elif self.task_num == 1:
            goal_coordinates = (width - 2, 1)
        elif self.task_num == 2:
            goal_coordinates = (width - 2, 7)
            
        self.put_obj(Goal(), goal_coordinates[0], goal_coordinates[1])
        self.goal_pos = np.array((goal_coordinates[0], goal_coordinates[1]))
        
        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "grand mission"

class MinigridWrapper(gym.Env):
    def __init__(self, env):
        self.env = env #SymbolicObsWrapper(env)
        self.env.unwrapped.actions = DiscreteActions
        self.env.unwrapped.action_space = spaces.Discrete(len(self.env.unwrapped.actions))
        self.observation_space = spaces.MultiDiscrete(nvec=[self.env.unwrapped.width, self.env.unwrapped.height]).sample()
        self.action_space = self.env.unwrapped.action_space
        
    def step(self, action):
        self.coverage_map[self.env.unwrapped.agent_pos[1], self.env.unwrapped.agent_pos[0]] += 1
        self.env.unwrapped.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Get the current position of the agent
        current_pos = np.array(self.env.unwrapped.agent_pos)

        # Calculate the new position based on the action
        if action == self.env.unwrapped.actions.move_left:
            new_pos = current_pos + np.array([0, -1])
        elif action == self.env.unwrapped.actions.move_right:
            new_pos = current_pos + np.array([0, 1])
        elif action == self.env.unwrapped.actions.move_up:
            new_pos = current_pos + np.array([-1, 0])
        elif action == self.env.unwrapped.actions.move_down:
            new_pos = current_pos + np.array([1, 0])
        else:
            raise ValueError(f"Invalid action: {action}")

        # Clip the new position to ensure it stays within the grid boundaries
        new_pos = np.clip(new_pos, [0, 0], [self.env.unwrapped.width - 1, self.env.unwrapped.height - 1])

        # Get the contents of the cell at the new position
        new_cell = self.env.unwrapped.grid.get(*new_pos)

        # Move the agent if the new cell is walkable
        if new_cell is None or new_cell.can_overlap():
            self.env.unwrapped.agent_pos = tuple(new_pos)
        else:
            new_pos = current_pos
        # Check if the new cell is a goal or lava
        if new_cell is not None and new_cell.type == "goal":
            terminated = True
            reward = self.env.unwrapped._reward()
        if new_cell is not None and new_cell.type == "lava":
            terminated = True

        # Check if the episode is truncated due to step limit
        if self.env.unwrapped.step_count >= self.env.unwrapped.max_steps:
            truncated = True

        # Generate the observation and return the result
        # obs = self.env.unwrapped.gen_obs()
        return new_pos, reward, terminated, truncated, {"goal_pos": self.env.unwrapped.goal_pos}
    
    def reset(self, seed = None, **kwargs):
        self.coverage_map = np.zeros(shape=(self.env.unwrapped.width, self.env.unwrapped.height))
        obs, info = self.env.reset(seed=seed)
        info['goal_pos'] = self.env.unwrapped.goal_pos
        return self.env.unwrapped.agent_pos, info
    
    def render(self):
        return self.env.render()
    
    def visualize_coverage(self):
        plt.imshow(self.coverage_map, cmap='inferno', vmin=0)
        plt.colorbar()
        plt.show()

    def get_grid_array(self):
        grid = (self.env.env.unwrapped.grid.encode()[:, :, 0].T).astype(np.int16)
        grid = np.where(grid == 2, -1000, grid)
        return grid
    
    def plot_grid(self, ax, grid, add_start=False):
        asbestos = (0.2, 0.2, 0.2, 1.0)
        grid_kwargs = {'color': (220 / 255, 220 / 255, 220 / 255, 0.5)}
        img = np.ones((self.env.unwrapped.width, self.env.unwrapped.height, 4))
        wall_y, wall_x = np.where(grid == -1000)
        for i in range(len(wall_y)):
            img[wall_y[i], wall_x[i]] = np.array(asbestos)
        ax.imshow(img, interpolation=None)
        ax.grid(0)
        plt.xticks([])
        plt.yticks([])
        if add_start:
            plt.text(
            self.start[1],
            self.start[0],
            r'$\mathbf{S}$',
            fontsize=16,
            ha='center',
            va='center')
        h, w = (self.env.unwrapped.width, self.env.unwrapped.height)
        for y in range(h - 1):
            plt.plot([-0.5, w - 0.5], [y + 0.5, y + 0.5], **grid_kwargs)
        for x in range(w - 1):
            plt.plot([x + 0.5, x + 0.5], [-0.5, h - 0.5], **grid_kwargs)
        return ax
    
# env = DynamicsGeneralization(render_mode="rgb_array", highlight=False, max_steps=200)
# # env._gen_grid = partial(env._gen_grid, layout_type=1)
# env = MinigridWrapper(env)

# env.reset()
# plt.imshow(env.render())