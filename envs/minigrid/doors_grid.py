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

class DiscreteActions(IntEnum):
    move_left = 0
    move_right = 1
    move_up = 2
    move_down = 3

class DynamicsGeneralization_Doors(MiniGridEnv):
    def __init__(
        self,
        size=9,
        agent_start_pos=(1, 4),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height, layout_type=None):
        # Create an empty grid
        self.grid = Grid(width, height)

        self.grid.wall_rect(0, 0, width, height)

        door_1 = np.random.randint(low=1, high=height - 1)
        door_2 = np.random.randint(low=1, high=height - 1)

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
                
        # # Place the door and key
        # if layout_type == 1:
        #     self.grid.set(4, 2, Door(COLOR_NAMES[0], is_locked=False, is_open=True))

        # elif layout_type == 0:
        #     self.grid.set(4, 6, Door(COLOR_NAMES[3], is_locked=False, is_open=True))
        # else:
        #     self.grid.set(4, 2, Door(COLOR_NAMES[0], is_locked=False, is_open=True))
        #     self.grid.set(4, 6, Door(COLOR_NAMES[3], is_locked=False, is_open=True))
        self.grid.set(3, door_3, Door(COLOR_NAMES[0], is_locked=False, is_open=True))
        self.grid.set(5, door_5, Door(COLOR_NAMES[3], is_locked=False, is_open=True))
    
        self.put_obj(Goal(), width - 2, 4)
        self.goal_pos = np.array((width-2, 4))
        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "grand mission"

class MinigridWrapper:
    def __init__(self, env):
        self.env = SymbolicObsWrapper(env)
        self.env.unwrapped.actions = DiscreteActions
        self.env.unwrapped.action_space = spaces.Discrete(len(self.env.unwrapped.actions))
        
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
            new_pos = current_pos + np.array([-1, 0])
        elif action == self.env.unwrapped.actions.move_right:
            new_pos = current_pos + np.array([1, 0])
        elif action == self.env.unwrapped.actions.move_up:
            new_pos = current_pos + np.array([0, -1])
        elif action == self.env.unwrapped.actions.move_down:
            new_pos = current_pos + np.array([0, 1])
        else:
            raise ValueError(f"Invalid action: {action}")

        # Clip the new position to ensure it stays within the grid boundaries
        new_pos = np.clip(new_pos, [0, 0], [self.env.unwrapped.width - 1, self.env.unwrapped.height - 1])

        # Get the contents of the cell at the new position
        new_cell = self.env.unwrapped.grid.get(*new_pos)

        # Move the agent if the new cell is walkable
        if new_cell is None or new_cell.can_overlap():
            self.env.unwrapped.agent_pos = tuple(new_pos)

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
        obs = self.env.unwrapped.gen_obs()
        return self.env.observation(obs), reward, terminated, truncated, {"goal_pos": self.env.unwrapped.goal_pos}
      
    def reset(self):
        self.coverage_map = np.zeros(shape=(self.env.unwrapped.width, self.env.unwrapped.height))
        obs, info = self.env.reset()
        info['goal_pos'] = self.env.unwrapped.goal_pos
        return obs, info
    
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