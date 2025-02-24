import numpy as np

from envs.custom_mazes import BaseMaze, BaseEnv, Object, DeepMindColor as color
from envs.custom_mazes.generators.four_room import generate_four_room_env
from envs.custom_mazes.motion import VonNeumannMotion
from gymnasium.spaces import Discrete, Dict, Box
import matplotlib.pyplot as plt

class Maze(BaseMaze):
    def __init__(self, maze_type: str='fourrooms', size: str = '11', **kwargs):
        if maze_type == 'fourrooms':
            self.maze_grid = generate_four_room_env(int(size), int(size))
        super().__init__(**kwargs)
        
    @property
    def size(self):
        return self.maze_grid.shape
    
    def make_objects(self):
        free = Object('free', 0, color.free, False, np.stack(np.where(self.maze_grid == 0), axis=1))
        obstacle = Object('obstacle', 1, color.obstacle, True, np.stack(np.where(self.maze_grid == 1), axis=1))
        agent = Object('agent', 2, color.agent, False, [])
        goal = Object('goal', 3, color.goal, False, [])
        return free, obstacle, agent, goal

class FourRoomsMazeEnv(BaseEnv):
    def __init__(self, maze, max_width=100, obs_type="xy"):
        super().__init__(max_width)
        
        self.maze = maze
        self.motions = VonNeumannMotion()
        self.action_space = Discrete(len(self.motions))
        if obs_type == "xy":
            self.observation_space = Box(low=1, high=self.maze.size[0] - 1, shape=(2, ), dtype=np.uint8)
        elif obs_type == "onehot":
            self.observation_space = Box(low=0, high=1, shape=(self.maze.size[0] ** 2, ), dtype=np.uint8)
        self.goal = None
        self.start = None
        self.maze_state = self.maze.maze_grid
        
    def reset(self, seed=None, options={}):
        super().reset(seed=seed, options=options)
        start_idx = options.get('start', None)
        goal_idx = options.get('goal', None)

        if start_idx is None:
            start_idx = self.generate_pos()
        if goal_idx is None:
            goal_idx = self.generate_goal()
            
        self.maze.objects.agent.positions = [start_idx]
        self.maze.objects.goal.positions = [goal_idx]
        self.goal = goal_idx
        self.start = start_idx
        self.step_count = 0
        self.maze_state = self.maze.to_value()
        return np.array(start_idx), {"goal_pos": np.array(goal_idx)}
    
    def setup_goals(self, seed: int, task_num=None):
        goal_list = [(2, 2), (2, 9),
                      (8, 8), (8, 2)]
        if task_num is None:
            random_goal = goal_list[np.random.randint(len(goal_list)) - 1]
        else:
            random_goal = goal_list[task_num - 1]
        return self.reset(seed=seed, options={"goal": random_goal})
    
    def generate_pos(self):
        return self.np_random.choice(self.maze.objects.free.positions)
    
    def generate_goal(self):
        return self.np_random.choice(self.maze.objects.free.positions)
    
    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1]
        passable = not self.maze.to_impassable()[position[0]][position[1]]
        return nonnegative and within_edge and passable
    
    def _is_goal(self, position):
        for goal_pos in self.maze.objects.goal.positions:
            if np.array_equal(goal_pos, position):
                return True
        return False
    
    def step(self, action):
        motion = self.motions[action]
        current_position = self.maze.objects.agent.positions[0]
        new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]
        valid = self._is_valid(new_position)
        reward = 0.0
        done=False
        self.step_count += 1
        if valid:
            self.maze.objects.agent.positions = [new_position]
        if self._is_goal(new_position):
            reward = 1.0
            done = True
        else:
            new_position = current_position
        self.maze_state = self.maze.to_value()
        if self.step_count >= 200:
            done = True
        return np.array(new_position), reward, done, False, {}
        
    def visualize_goals(self):
        fig, ax = plt.subplots(nrows=2, ncols=2)
        for i, cur_ax in enumerate(ax.flat, start=1):
            self.setup_goals(seed=None, task_num=i)
            self.render(ax=cur_ax)
            cur_ax.set_title(f"Goal: {self.goal}")
        plt.tight_layout()
        
    def get_image(self):
        return self.maze.to_rgb()