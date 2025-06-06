import numpy as np

from envs.custom_mazes import BaseMaze, BaseEnv, Object, DeepMindColor as color
from envs.custom_mazes.generators.four_room import generate_four_room_env, fourrooms_random_layouts
from envs.custom_mazes.generators.gridworld import gridworld

from envs.custom_mazes.motion import VonNeumannMotion
import gymnasium.spaces as spaces
import matplotlib.pyplot as plt

from envs.custom_mazes.env_utils import policy_image_fourrooms, value_image_fourrooms
from functools import partial
from utils.evaluation import supply_rng
from IPython.display import clear_output
import jax

class Maze(BaseMaze):
    def __init__(self, seed, maze_type: str='fourrooms', size: str = 11, **kwargs):
        if maze_type == 'fourrooms':
            self.maze_grid = generate_four_room_env(size, size)
        elif maze_type == "gridworld":
            self.maze_grid = gridworld()
        elif maze_type == "fourrooms_random_layouts":
            self.maze_grid = fourrooms_random_layouts(size, size, seed=seed)
        self.num_tasks = 4
        self.maze_type = maze_type
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
    def __init__(self, maze, max_width=100, obs_type="xy", max_steps: int = 100):
        super().__init__(max_width)
        
        self.maze = maze
        self.motions = VonNeumannMotion()
        self.action_space = spaces.Discrete(len(self.motions))
        if obs_type == "xy":
            self.observation_space = spaces.MultiDiscrete(nvec=[self.maze.size[0], self.maze.size[1]]).sample()
        self.goal = None
        self.start = None
        self.maze_state = self.maze.maze_grid
        self.max_steps = max_steps
        
    def map_state_to_idx(self, states):
        return np.sum(states * np.array((self.maze_state.shape[0], 1)), axis=-1)
    
    def map_idx_to_state(self, index):
        return np.stack((index // self.maze_state.shape[0], index % self.maze_state.shape[0]), axis=-1)
    
    def reset(self, seed=None, options={}):
        super().reset(seed=seed, options=options)
        self.coverage_map = np.zeros(shape=self.maze.size)
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
        self.cur_pos = start_idx
        self.maze_state = self.maze.to_value()

        return np.array(start_idx), {"goal_pos": np.array(goal_idx)}

    def get_state_list(self):
        state_list = []
        for y in range(self.maze.maze_grid.shape[0]):
            for x in range(self.maze.maze_grid.shape[1]):
                if self.maze.maze_grid[y, x] > 0:
                    state_list.append((x, y))
        return state_list
    
    def setup_goals(self, seed: int, task_num=None, start_pos=None):
        if self.maze.maze_type == "fourrooms":
            goal_list = [(2, 8), (8, 2), (8, 8)]
            
        elif self.maze.maze_type == "gridworld":
            goal_list = [(3,2), (3,7), (7, 5), (6, 8)]
        
        elif self.maze.maze_type == "fourrooms_random_layouts":
            goal_list = [
                (2, self.maze.size[-1] - 3),
                (self.maze.size[0] - 3, 2), (self.maze.size[0] - 3, self.maze.size[-1]-3)
            ]
        
        if task_num is None:
            random_goal = goal_list[np.random.randint(len(goal_list)) - 1]
        else:
            random_goal = goal_list[task_num - 1]
        options = {"goal": random_goal}
        if start_pos is not None:
            options.update({"start": start_pos})
        return self.reset(seed=seed, options=options)
    
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
        current_position = self.maze.objects.agent.positions[0]
        self.coverage_map[current_position[1], current_position[0]] += 1
        motion = self.motions[action]
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
        self.cur_pos = new_position
        if self.step_count >= self.max_steps:
            done = True
        return np.array(new_position), reward, done, False, {}
        
    def visualize_goals(self):
        fig, ax = plt.subplots(nrows=2, ncols=2) # currently hardcoded to 4 goals, one for each room
        for i, cur_ax in enumerate(ax.flat, start=1):
            self.setup_goals(seed=None, task_num=i)
            self.render(ax=cur_ax)
            cur_ax.set_title(f"Goal: {self.goal}")
        plt.tight_layout()
        
    def get_image(self):
        return self.maze.to_rgb()

    def plot_grid(self, ax, add_start=True):
        asbestos = (0.2, 0.2, 0.2, 1.0)
        grid_kwargs = {'color': (220 / 255, 220 / 255, 220 / 255, 0.5)}
        img = np.ones((self.maze.maze_grid.shape[0], self.maze.maze_grid.shape[1], 4))
        wall_y, wall_x = np.where(self.maze.maze_grid == 1)
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
        h, w = self.maze.maze_grid.shape
        for y in range(h - 1):
            plt.plot([-0.5, w - 0.5], [y + 0.5, y + 0.5], **grid_kwargs)
        for x in range(w - 1):
            plt.plot([x + 0.5, x + 0.5], [-0.5, h - 0.5], **grid_kwargs)
        return ax

def visualize_value_image(agent, example_batch, task_num):
    env = FourRoomsMazeEnv(Maze())
    env.reset()
    observation, info = env.setup_goals(seed=None, task_num=task_num)
    goal = info.get("goal_pos", None)
    latent_z = jax.device_get(agent.infer_z(goal)[None])
    N, M = env.maze.size
    pred_value_img = value_image_fourrooms(env, example_batch, N=N, M=M,
                                value_fn=partial(agent.predict_q, z=latent_z), goal=goal)
    return pred_value_img

def visualize_policy(agent, whole_dataset, task_num):
    env = FourRoomsMazeEnv(Maze())
    env.reset()
    observation, info = env.setup_goals(seed=None, task_num=task_num)
    goal = info.get("goal_pos", None)
    latent_z = agent.infer_z(goal)
    start = info.get("start_pos", None)
    example_batch = whole_dataset.sample(1)
    pred_policy_img = policy_image_fourrooms(env, example_batch, N=env.maze.size[0], M=env.maze.size[1],
                                                    action_fn=partial(supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32))), latent_z=latent_z, temperature=0.0),
                                                    goal=goal, start=start)
    return pred_policy_img