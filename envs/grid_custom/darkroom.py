import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt

def all_goals(grid_size):
    goals = np.mgrid[0:grid_size, 0:grid_size].reshape(2, -1).T
    return goals

def grid_types(name):
    if name == "fourroom":
        goal_list = np.array([(3, 7), (2, 6), (7, 6), (8, 8), (7, 4), (7, 2), (2, 2), (4, 2), (4, 7), (1, 7)])
        return np.array([
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                [-1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1],
                [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]), goal_list
    elif name == "simple":
        goal_list = np.array([(2, 7), (3, 7), (7, 2), (2, 2), (4, 2), (2, 6), (1, 2), (7, 5), (3, 6)])
        return np.array(
            [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, -1, -1, 0, 0, 0, -1],
                [-1, 0, 0, 0, -1, -1, 0, 0, 0, -1],
                [-1, 0, 0, 0, -1, -1, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
        ), goal_list

    elif name == "obstacle":
        goal_list = np.array([(2, 8)])
        return np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
              [-1, 0, 0, 0, 0, 0, -1, 0, 0, -1],
              [-1, 0, 0, 0, -1, 0, 0, 0, 0, -1],
              [-1, 0, 0, 0, -1, -1, 0, 0, 0, -1],
              [-1, 0, 0, 0, -1, -1, 0, 0, 0, -1],
              [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
              [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
              [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
              [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]), goal_list
    
class DarkRoom(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 1}
    
    def __init__(self, goal=None, grid_name='fourroom', random_start=True, terminate_on_goal=True, render_mode="rgb_array"):
        self.agent_pos = None
        self._layout, self.goal_list = grid_types(grid_name)        
        #self.goal_list = np.array([(3, 7), (2, 6), (7, 6), (8, 8), (7, 4), (7, 2), (2, 2), (4, 2), (4, 7), (1, 7)])
        
        if goal is not None:
            self.goal_pos = np.asarray(goal)
            assert self.goal_pos.ndim == 1
        else:
            self.goal_pos = self.generate_goal()
        
        self.goal_state = self.pos_to_state(self.goal_pos)
        
        self.observation_space = spaces.Discrete(self.size**2)
        self.action_space = spaces.Discrete(5)
        
        self.action_to_direction = {
                    0: np.array((0, 0), dtype=np.float32),  # noop
                    1: np.array((-1, 0), dtype=np.float32),  # up
                    2: np.array((0, 1), dtype=np.float32),  # right
                    3: np.array((1, 0), dtype=np.float32),  # down
                    4: np.array((0, -1), dtype=np.float32),  # left
                }
        self.center_pos = (self.size // 2, self.size // 2)
        self.terminate_on_goal = terminate_on_goal
        self.render_mode = render_mode
        self.random_start = random_start
    
    @property
    def size(self):
        return self._layout.shape[-1]
    
    def generate_pos(self):
        return self.np_random.choice(np.where(self._layout.flatten() == 0)[0])

    def generate_goal(self):
        return self.np_random.choice(self.goal_list)

    def pos_to_state(self, pos):
        return int(pos[0] * self.size + pos[1])

    def state_to_pos(self, state):
        return np.array(divmod(state, self.size))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if self.random_start:
            self.agent_pos = self.generate_pos()
        else:
            self.agent_pos = np.array(self.center_pos, dtype=np.float32)
        self.start_pos = self.state_to_pos(self.agent_pos)
        self.agent_pos = self.start_pos
        return self.agent_pos, {}

    def step(self, action):
        next_pos_x, next_pos_y = np.clip(self.agent_pos + self.action_to_direction[action], 0, self.size - 1).astype(np.int8)
        print(self.agent_pos)
        print(self.action_to_direction[action])
        if self._layout[next_pos_x, next_pos_y] == -1: # wall
            next_pos = self.agent_pos
        else:
            next_pos = np.array((next_pos_x, next_pos_y))
        self.agent_pos = next_pos
        print(self.agent_pos)
        reward = 1.0 if np.array_equal(self.agent_pos, self.goal_pos) else 0.0
        terminated = True if reward and self.terminate_on_goal else False
        return self.pos_to_state(self.agent_pos), reward, terminated, False, {}

    def render(self):
        ax = self.plot_grid(add_start=True)
        print(self.agent_pos)
        # Add the agent location
        render_y = self._layout.shape[0] - 1 - self.agent_pos[1]
        plt.text(
            self.agent_pos[1],
            self.agent_pos[0],
            'A',
            fontsize=18,
            ha='center',
            va='center',)

    
    def plot_grid(self, add_start=True):
        asbestos = (0, 0, 0, 0.6)
        dodger_blue = (25 / 255, 140 / 255, 255 / 255, 0.8)
        grid_kwargs = {'color': (220 / 255, 220 / 255, 220 / 255, 0.5)}
        plt.figure(figsize=(4, 4))
        img = np.ones((self._layout.shape[0], self._layout.shape[1], 4))
        wall_y, wall_x = np.where(self._layout <= -1)
        for i in range(len(wall_y)):
            img[wall_y[i], wall_x[i]] = np.array(asbestos)

        plt.imshow(img, interpolation=None)
        ax = plt.gca()
        ax.grid(0)
        plt.xticks([])
        plt.yticks([])
        # Add start/goal
        render_y = self._layout.shape[0] - 1 - self.start_pos[1]
        if add_start:
            plt.text(
            self.start_pos[1],
            self.start_pos[0],
            r'$\mathbf{S}$',
            fontsize=16,
            ha='center',
            va='center')
        render_y = self._layout.shape[0] - 1 - self.goal_pos[1]
        plt.text(
            self.goal_pos[0],
            self.goal_pos[1],
            r'$\mathbf{G}$',
            fontsize=16,
            ha='center',
            va='center',
            color=dodger_blue)
        h, w = self._layout.shape
        for y in range(h - 1):
            plt.plot([-0.5, w - 0.5], [y + 0.5, y + 0.5], **grid_kwargs)
        for x in range(w - 1):
            plt.plot([x + 0.5, x + 0.5], [-0.5, h - 0.5], **grid_kwargs)
        return ax
    
    def plot_policy_from_list(self, obs_list, act_list):
        print('Plotting policy')
        action_names = [
            r'$\uparrow$', r'$\rightarrow$', r'$\downarrow$', r'$\leftarrow$', r'$\cdot$'
        ]
        self.plot_grid()
        # plt.title(title)
        for i, obs in enumerate(obs_list):
            y, x = self.state_to_pos(obs)
            action_name = action_names[act_list[i]]
            plt.text(x, y, action_name, ha='center', va='center', fontsize='large', color='green')
            
    def plot_v_function(self, obs_list, v_list, act_list):
        print('Plotting V function')
        action_names = [
            r'$\uparrow$', r'$\rightarrow$', r'$\downarrow$', r'$\leftarrow$', r'$\cdot$'
        ]
        self.plot_grid()
        h, w = self._layout.shape
        VMIN = -1000
        v_map = np.zeros((h, w)) + VMIN
        for i, obs in enumerate(obs_list):
            # print(obs)
            y, x = self.state_to_pos(obs)
            
            action_name = action_names[act_list[i]]
            plt.text(x, y, action_name, ha='center', va='center', fontsize='large', color='green')
            v_map[y, x] = v_list[i]
            # v_min = np.min(v_list)
            if y==0 or y==h or x==0 or x==h:
                v_map[y, x] = -1000
        plt.imshow(v_map, cmap='magma', interpolation='nearest')