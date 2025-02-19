import gymnasium
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import shapely
from shapely import Point, GeometryCollection
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import jax
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from functools import partial


class MazeVizWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)   
        
    # ======== BELOW is helper stuff for drawing and visualizing ======== #
    def get_starting_boundary(self):
        torso_x, torso_y = self.unwrapped._offset_x, self.unwrapped._offset_y
        S =  self.unwrapped._maze_unit
        return (0 - S / 2 + S - torso_x, 0 - S/2 + S - torso_y), (len(self.unwrapped.maze_map[0]) * S - torso_x - S/2 - S, len(self.unwrapped.maze_map) * S - torso_y - S/2 - S)

    def XY(self, n=20, m=30):
        bl, tr = self.get_starting_boundary()
        X = np.linspace(bl[0] + 0.04 * (tr[0] - bl[0]) , tr[0] - 0.04 * (tr[0] - bl[0]), m)
        Y = np.linspace(bl[1] + 0.04 * (tr[1] - bl[1]) , tr[1] - 0.04 * (tr[1] - bl[1]), n)
        
        X,Y = np.meshgrid(X,Y)
        states = np.array([X.flatten(), Y.flatten()]).T
        return states
    
    def draw(self, ax=None, scale=1.0):
        if not ax: ax = plt.gca()
        polyg = []
        torso_x, torso_y = self.unwrapped._offset_x, self.unwrapped._offset_y
        S =  self.unwrapped._maze_unit
        if scale < 1.0:
            S *= 0.965
            torso_x -= 0.7
            torso_y -= 0.95
        for i in range(len(self.unwrapped.maze_map)):
            for j in range(len(self.unwrapped.maze_map[0])):
                struct = self.unwrapped.maze_map[i][j]
                if struct == 1:
                    xy = (j * S - torso_x - S / 2, i * S - torso_y - S / 2)
                    rect = patches.Rectangle((j *S - torso_x - S/ 2,
                                            i * S- torso_y - S/ 2),
                                            S,
                                            S, linewidth=1, facecolor='RosyBrown', alpha=1.0)
                    sharply_pol = shapely.Polygon((xy, (xy[0], xy[1] + S), (xy[0]+S, xy[1]+S), (xy[0] +S, xy[1])))
                    polyg.append(sharply_pol)
                    
                    ax.add_patch(rect)
        
        self.xlims = (0 - S /2 + 0.6 * S - torso_x, len(self.unwrapped.maze_map[0]) * S - torso_x - S/2 - S * 0.6)
        self.ylims = (0 - S/2 + 0.6 * S - torso_y, len(self.unwrapped.maze_map) * S - torso_y - S/2 - S * 0.6)
        ax.set_xlim(*self.xlims)
        ax.set_ylim(*self.ylims)
        # ax.axis('off')
        return polyg
    
    @property
    def get_env_limits(self):
        return self.xlims, self.ylims

def plot_value(env, dataset, value_fn, N=14, M=20, fig=None, ax=None, random=False, title=None, action_fn=None, **kwargs):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    observations = env.XY(n=N, m=M)
    if random:
        base_observations = np.copy(dataset['observations'][np.random.choice(dataset.size, len(observations))])
    else:
        base_observation = np.copy(dataset['observations'][0])
        base_observations = np.tile(base_observation, (observations.shape[0], 1))
    base_observations[:, :2] = observations

    if action_fn is not None:
        actions = action_fn(observations=base_observations)
        values = value_fn(observation=base_observations, action=actions)
    else:
        values = value_fn(base_observations)
    x, y = observations[:, 0], observations[:, 1]
    x = x.reshape(N, M)
    y = y.reshape(N, M) * 0.975 + 0.7
        
    values = values.reshape(N, M)
    mesh = ax.pcolormesh(x, y, values, cmap='viridis')
    env.draw(ax, scale=0.95)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(mesh, cax=cax, orientation='vertical')
    goal = kwargs.get('goal', None)
    if goal is not None:
        ax.set_title('Goal: ({:.2f}, {:.2f})'.format(goal[0], goal[1])) 
        ax.scatter(goal[0], goal[1], s=80, c='yellow', marker='*')
    
    if title:
        ax.set_title(title)

def plot_policy(env, dataset, N=14, M=20, fig=None, ax=None, random=False, title=None, action_fn=None, **kwargs):
    observations = env.XY(n=N, m=M)

    if random:
        base_observations = np.copy(dataset['observations'][np.random.choice(dataset.size, len(observations))])
    else:
        base_observation = np.copy(dataset['observations'][0])
        base_observations = np.tile(base_observation, (observations.shape[0], 1))

    base_observations[:, :2] = observations

    policies = action_fn(base_observations)

    x, y = observations[:, 0], observations[:, 1]
    x = x.reshape(N, M)
    y = y.reshape(N, M) * 0.975 + 0.7

    policy_x = policies[:, 0].reshape(N, M)
    policy_y = policies[:, 1].reshape(N, M)
    mesh = ax.quiver(x, y, policy_x, policy_y, color='r', pivot='mid', scale=0.75, scale_units='xy')
    env.draw(ax, scale=0.95)
    
    goal = kwargs.get('goal', None)
    start = kwargs.get('start', None)
    if goal is not None:
        ax.set_title('Goal: ({:.2f}, {:.2f})'.format(goal[0], goal[1])) 
        ax.scatter(goal[0], goal[1], s=80, c='yellow', marker='*')

    if start is not None:
        ax.scatter(start[0], start[1], s=80, c='green', marker='o')
        
    if title:
        ax.set_title(title)

def plot_trajectories(env, dataset, trajectories, fig, ax, color_list=None):
    if color_list is None:
        from itertools import cycle
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_list = cycle(color_cycle)

    for color, trajectory in zip(color_list, trajectories):        
        obs = np.array(trajectory['observation'])
        all_x = obs[:, 0]
        all_y = obs[:, 1]
        ax.scatter(all_x, all_y, s=5, c=color, alpha=0.02)
        ax.scatter(all_x[-1], all_y[-1], s=50, c=color, marker='*', alpha=0.3)

    env.draw(ax)

def get_canvas_image(canvas):
    canvas.draw() 
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(canvas.get_width_height()[::-1] + (3,))
    return out_image

def gc_sampling_adaptor(policy_fn):
    def f(observations, *args, **kwargs):
        return policy_fn(observations['observation'], observations['goal'], *args, **kwargs)
    return f

def trajectory_image(env, dataset, trajectories, **kwargs):
    fig = plt.figure(tight_layout=True)
    canvas = FigureCanvas(fig)

    plot_trajectories(env, dataset, trajectories, fig, plt.gca(), **kwargs)

    plt.tight_layout()
    image = get_canvas_image(canvas)
    plt.close(fig)
    return image

def value_image(env, dataset, value_fn, N, M, action_fn=None, **kwargs):
    """
    Visualize the value function.
    Args:
        env: The environment.
        value_fn: a function with signature value_fn([# states, state_dim]) -> [#states, 1]
    Returns:
        A numpy array of the image.
    """
    fig = plt.figure(tight_layout=True)
    canvas = FigureCanvas(fig)
    plot_value(env, dataset, value_fn, N, M, fig=fig, ax=plt.gca(), action_fn=action_fn, **kwargs)
    image = get_canvas_image(canvas)
    plt.close(fig)
    return image

def policy_image(env, dataset, N, M, action_fn=None, **kwargs):
    """
    Visualize a 2d representation of a policy.

    Args:
        env: The environment.
        policy_fn: a function with signature policy_fn([# states, state_dim]) -> [#states, 2]
    Returns:
        A numpy array of the image.
    """
    fig = plt.figure(tight_layout=True)
    canvas = FigureCanvas(fig)
    plot_policy(env, dataset, N, M, fig=fig, ax=plt.gca(), action_fn=action_fn, **kwargs)
    image = get_canvas_image(canvas)
    plt.close(fig)
    return image


def most_squarelike(n):
    c = int(n ** 0.5)
    while c > 0:
        if n %c in [0 , c-1]:
            return (c, int(math.ceil(n / c)))
        c -= 1

def make_visual(env, dataset, methods):
    
    h, w = most_squarelike(len(methods))
    gs = gridspec.GridSpec(h, w)

    fig = plt.figure(tight_layout=True)
    canvas = FigureCanvas(fig)

    for i, method in enumerate(methods):
        wi, hi = i % w, i // w
        ax = fig.add_subplot(gs[hi, wi])
        method(env, dataset, fig=fig, ax=ax)

    plt.tight_layout()
    image = get_canvas_image(canvas)
    plt.close(fig)
    return image

def gcvalue_image(env, dataset, value_fn):
    """
    Visualize the value function for a goal-conditioned policy.

    Args:
        env: The environment.
        value_fn: a function with signature value_fn(goal, observations) -> values
    """
    base_observation = dataset['observations'][0]

    point1, point2, point3, point4 = env.four_goals()
    point3 = (32.75, 24.75)

    fig = plt.figure(tight_layout=True)
    canvas = FigureCanvas(fig)

    points = [point1, point2, point3, point4]
    for i, point in enumerate(points):
        point = np.array(point)
        ax = fig.add_subplot(2, 2, i + 1)

        goal_observation = base_observation.copy()
        goal_observation[:2] = point

        plot_value(env, dataset, partial(value_fn, goal_observation), fig, ax)

        ax.set_title('Goal: ({:.2f}, {:.2f})'.format(point[0], point[1])) 
        ax.scatter(point[0], point[1], s=50, c='red', marker='*')

    image = get_canvas_image(canvas)
    plt.close(fig)
    return image