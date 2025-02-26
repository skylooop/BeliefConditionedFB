import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import shapely
from shapely import Point, GeometryCollection
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from functools import partial
import jax

def get_canvas_image(canvas):
    canvas.draw() 
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(canvas.get_width_height()[::-1] + (3,))
    return out_image

def policy_image_fourrooms(env, dataset, N, M, action_fn=None, **kwargs):
    fig = plt.figure(tight_layout=True)
    canvas = FigureCanvas(fig)
    plot_policy(env, dataset, N, M, fig=fig, ax=plt.gca(), action_fn=action_fn, **kwargs)
    image = get_canvas_image(canvas)
    plt.close(fig)
    return image

def plot_policy(env, dataset, N=14, M=20, fig=None, ax=None, random=False, title=None, action_fn=None, **kwargs):
    action_names = [
            r'$\uparrow$', r'$\downarrow$', r'$\leftarrow$', r'$\rightarrow$'# r'$\cdot$'
        ]
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    
    # TODO: fix
    coverage_map = np.where(env.maze.maze_grid == 1, -1000, env.maze.maze_grid)
    ax = env.plot_grid(ax=ax)
    for (x, y), value in np.ndenumerate(coverage_map):
        if value == 0:
            action = action_fn(np.concatenate([[x], [y]], -1)).squeeze()
            action_name = action_names[action]
            plt.text(x, y, action_name, ha='center', va='center', fontsize='large', color='green')
 
    goal = kwargs.get('goal', None)
    start = kwargs.get('start', None)
    if goal is not None:
        ax.set_title('Goal: ({:.2f}, {:.2f})'.format(goal[0], goal[1])) 
        ax.scatter(goal[1], goal[0], s=80, c='black', marker='*')

    if start is not None:
        ax.scatter(start[1], start[0], s=80, c='orange', marker='o')
        
    if title:
        ax.set_title(title)
        
    return fig, ax
def value_image_fourrooms(env, dataset, value_fn, N, M, action_fn=None, **kwargs):
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
    plot_value_image_fourrooms(env, dataset, value_fn, N, M, fig=fig, ax=plt.gca(), action_fn=action_fn, **kwargs)
    image = get_canvas_image(canvas)
    plt.close(fig)
    return image

def plot_value_image_fourrooms(env, dataset, value_fn, N=11, M=11, fig=None, ax=None, title=None, **kwargs):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    
    coverage_map = np.where(env.maze.maze_grid == 1, -1000, env.maze.maze_grid)
    for (x, y), value in np.ndenumerate(coverage_map):
        if value == 0:
            coverage_map[x, y] = jax.device_get(value_fn(np.concatenate([[x], [y]], -1)).max(-1)[0])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(coverage_map, cmap='inferno', vmin=-200)
    fig.colorbar(im, cax=cax, orientation='vertical')
    goal = kwargs.get('goal', None)
    if goal is not None:
        ax.set_title('Goal: ({:.2f}, {:.2f})'.format(goal[0], goal[1])) 
        ax.scatter(goal[1], goal[0], s=80, c='black', marker='*')
    return fig, ax