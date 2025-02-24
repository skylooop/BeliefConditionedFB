import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import shapely
from shapely import Point, GeometryCollection
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from functools import partial

def get_canvas_image(canvas):
    canvas.draw() 
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(canvas.get_width_height()[::-1] + (3,))
    return out_image

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
        
    coverage_map = np.where(env.maze_state == 1, -1000, env.maze_state)
    
    base_observation = np.copy(dataset['observations'][0])
    base_observations = np.tile(base_observation, (N * M, 1))
    values = value_fn(base_observations)
    x = x.reshape(N, M)
    y = y.reshape(N, M)

    values = values.reshape(N, M)
    mesh = ax.pcolormesh(x, y, values, cmap='viridis')
    plt.imshow(coverage_map, cmap='inferno', vmin=0)
    ax = plt.gca()
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(mesh, cax=cax, orientation='vertical')
    goal = kwargs.get('goal', None)
    if goal is not None:
        ax.set_title('Goal: ({:.2f}, {:.2f})'.format(goal[0], goal[1])) 
        ax.scatter(goal[0], goal[1], s=80, c='black', marker='*')
    
    