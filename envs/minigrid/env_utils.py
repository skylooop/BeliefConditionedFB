import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.colors import LinearSegmentedColormap

def random_exploration(env, num_episodes: int, layout_type: int):
    dataset = dict()
    observations = []
    actions = []
    dones = []

    # available_actions = np.array([0, 1, 2, 3])
    for _ in range(num_episodes):
        env.reset()
        cur_observations = []
        cur_actions = []
        cur_dones = []
        done = False
        while not done:
            cur_observations.append(np.array(env.env.unwrapped.agent_pos, dtype=np.float32))
            #action = np.random.choice(available_actions, replace=True)
            action = env.env.unwrapped.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            cur_actions.append(np.array(action, dtype=np.float32))
            done = truncated
            cur_dones.append(np.array(done, dtype=np.float32))

        observations.append(np.stack(cur_observations))
        actions.append(np.stack(cur_actions))
        dones.append(np.stack(cur_dones))
    
    dataset['observations'] = np.concatenate(observations)
    dataset['terminals'] = np.concatenate(dones)
    dataset['actions'] = np.concatenate(actions)
    
    ob_mask = (1.0 - dataset['terminals']).astype(bool)
    next_ob_mask = np.concatenate([[False], ob_mask[:-1]])
    dataset['next_observations'] = dataset['observations'][next_ob_mask]
    dataset['observations'] = dataset['observations'][ob_mask]
    dataset['actions'] = dataset['actions'][ob_mask].astype(np.int8)
    new_terminals = np.concatenate([dataset['terminals'][1:], [1.0]])
    dataset['terminals'] = new_terminals[ob_mask].astype(np.float32)
    dataset['layout_type'] = np.repeat(np.array(layout_type), repeats=(dataset['actions'].shape[0], ))
    return dataset, env

def get_canvas_image(canvas):
    canvas.draw() 
    out_image = np.asarray(canvas.buffer_rgba())
    return out_image

def policy_image_doors(env, dataset, action_fn=None, **kwargs):
    fig = plt.figure(tight_layout=True)
    canvas = FigureCanvas(fig)
    plot_policy(env, dataset, fig=fig, ax=plt.gca(), action_fn=action_fn, **kwargs)
    image = get_canvas_image(canvas)
    plt.close(fig)
    return image

def plot_policy(env, dataset, fig=None, ax=None, title=None, action_fn=None, **kwargs):
    action_names = [
            r'$\leftarrow$', r'$\rightarrow$', r'$\uparrow$', r'$\downarrow$'
        ]
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    
    grid = env.get_grid_array()
    ax = env.plot_grid(ax=ax, grid=grid)
    
    goal = kwargs.get('goal', None)
    for (y, x), value in np.ndenumerate(grid):
        if value == 1 or value == 4:
            action = action_fn(np.concatenate([[x], [y]], -1)).squeeze()
            action_name = action_names[action]
            ax.text(x, y, action_name, ha='center', va='center', fontsize='large', color='green')
            
    ax.set_title('Goal: ({:.2f}, {:.2f})'.format(goal[0], goal[1])) 
    ax.scatter(goal[0], goal[1], s=80, c='black', marker='*')
        
    if title:
        ax.set_title(title)
        
    return fig, ax

def plot_pca(embs, fig, ax):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    # Make PCA & TSNE
    pca = PCA()
    tsne = TSNE(random_state=42)
    cmap = LinearSegmentedColormap.from_list('recent_to_old', ['cyan', 'black'])
    indices = np.linspace(0, 1, len(embs))  # Normalized indices for colormap
    colors = cmap(indices)
    for idx, (reduction, name) in enumerate(zip([pca, tsne], ['PCA', 'TSNE'])):
        ax = fig.add_subplot(1, 2, idx + 1)
        projected = reduction.fit_transform(embs)
        ax.scatter(projected[:, 0], projected[:, 1], c = colors)
        ax.set_title(f"{name}")
    plt.tight_layout()

def plot_image_pcas(embs):
    fig = plt.figure(tight_layout=True, figsize=(20, 10))
    canvas = FigureCanvas(fig)
    plot_pca(embs, fig=fig, ax=plt.gca())
    image = get_canvas_image(canvas)
    plt.close(fig)
    return image

def plot_image_mdps(dataset, fig, ax, embedding_fn, context_len):
    for layout_type, name, color in zip([0, 3, 6], ['MDP1', 'MDP3', 'MDP6'], ['red', 'green', 'blue']):
        batch, batch_context, _ = dataset.sample(128, layout_type=layout_type,
                                                                    context_length=context_len)
        dynamics_embedding, _ = embedding_fn(batch_context['observations'], batch_context['actions'], batch_context['next_observations'])
        pca = PCA().fit_transform(dynamics_embedding)
        ax.scatter(pca[:, 0], pca[:, 1], label=name, c=color)
        
    plt.legend()
    return fig, ax

def image_mdps(dataset, embedding_fn, context_len): # get embeddings from context encoder for different MDPs
    fig = plt.figure(tight_layout=True)
    canvas = FigureCanvas(fig)
    plot_image_mdps(dataset, fig=fig, ax=plt.gca(), embedding_fn=embedding_fn, context_len=context_len)
    image = get_canvas_image(canvas)
    plt.close(fig)
    return image

def value_image_doors(env, dataset, value_fn, action_fn=None, **kwargs):
    fig = plt.figure(tight_layout=True)
    canvas = FigureCanvas(fig)
    plot_value_image_doors(env, dataset, value_fn, fig=fig, ax=plt.gca(), action_fn=action_fn, **kwargs)
    image = get_canvas_image(canvas)
    plt.close(fig)
    return image

def doors_value_fn(agent, obs, goal, action, dynamics_embedding=None):
    q1, q2 = agent.network.select('critic')(obs, goal, action, dynamics_embedding=dynamics_embedding)
    q = jnp.minimum(q1, q2)
    return q / 0.03

def plot_value_image_doors(env, dataset, value_fn, action_fn, fig=None, ax=None, title=None, **kwargs):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    
    goal = kwargs.get('goal', None)
    grid = env.get_grid_array()
    ax = env.plot_grid(ax=ax, grid=grid)
    for (y, x), value in np.ndenumerate(grid):
        if value == 1 or value == 4:
            action = action_fn(np.concatenate([[x], [y]], -1)).squeeze()
            grid[y, x] = jax.device_get(value_fn(np.concatenate([[x], [y]], -1), goal, action))
            
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(grid, cmap='inferno', vmin=-1000)
    fig.colorbar(im, cax=cax, orientation='vertical')
    if goal is not None:
        ax.set_title('Goal: ({:.2f}, {:.2f})'.format(goal[0], goal[1])) 
        ax.scatter(goal[0], goal[1], s=80, c='black', marker='*')
    return fig, ax