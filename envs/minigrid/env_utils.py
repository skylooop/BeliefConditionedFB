import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import math
import functools
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec as gridspec
from collections import defaultdict
from utils.evaluation import flatten, add_to
from tqdm.auto import tqdm


plt.style.use(['seaborn-v0_8-colorblind'])

def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def convert_trajs_to_dict(trajs, pad_value=-1.0):
    """Convert list of trajectory dicts to a single dict with stacked arrays.
    
    Args:
        trajs: List of dictionaries, each containing trajectory data.
        pad_value: Value to use for padding shorter trajectories.
        
    Returns:
        Dictionary with stacked arrays of shape (num_trajectories, max_traj_length, dim).
    """
    if not trajs:
        return {}
    
    # Determine maximum trajectory length
    max_length = max(len(traj['observations']) for traj in trajs)
    
    # Initialize output dictionary
    result = defaultdict(list)
    
    # Get all field names from the first trajectory
    field_names = trajs[0].keys()
    
    for traj in trajs:
        traj_length = len(traj['observations'])
        pad_length = max_length - traj_length
        
        for field in field_names:
            # Get the array for this field
            arr = np.array(traj[field])
            
            # Pad if necessary
            if pad_length > 0:
                if arr.ndim == 1:
                    # For 1D arrays (like rewards, done)
                    arr = np.pad(arr, (0, pad_length), constant_values=pad_value)
                else:
                    # For multi-dimensional arrays (like observations, actions)
                    pad_width = [(0, pad_length)] + [(0, 0)] * (arr.ndim - 1)
                    arr = np.pad(arr, pad_width, constant_values=pad_value)
            
            result[field].append(arr)
    
    # Stack all trajectories for each field
    for field in result:
        result[field] = np.stack(result[field])
    
    return dict(result)

def random_exploration_fourrooms(env, num_episodes: int, layout_type: int, num_mdp: int):
    dataset = dict()
    observations = []
    actions = []
    dones = []
    next_observations = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        cur_observations = []
        cur_actions = []
        cur_dones = []
        cur_next_observations = []
        done = False
        step = 0
        while not done and step < env.max_steps:
            step += 1
            cur_observations.append(np.array(obs, dtype=np.float32))
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            cur_next_observations.append(np.array(next_state, dtype=np.float32))
            cur_actions.append(np.array(action, dtype=np.float32))
            done = truncated# or terminated
            obs = next_state
            cur_dones.append(np.array(done, dtype=np.float32))
            
        observations.append(np.stack(cur_observations)) # seq_len x dim
        actions.append(np.stack(cur_actions))
        dones.append(np.stack(cur_dones))
        next_observations.append(np.stack(cur_next_observations))
        
    dataset['observations'] = np.concatenate(observations)
    dataset['terminals'] = np.concatenate(dones)
    dataset['actions'] = np.concatenate(actions)
    dataset['next_observations'] = np.concatenate(next_observations)
    
    ob_mask = (1.0 - dataset['terminals']).astype(bool)
    next_ob_mask = np.concatenate([[False], ob_mask[:-1]])
    # dataset['next_observations'] = dataset['observations'][next_ob_mask]
    dataset['next_observations'] = dataset['next_observations'][ob_mask]
    dataset['observations'] = dataset['observations'][ob_mask]
    dataset['actions'] = dataset['actions'][ob_mask].astype(np.int8)
    new_terminals = np.concatenate([dataset['terminals'][1:], [1.0]])
    dataset['terminals'] = new_terminals[ob_mask].astype(np.float32)
    dataset['layout_type'] = np.tile(one_hot(np.array(layout_type), num_mdp), reps=(dataset['actions'].shape[0], 1))
    return dataset, env

def random_exploration(env, num_episodes: int, layout_type: int, num_mdp: int):
    dataset = dict()
    observations = []
    actions = []
    dones = []
    next_observations = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        cur_observations = []
        cur_actions = []
        cur_dones = []
        cur_next_observations = []
        done = False
        step = 0
        cur_action_num = 0
        while not done and step < env.env.unwrapped.max_steps:
            step += 1
            cur_observations.append(np.array(obs, dtype=np.float32))
            #if cur_action_num > 4 or step == 1:
            action = env.action_space.sample()
            #    cur_action_num = 0
            #cur_action_num += 1
            next_state, reward, terminated, truncated, info = env.step(action)
            cur_next_observations.append(np.array(next_state, dtype=np.float32))
            cur_actions.append(np.array(action, dtype=np.float32))
            done = truncated# or terminated
            obs = next_state
            cur_dones.append(np.array(done, dtype=np.float32))
            
        observations.append(np.stack(cur_observations)) # seq_len x dim
        actions.append(np.stack(cur_actions))
        dones.append(np.stack(cur_dones))
        next_observations.append(np.stack(cur_next_observations))
        
    dataset['observations'] = np.concatenate(observations)
    dataset['terminals'] = np.concatenate(dones)
    dataset['actions'] = np.concatenate(actions)
    dataset['next_observations'] = np.concatenate(next_observations)
    
    ob_mask = (1.0 - dataset['terminals']).astype(bool)
    next_ob_mask = np.concatenate([[False], ob_mask[:-1]])
    # dataset['next_observations'] = dataset['observations'][next_ob_mask]
    dataset['next_observations'] = dataset['next_observations'][ob_mask]
    dataset['observations'] = dataset['observations'][ob_mask]
    dataset['actions'] = dataset['actions'][ob_mask].astype(np.int8)
    new_terminals = np.concatenate([dataset['terminals'][1:], [1.0]])
    dataset['terminals'] = new_terminals[ob_mask].astype(np.float32)
    dataset['layout_type'] = np.tile(one_hot(np.array(layout_type), num_mdp), reps=(dataset['actions'].shape[0], 1))
    return dataset, env

def q_learning_fourrooms(env, num_episodes: int, layout_type: int, alpha=0.1, gamma=0.99, epsilon=0.7, num_mdp=1):
    Q = np.zeros((env.maze.size[0], env.maze.size[1], env.action_space.n))
    dataset = dict()
    observations = []
    actions = []
    dones = []
    next_observations = []
    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        cur_observations = []
        cur_actions = []
        cur_dones = []
        cur_next_observations = []
        
        while not done:
            cur_observations.append(np.array(obs, dtype=np.float32))
            if (np.random.rand() < epsilon) or step < 50:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[obs[0], obs[1], :])
            next_state, reward, terminated, truncated, info = env.step(action)
            done = truncated
            cur_actions.append(np.array(action, dtype=np.float32))
            Q[obs[0], obs[1], action] += alpha * (reward + gamma * np.max(Q[next_state[0], next_state[1], :]) - Q[obs[0], obs[1], action])
            obs = next_state
            cur_dones.append(np.array(done, dtype=np.float32))
            cur_next_observations.append(np.array(next_state))
            step+=1
            
        observations.append(np.stack(cur_observations))
        actions.append(np.stack(cur_actions))
        dones.append(np.stack(cur_dones))
        next_observations.append(np.stack(cur_next_observations))
        
    dataset['observations'] = np.concatenate(observations)
    dataset['terminals'] = np.concatenate(dones)
    dataset['actions'] = np.concatenate(actions)
    dataset['next_observations'] = np.concatenate(next_observations)
    
    ob_mask = (1.0 - dataset['terminals']).astype(bool)
    next_ob_mask = np.concatenate([[False], ob_mask[:-1]])
    # dataset['next_observations'] = dataset['observations'][next_ob_mask]
    dataset['next_observations'] = dataset['next_observations'][ob_mask]
    dataset['observations'] = dataset['observations'][ob_mask]
    dataset['actions'] = dataset['actions'][ob_mask].astype(np.int8)
    new_terminals = np.concatenate([dataset['terminals'][1:], [1.0]])
    dataset['terminals'] = new_terminals[ob_mask].astype(np.float32)
    dataset['layout_type'] = np.tile(one_hot(np.array(layout_type), num_mdp), reps=(dataset['actions'].shape[0], 1))
    return dataset, env


def q_learning(env, num_episodes: int, layout_type: int, alpha=0.1, gamma=0.99, epsilon=0.7, num_mdp=1):
    Q = np.zeros((env.env.unwrapped.width, env.env.unwrapped.height, env.env.unwrapped.action_space.n))
    dataset = dict()
    observations = []
    actions = []
    dones = []
    next_observations = []
    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        cur_observations = []
        cur_actions = []
        cur_dones = []
        cur_next_observations = []
        
        while not done:
            cur_observations.append(np.array(obs, dtype=np.float32))
            if (np.random.rand() < epsilon) or step < 50:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[obs[0], obs[1], :])
            next_state, reward, terminated, truncated, info = env.step(action)
            done = truncated
            cur_actions.append(np.array(action, dtype=np.float32))
            Q[obs[0], obs[1], action] += alpha * (reward + gamma * np.max(Q[next_state[0], next_state[1], :]) - Q[obs[0], obs[1], action])
            obs = next_state
            cur_dones.append(np.array(done, dtype=np.float32))
            cur_next_observations.append(np.array(next_state))
            step+=1
            
        observations.append(np.stack(cur_observations))
        actions.append(np.stack(cur_actions))
        dones.append(np.stack(cur_dones))
        next_observations.append(np.stack(cur_next_observations))
        
    dataset['observations'] = np.concatenate(observations)
    dataset['terminals'] = np.concatenate(dones)
    dataset['actions'] = np.concatenate(actions)
    dataset['next_observations'] = np.concatenate(next_observations)
    
    ob_mask = (1.0 - dataset['terminals']).astype(bool)
    next_ob_mask = np.concatenate([[False], ob_mask[:-1]])
    # dataset['next_observations'] = dataset['observations'][next_ob_mask]
    dataset['next_observations'] = dataset['next_observations'][ob_mask]
    dataset['observations'] = dataset['observations'][ob_mask]
    dataset['actions'] = dataset['actions'][ob_mask].astype(np.int8)
    new_terminals = np.concatenate([dataset['terminals'][1:], [1.0]])
    dataset['terminals'] = new_terminals[ob_mask].astype(np.float32)
    dataset['layout_type'] = np.tile(one_hot(np.array(layout_type), num_mdp), reps=(dataset['actions'].shape[0], 1))
    return dataset, env

def collect_belief_env(env, num_episodes):
    max_episode_length = env.max_steps
    trajs = []

    for i in tqdm(range(num_episodes)):
        traj = defaultdict(list)

        observation, info = env.reset()
        done = False
        step = 0
        state = env.get_state()
        item_location = env.item_location
        
        while not done and step <= max_episode_length:
            action = env.action_space.sample()
            cur_state = state
            action = np.array(action)

            next_observation, reward, done, info = env.step(action)
            step += 1
    
            transition = dict(
                observations=observation['image'].reshape(-1),
                next_observations=next_observation['image'].reshape(-1),
                item_location=item_location,
                state=cur_state['image'].reshape(-1),
                actions=action[None],
                reward=reward,
                done=done,
            )
            state = env.get_state()
            observation = next_observation
            add_to(traj, transition)
            
        if i <= num_episodes:
            trajs.append(traj)
    return trajs

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

def plot_image_mdps(dataset, fig, ax, embedding_fn, method, colors):
    dynamics_embedding, _ = embedding_fn(dataset['observations'], dataset['actions'], dataset['next_observations'])
    projection = method().fit_transform(dynamics_embedding)
    ax.scatter(projection[:, 0], projection[:, 1], c=colors)
    
    return fig, ax

def most_squarelike(n):
    c = int(n ** 0.5)
    while c > 0:
        if n %c in [0 , c-1]:
            return (c, int(math.ceil(n / c)))
        c -= 1
        
def image_mdps(dataset, embedding_fn, colors): # get embeddings from context encoder for different MDPs
    h, w = most_squarelike(2)
    gs = gridspec.GridSpec(h, w)
    fig = plt.figure(tight_layout=True)
    canvas = FigureCanvas(fig)
    for i, (method, method_name) in enumerate(zip([PCA, functools.partial(TSNE, random_state=42)], ['PCA', 'TSNE'])):
        wi, hi = i % w, i // w
        ax = fig.add_subplot(gs[hi, wi])
        plot_image_mdps(dataset, fig=fig, ax=ax, embedding_fn=embedding_fn, method=method, colors=colors)
        ax.set_title(method_name)
        
    plt.tight_layout()
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
    return q / 0.02

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