import os
import sys
os.environ['MUJOCO_GL']='egl'
os.environ['CUDA_VISIBLE_DEVICES']='0'

# import shutup
# shutup.please()

import rootutils
ROOT = rootutils.setup_root(search_from=__file__, cwd=True, pythonpath=True, indicator='requirements.txt')

import random
import time
from rich.pretty import pprint
from functools import partial
import hydra
from omegaconf import OmegaConf, DictConfig
import functools

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['seaborn-v0_8-colorblind', 'seaborn-v0_8-notebook'])

from tqdm.auto import tqdm
import wandb
from absl import app, flags
from collections import defaultdict
from colorama import Fore, Style

from agents import agents
from envs.env_utils import make_env_and_datasets
from sklearn.manifold import TSNE

from utils.datasets import Dataset, ReplayBuffer, GCDataset
from utils.evaluation import evaluate, evaluate_fourrooms, flatten, supply_rng
from utils.log_utils import CsvLogger, get_exp_name, get_wandb_video, setup_wandb
from envs.ogbench.ant_utils import policy_image, value_image
from envs.custom_mazes.env_utils import value_image_fourrooms, policy_image_fourrooms
from envs.minigrid.env_utils import value_image_doors, policy_image_doors, doors_value_fn,\
    random_exploration, image_mdps
from envs.minigrid.doors_grid import MinigridWrapper

FLAGS = flags.FLAGS
flags.DEFINE_bool('disable_jit', False, 'Whether to disable JIT compilation.')

from functools import partial
from utils.evaluation import supply_rng
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from mpl_toolkits.axes_grid1 import make_axes_locatable

def value_image_fourrooms(env, dataset, value_fn, action_fn=None, **kwargs):
    fig = plt.figure(tight_layout=True)
    canvas = FigureCanvas(fig)
    plot_value_image_grid(env, dataset, value_fn, fig=fig, ax=plt.gca(), action_fn=action_fn, **kwargs)
    image = get_canvas_image(canvas)
    plt.close(fig)
    return image

def value_fn(agent, obs, goal, action, mdp_type=None, dynamics_embedding=None):
    q1, q2 = agent.network.select('critic')(obs, goal, action, mdp_num=mdp_type, dynamics_embedding=dynamics_embedding)
    q = jnp.minimum(q1, q2)
    return q / 0.02

def plot_value_image_grid(env, dataset, value_fn, action_fn, fig=None, ax=None, title=None, **kwargs):
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

def visualize_value_image(env, config, agent, key, example_batch, layout_type):
    mdp_type = np.zeros((config['agent']['number_of_meta_envs'], ))
    if mdp_type is not None:
        if layout_type == 0:
            mdp_type[0] = 1
        elif layout_type == 1:
            mdp_type[1] = 1
        else:
            mdp_type[2] = 1
    
    env._gen_grid = partial(env._gen_grid, layout_type=layout_type)
    env = MinigridWrapper(env)
    dynamics_embedding=None
    obs, info = env.reset()
    goal = info.get("goal_pos", None)
    
    if config['agent']['use_context']:
        dataset_inference, env = random_exploration(env, num_episodes=1, layout_type=layout_type, num_mdp=config['agent']['number_of_meta_envs'])
        dynamics_embedding_mean, dynamics_mean_std = agent.network.select('dynamic_transformer')(dataset_inference['observations'][None], dataset_inference['actions'][None, :, None],
                                                                                    dataset_inference['next_observations'][None], train=False, return_embedding=True)
        dynamics_embedding = dynamics_embedding_mean + jax.random.normal(key=key, shape=dynamics_embedding_mean.shape) * jnp.exp(dynamics_mean_std)
        dynamics_embedding = dynamics_embedding.squeeze()
        mdp_type=None
        
    pred_value_img = value_image_fourrooms(env, example_batch,
                                value_fn=partial(value_fn, agent, mdp_type=mdp_type, dynamics_embedding=dynamics_embedding), action_fn=partial(supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32))),
                                                                    goals=goal, mdp_type=mdp_type, temperature=0.0, dynamics_embedding=dynamics_embedding), goal=goal)
    return pred_value_img

def get_canvas_image(canvas):
    canvas.draw() 
    out_image = np.asarray(canvas.buffer_rgba())
    return out_image

def policy_image_grid(env, dataset, action_fn=None, **kwargs):
    fig = plt.figure(tight_layout=True)
    canvas = FigureCanvas(fig)
    plot_policy(env, dataset, fig=fig, ax=plt.gca(), action_fn=action_fn, **kwargs)
    image = get_canvas_image(canvas)
    plt.close(fig)
    return image

def plot_policy(env, dataset, fig=None, ax=None, title=None, action_fn=None, **kwargs):
    action_names = [
            r'$\leftarrow$', r'$\rightarrow$', r'$\uparrow$', r'$\downarrow$' #'↑', '↓', '←', '→'#
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

def visualize_policy_image(env, agent, key, example_batch, config, layout_type):
    mdp_type = np.zeros((config['agent']['number_of_meta_envs'], )) if config['agent']['number_of_meta_envs'] > 1 else None
    if mdp_type is not None:
        if layout_type == 0:
            mdp_type[0] = 1
        elif layout_type == 1:
            mdp_type[1] = 1
        else:
            mdp_type[2] = 1
    
    env._gen_grid = partial(env._gen_grid, layout_type=layout_type)
    env = MinigridWrapper(env)
    dynamics_embedding=None
    obs, info = env.reset()
    goal = info.get("goal_pos", None)
    
    if config['agent']['use_context']:
        dataset_inference, env = random_exploration(env, num_episodes=1, layout_type=layout_type, num_mdp=config['agent']['number_of_meta_envs'])
        dynamics_embedding_mean, dynamics_mean_std = agent.network.select('dynamic_transformer')(dataset_inference['observations'][None], dataset_inference['actions'][None,:,None],
                                                                                    dataset_inference['next_observations'][None], train=False, return_embedding=True)
        dynamics_embedding = dynamics_embedding_mean + jax.random.normal(key=key, shape=dynamics_embedding_mean.shape) * jnp.exp(dynamics_mean_std)
        dynamics_embedding=dynamics_embedding.squeeze()
        mdp_type=None
        
    pred_policy_img = policy_image_grid(env, example_batch,
                                                    action_fn=partial(supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32))),
                                                                    goals=goal, mdp_type=mdp_type, dynamics_embedding=dynamics_embedding, temperature=0.0),
                                                    goal=goal)
    return pred_policy_img

def concatenate_dicts(dict1, dict2):
    return jax.tree.map(lambda x, y: jnp.concatenate([x, y]), dict1, dict2)

@hydra.main(version_base='1.2', config_name="entry", config_path=str(ROOT) + "/configs")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    os.makedirs(cfg.save_dir, exist_ok=True)
    key = jax.random.key(cfg.seed)
    exp_name = get_exp_name(cfg.seed)

    config = OmegaConf.to_container(cfg, resolve=True)
    pprint(config)
    run = setup_wandb(project='ZeroShotRL', group=config['run_group'], name=exp_name,
                    mode="offline" if FLAGS.disable_jit else "online", config=config, tags=config['tags'])
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(dataset_name=config['env']['env_name'],
                                                                    frame_stack=config['agent']['frame_stack'],
                                                                    action_clip_eps=1e-5 if not config['env']['discrete'] else None,
                                                                    context_len=config['agent']['context_len'] if config['agent']['use_context'] else None)
    dataset_class = {
        'GCDataset': GCDataset,
    }[config['agent']['dataset_class']]
    
    train_dataset = dataset_class(Dataset.create(**train_dataset), config['agent'])
    if val_dataset is not None:
        val_dataset = dataset_class(Dataset.create(**val_dataset), config['agent'])
    
    random.seed(config['seed'])
    np.random.seed(config['seed'])

    example_batch = train_dataset.sample(1)
    agent_class = agents[config['agent']['agent_name']]
    agent = agent_class.create(
        config['seed'],
        example_batch['observations'],
        np.full_like(example_batch['actions'], env.action_space.n - 1),
        config['agent'],
    )

    train_logger = CsvLogger(os.path.join(config['save_dir'], 'train.csv'))
    eval_logger = CsvLogger(os.path.join(config['save_dir'], 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    layout_types = [0, 1, 2]
    
    if config['env']['env_name'] == "doors-dynamics":
        plot_layouts = []
        for i in range(config['agent']['number_of_meta_envs']):
            cur_layout = train_dataset.sample(512, layout_type=i, context_length=config['agent']['context_len'])[1]
            plot_layouts.append(cur_layout)
        
        plot_layouts = functools.reduce(concatenate_dicts, plot_layouts)

        first_layout_indxs = jnp.where(jnp.any((plot_layouts['traj_states'][:, :, 0] == 3.0) & (plot_layouts['traj_states'][:, :, 1] == 6.0), 1))[0]
        first_layout_trajs = jax.tree_util.tree_map(lambda arr: arr[first_layout_indxs], plot_layouts)

        sec_layout_indxs = jnp.where(jnp.any((plot_layouts['traj_states'][:, :, 0] == 3.0) & (plot_layouts['traj_states'][:, :, 1] == 2.0), 1))[0]
        sec_layout_trajs = jax.tree_util.tree_map(lambda arr: arr[sec_layout_indxs], plot_layouts)

        third_layout_indxs = jnp.where(jnp.any((plot_layouts['traj_states'][:, :, 0] == 3.0) & (plot_layouts['traj_states'][:, :, 1] == 1.0), 1))[0]
        third_layout_trajs = jax.tree_util.tree_map(lambda arr: arr[third_layout_indxs], plot_layouts)

        plot_layouts = jax.tree.map(lambda x, y, z: jnp.concatenate([x, y, z]), first_layout_trajs, sec_layout_trajs, third_layout_trajs)
        colors = ['blue'] * first_layout_trajs['traj_actions'].shape[0] + ['red'] * sec_layout_trajs['traj_actions'].shape[0] +\
            ['orange'] * third_layout_trajs['traj_actions'].shape[0]
            
    pbar = tqdm(range(1, config['train_steps'] + 1), colour='green', dynamic_ncols=True, position=0, leave=True)
    for step in pbar:
        key = jax.random.fold_in(key, step)
        if not config['agent']['use_context']:
            batch = train_dataset.sample(config['agent']['batch_size'])
            agent, update_info = agent.update(batch)
        else:
            batch = train_dataset.sample(config['agent']['batch_size'], layout_type=step % config['agent']['number_of_meta_envs'],
                                        context_length=config['agent']['context_len'])[1]
            agent, update_info = agent.update(batch, train_context_embedding=True if step < config['agent']['dyn_encoder_warmup_steps'] else False)
                
        # Log metrics.
        if step % config['log_interval'] == 0 or step == 1:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            train_metrics['time/epoch_time'] = (time.time() - last_time) / config['log_interval']
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=step)
            train_logger.log(train_metrics, step=step)

        # Evaluate agent.
        if step == 1 or step % config['eval_interval'] == 0:
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            
            if 'doors' in config['env']['env_name']:
                if 'gciql' in config['agent']['agent_name']:
                    for layout_type in tqdm(layout_types, leave=False, position=1, colour='blue'):
                        if config['agent']['use_context']:
                            pred_policy_img = visualize_policy_image(env, agent, key, example_batch, config, layout_type=layout_type)
                            pred_value_img = visualize_value_image(env, config, agent, key, example_batch, layout_type=layout_type)
                            
                        eval_metrics[f'draw_Q/draw_value_task_{layout_type}'] = wandb.Image(pred_value_img)
                        eval_metrics[f'draw_policy/draw_policy_task_{layout_type}'] = wandb.Image(pred_policy_img)
                        
                    fig, ax = plt.subplots(figsize=(15, 10))
                    dynamics_embedding_mean, std = agent.network.select('dynamic_transformer')(plot_layouts['traj_states'], plot_layouts['traj_actions'],
                                                                plot_layouts['traj_next_states'], train=False)
                    dynamics_embedding = dynamics_embedding_mean + jax.random.normal(key=key, shape=dynamics_embedding_mean.shape) * jnp.exp(std)
                    tsne = TSNE(random_state=42, perplexity=30).fit_transform(dynamics_embedding)
                    ax.scatter(tsne[:, 0], tsne[:, 1], color=colors)
            
            if 'fourrooms-dynamics' in config['env']['env_name']:
                from envs.custom_mazes.dynamics_utils import visualize_policy, visualize_value_image
                from envs.custom_mazes.darkroom import FourRoomsMazeEnv, Maze
                
                # First random layout
                pred_policy_img = visualize_policy(env, agent, key, train_dataset, layout_type=0, task_num=step % 3)
                pred_value_img = visualize_value_image(env, agent, key, example_batch, layout_type=0, task_num=step % 3)
                
                eval_metrics[f'draw_Q/draw_value_task_{step % 3}_layout_1'] = wandb.Image(pred_value_img)
                eval_metrics[f'draw_policy/draw_policy_task_{step % 3}_layout_1'] = wandb.Image(pred_policy_img)
                
                # Second random layout
                env = FourRoomsMazeEnv(Maze(maze_type='fourrooms_random_layouts'), max_steps=100)
                pred_policy_img = visualize_policy(env, agent, key, train_dataset, layout_type=1, task_num=step % 3) # layout_type is dummy variable
                pred_value_img = visualize_value_image(env, agent, key, example_batch, layout_type=1, task_num=step % 3)
                
                eval_metrics[f'draw_Q/draw_value_task_{step % 3}_layout_2'] = wandb.Image(pred_value_img)
                eval_metrics[f'draw_policy/draw_policy_task_{step % 3}_layout_2'] = wandb.Image(pred_policy_img)
                
            if 'fourrooms-vanilla' in config['env']['env_name']:
                from envs.custom_mazes.darkroom import visualize_value_image, visualize_policy
                
                pred_policy_img = visualize_policy(agent, train_dataset, task_num=step % 3) # just dummy hardcoded tasks to ensure that FB works
                pred_value_img = visualize_value_image(agent, example_batch, task_num=step % 3)
                
                eval_metrics[f'draw_Q/draw_value_task_{step % 3}'] = wandb.Image(pred_value_img)
                eval_metrics[f'draw_policy/draw_policy_task_{step % 3}'] = wandb.Image(pred_policy_img)
                
                for task_id in range(4):
                    env.reset()
                    eval_info, _, _ = evaluate_fourrooms(
                            agent=agent,
                            env=env,
                            task_id=task_id,
                            config=None,
                            num_eval_episodes=1,
                            num_video_episodes=0,
                            video_frame_skip=1,
                            eval_temperature=0.0,
                            eval_gaussian=None
                        )
                    eval_metrics.update(
                        {f'evaluation/task_{task_id}_{k}': v for k, v in eval_info.items() if k != 'total.timesteps'}
                    )
                    for k, v in eval_info.items():
                        overall_metrics[k].append(v)
                            
                for k, v in overall_metrics.items():
                    eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)
                    
            wandb.log(eval_metrics, step=step)
            eval_logger.log(eval_metrics, step=step)
            
    train_logger.close()
    eval_logger.close()
    
def entry(argv):
    sys.argv = argv
    disable_jit = FLAGS.disable_jit
    try:
        if disable_jit:
            with jax.disable_jit():
                main()
        else:
            main()
    except KeyboardInterrupt:
        wandb.finish()
        print(f"{Fore.GREEN}{Style.BRIGHT}Finished!{Style.RESET_ALL}")

if __name__ == "__main__":
    app.run(entry)