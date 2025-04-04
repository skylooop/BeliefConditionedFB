import os
import sys
os.environ['MUJOCO_GL']='egl'
# os.environ['CUDA_VISIBLE_DEVICES']='0'

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

from utils.datasets import Dataset, ReplayBuffer, GCDataset
from utils.evaluation import evaluate, evaluate_fourrooms, flatten, supply_rng
from utils.log_utils import CsvLogger, get_exp_name, get_wandb_video, setup_wandb
from envs.ogbench.ant_utils import policy_image, value_image
from envs.custom_mazes.env_utils import value_image_fourrooms, policy_image_fourrooms
from envs.minigrid.env_utils import value_image_doors, policy_image_doors, doors_value_fn,\
    plot_image_pcas, random_exploration, image_mdps

FLAGS = flags.FLAGS
flags.DEFINE_bool('disable_jit', False, 'Whether to disable JIT compilation.')

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
        #'HGCDataset': HGCDataset,
    }[config['agent']['dataset_class']]
    
    train_dataset = dataset_class(Dataset.create(**train_dataset), config['agent'])
    if val_dataset is not None:
        val_dataset = dataset_class(Dataset.create(**val_dataset), config['agent'])
    
    random.seed(config['seed'])
    np.random.seed(config['seed'])

    example_batch = train_dataset.sample(1)
    if config['env']['discrete']:
        fill_value = getattr(env, "action_space", env.env)
        example_batch['actions'] = np.full_like(example_batch['actions'], fill_value=fill_value.action_space.n - 1)
    
    agent_class = agents[config['agent']['agent_name']]
    agent = agent_class.create(
        config['seed'],
        example_batch['observations'],
        example_batch['actions'],
        config['agent'],
    )

    train_logger = CsvLogger(os.path.join(config['save_dir'], 'train.csv'))
    eval_logger = CsvLogger(os.path.join(config['save_dir'], 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    if config['agent']['use_context']:
        layout_types = range(config['agent']['number_of_meta_envs']) # change to test zero-shot later
        # For debugging plots
        # layout0_embs = []
        # layout1_embs = []
        # layout2_embs = []
        _, batch_context_test0, _ = train_dataset.sample(512, layout_type=0,
                                                                        context_length=config['agent']['context_len'])
        _, batch_context_test1, _ = train_dataset.sample(512, layout_type=1,
                                                                                context_length=config['agent']['context_len'])
        colors = ['blue'] * batch_context_test0['observations'].shape[0] + ['red'] * batch_context_test1['observations'].shape[0]
        test_embedding_data = jax.tree.map(lambda x, y: jnp.concatenate([x, y]), batch_context_test0, batch_context_test1)

    pbar = tqdm(range(1, config['train_steps'] + 1), colour='green', dynamic_ncols=True, position=0, leave=True)
    for step in pbar:
        key = jax.random.fold_in(key, step)
        if not config['agent']['use_context']:
            batch = train_dataset.sample(config['agent']['batch_size'])
            agent, update_info = agent.update(batch)
        else:
            batch, batch_context, _ = train_dataset.sample(config['agent']['batch_size'], layout_type=step % 2,
                                                            context_length=config['agent']['context_len'])
            _, negative_context, _ = train_dataset.sample(config['agent']['batch_size'], layout_type=(step + 1) % 2,
                                                                context_length=config['agent']['context_len'])
            agent, update_info = agent.update(batch, batch_context, train_context_embedding=step % 500 == 0 and step < 150_000, negative_context=negative_context)
                
        # Log metrics.
        if step % config['log_interval'] == 0 or step == 1:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}

            # if config['agent']['use_context']:
            #     batch, batch_context, _ = train_dataset.sample(1, layout_type=0,
            #                                                         context_length=config['agent']['context_len'])
            #     dynamics_embedding, _ = agent.network.select('dynamic_transformer')(batch_context['observations'], batch_context['actions'],
            #                                                                             batch_context['next_observations'], train=False, return_embedding=True)
            #     layout0_embs.append(jax.device_get(dynamics_embedding))
            
            #     batch, batch_context, _ = train_dataset.sample(1, layout_type=1,
            #                                                         context_length=config['agent']['context_len'])
            #     dynamics_embedding, _ = agent.network.select('dynamic_transformer')(batch_context['observations'], batch_context['actions'],
            #                                                                             batch_context['next_observations'], train=False, return_embedding=True)
            #     layout1_embs.append(jax.device_get(dynamics_embedding))
                
            #     batch, batch_context, _ = train_dataset.sample(1, layout_type=2,
            #                                                         context_length=config['agent']['context_len'])
            #     dynamics_embedding, _ = agent.network.select('dynamic_transformer')(batch_context['observations'], batch_context['actions'],
            #                                                                             batch_context['next_observations'], train=False, return_embedding=True)
            #     layout2_embs.append(jax.device_get(dynamics_embedding))
                
    
            train_metrics['time/epoch_time'] = (time.time() - last_time) / config['log_interval']
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=step)
            train_logger.log(train_metrics, step=step)

        # Evaluate agent.
        if step == 1 or step % config['eval_interval'] == 0:
            renders = []
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            
            # A LOT OF LOGGING FUNCTIONS
            if 'ogbench' in config['env']['env_name']:
                task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
                num_tasks = config['eval_tasks'] if config['eval_tasks'] is not None else len(task_infos)
                for task_id in tqdm(range(1, num_tasks + 1), leave=False, position=1, colour='blue'):
                    task_name = task_infos[task_id - 1]['task_name']
                    eval_info, trajs, cur_renders = evaluate(
                        agent=agent,
                        env=env,
                        task_id=task_id,
                        config=config['env'],
                        num_eval_episodes=config['eval_episodes'],
                        num_video_episodes=config['video_episodes'],
                        video_frame_skip=config['video_frame_skip'],
                        eval_temperature=config['eval_temperature'],
                        eval_gaussian=config['eval_gaussian'],
                    )
                    renders.extend(cur_renders)
                    metric_names = ['success']
                    eval_metrics.update(
                        {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                    )
                    for k, v in eval_info.items():
                        if k in metric_names:
                            overall_metrics[k].append(v)
                            
                    if config['env']['env_name'].split("-")[1] in ['antmaze', 'pointmaze']:
                        observation, info = eval_env.reset(options=dict(task_id=task_id, render_goal=True))
                        goal = info.get('goal')
                        start = eval_env.unwrapped.get_xy()
                        latent_z = jax.device_get(agent.infer_z(goal)[None])
                        N, M = 14, 20
                        latent_z = np.tile(latent_z, (N * M, 1))
                        pred_value_img = value_image(eval_env, example_batch, N=N, M=M,
                                                    value_fn=partial(agent.predict_q, z=latent_z),
                                                    action_fn=partial(supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32))), latent_z=latent_z, temperature=0.1),
                                                    goal=goal)
                        pred_policy_img = policy_image(eval_env, example_batch, N=N, M=M,
                                                    action_fn=partial(supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32))), latent_z=latent_z, temperature=0.1),
                                                    goal=goal, start=start)
                        eval_metrics[f'draw_Q/draw_value_task_{task_id}'] = wandb.Image(pred_value_img)
                        eval_metrics[f'draw_policy/draw_policy_task_{task_id}'] = wandb.Image(pred_policy_img)
                
                for k, v in overall_metrics.items():
                    eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

                if config['video_episodes'] > 0:
                    video = get_wandb_video(renders=renders, n_cols=num_tasks)
                    eval_metrics['video'] = video

                wandb.log(eval_metrics, step=step)
                eval_logger.log(eval_metrics, step=step)
            
            if 'fourrooms' in config['env']['env_name'] or 'gridworld' in config['env']['env_name']:
                num_tasks = env.maze.num_tasks
                for task_id in tqdm(range(1, num_tasks + 1), leave=False, position=1, colour='blue'):
                    eval_info, trajs, cur_renders = evaluate_fourrooms(
                        agent=agent,
                        env=env,
                        task_id=task_id,
                        config=config['env'],
                        num_eval_episodes=config['eval_episodes'],
                        num_video_episodes=config['video_episodes'],
                        video_frame_skip=config['video_frame_skip'],
                        eval_temperature=config['eval_temperature'],
                        eval_gaussian=config['eval_gaussian'],
                    )
                    renders.extend(cur_renders)
                    metric_names = ['success']
                    eval_metrics.update(
                        {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                    )
                    for k, v in eval_info.items():
                        if k in metric_names:
                            overall_metrics[k].append(v)
                    
                    observation, info = env.setup_goals(seed=None, task_num=task_id)
                    goal = info.get("goal_pos", None)
                    start = eval_env.start
                    latent_z = jax.device_get(agent.infer_z(goal)[None])
                    N, M = eval_env.maze.size
                    pred_value_img = value_image_fourrooms(eval_env, example_batch, N=N, M=M,
                                                value_fn=partial(agent.predict_q, z=latent_z), goal=goal)
                    pred_policy_img = policy_image_fourrooms(eval_env, example_batch, N=N, M=M,
                                                action_fn=partial(supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32))), latent_z=latent_z, temperature=0.0),
                                                goal=goal, start=start)
                    eval_metrics[f'draw_Q/draw_value_task_{task_id}'] = wandb.Image(pred_value_img)
                    eval_metrics[f'draw_policy/draw_policy_task_{task_id}'] = wandb.Image(pred_policy_img)
                
                for k, v in overall_metrics.items():
                    eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

                if config['video_episodes'] > 0:
                    video = get_wandb_video(renders=renders, n_cols=num_tasks)
                    eval_metrics['video'] = video

                wandb.log(eval_metrics, step=step)
                eval_logger.log(eval_metrics, step=step)
            
            if 'doors' in config['env']['env_name']:
                if 'gciql' in config['agent']['agent_name']:
                    for layout_type in tqdm(layout_types, leave=False, position=1, colour='blue'):
                        if config['agent']['use_context']:
                            env.env.unwrapped._gen_grid = partial(env.env.unwrapped._gen_grid, layout_type=layout_type)
                            dataset_inference, env = random_exploration(env, num_episodes=1, layout_type=layout_type)
                            dynamics_embedding, _ = agent.network.select('dynamic_transformer')(dataset_inference['observations'][None], dataset_inference['actions'][None, :, None],
                                                                                                        dataset_inference['next_observations'][None], train=False, return_embedding=True)
                            dynamics_embedding = dynamics_embedding.squeeze()
                        else:
                            env.env.unwrapped._gen_grid = partial(env.env.unwrapped._gen_grid, layout_type=0)
                            dynamics_embedding=None
                            
                        obs, info = env.reset()
                        goal = info.get("goal_pos", None)
                        
                        pred_policy_img = policy_image_doors(env, example_batch,
                                                                        action_fn=partial(supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32))),
                                                                                        goals=goal, dynamics_embedding=dynamics_embedding, temperature=0.0),
                                                                        goal=goal)
                        pred_value_img = value_image_doors(env, example_batch,
                                value_fn=partial(doors_value_fn, agent, dynamics_embedding=dynamics_embedding), action_fn=partial(supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32))),
                                                                    goals=goal, temperature=0.0, dynamics_embedding=dynamics_embedding), goal=goal)
                        eval_metrics[f'draw_Q/draw_value_task_{layout_type}'] = wandb.Image(pred_value_img)
                        eval_metrics[f'draw_policy/draw_policy_task_{layout_type}'] = wandb.Image(pred_policy_img)
                    
                    if config['agent']['use_context']:
                        embedding_img_mdps = image_mdps(test_embedding_data, embedding_fn=partial(agent.network.select('dynamic_transformer'),
                                                                                                  train=False, return_embedding=True), colors=colors)
                        eval_metrics['draw_mdps_emb/PCA_mdps_emb_'] = wandb.Image(embedding_img_mdps)
                        
                    # if len(layout0_embs) > 30:
                    #     evolution_pca_0_img = plot_image_pcas(np.concatenate(layout0_embs))
                    #     evolution_pca_1_img = plot_image_pcas(np.concatenate(layout1_embs))
                    #     evolution_pca_2_img = plot_image_pcas(np.concatenate(layout2_embs))
                        
                    #     eval_metrics[f'draw_emb/draw_evolution_emb_mdp_0'] = wandb.Image(evolution_pca_0_img)
                    #     eval_metrics[f'draw_emb/draw_evolution_emb_mdp_1'] = wandb.Image(evolution_pca_1_img)
                    #     eval_metrics[f'draw_emb/draw_evolution_emb_mdp_2'] = wandb.Image(evolution_pca_2_img)
                    
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