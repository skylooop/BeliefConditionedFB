import os
os.environ['MUJOCO_GL']='egl'
# os.environ['CUDA_VISIBLE_DEVICES']='1'

import shutup
shutup.please()

import rootutils
ROOT = rootutils.setup_root(search_from=__file__, cwd=True, pythonpath=True, indicator='requirements.txt')

import random
import time
from rich.pretty import pprint
from functools import partial

import jax
import numpy as np
from tqdm.auto import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags
from collections import defaultdict
from colorama import Fore, Style

from agents import agents
from envs.env_utils import make_env_and_datasets

from utils.datasets import Dataset, ReplayBuffer, GCDataset
from utils.evaluation import evaluate, evaluate_fourrooms, flatten, supply_rng
# from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_wandb_video, setup_wandb
from envs.ogbench.ant_utils import policy_image, value_image
from envs.custom_mazes.env_utils import value_image_fourrooms, policy_image_fourrooms

FLAGS = flags.FLAGS

flags.DEFINE_bool('disable_jit', True, 'Disable jit.')
flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
# flags.DEFINE_string('env_name', 'ogbench-pointmaze-medium-stitch-v0', 'Environment (dataset) name.')
flags.DEFINE_string('env_name', 'maze-fourrooms-11', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'experiment_logs/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('train_steps', 1_000_000, 'Number of online steps.')

flags.DEFINE_integer('log_interval', 5_000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 10_000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_integer('eval_tasks', None, 'Number of online steps.')
flags.DEFINE_float('eval_temperature', 0.3, 'Number of online steps.')
flags.DEFINE_integer('eval_episodes', 1, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')

config_flags.DEFINE_config_file('agent', 'agents/fb.py', lock_config=False)

# def parse_wandb_absl(wandb_cfg, absl_cfg):
#     for 

def main():
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    key = jax.random.key(FLAGS.seed)
    exp_name = get_exp_name(FLAGS.seed)
    config = FLAGS.agent    
    # pprint(config.to_dict())
    
    run = setup_wandb(project='ZeroShotRL', group=FLAGS.run_group, name=exp_name, mode="offline" if FLAGS.disable_jit else "online")#, config=config)
    
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(dataset_name=FLAGS.env_name,
                                                                      frame_stack=FLAGS.frame_stack,
                                                                      action_clip_eps=1e-5 if not config['discrete'] else None)
    dataset_class = {
        'GCDataset': GCDataset,
        #'HGCDataset': HGCDataset,
    }[config['dataset_class']]
    train_dataset = dataset_class(Dataset.create(**train_dataset), config)
    if val_dataset is not None:
        val_dataset = dataset_class(Dataset.create(**val_dataset), config)
        
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    
    example_batch = train_dataset.sample(1)
    if config['discrete']:
        example_batch['actions'] = np.full_like(example_batch['actions'], fill_value=env.action_space.n - 1)
    
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    #### SHELL HELPER ####
    print(Fore.GREEN + "Train step" + Style.RESET_ALL)    
    print(Fore.BLUE + "Task" + Style.RESET_ALL)
    print(Fore.RED + "Episode" + Style.RESET_ALL)
    
    pbar = tqdm(range(1, FLAGS.train_steps + 1), colour='green', dynamic_ncols=True, position=0, leave=True)
    for step in pbar:
        key = jax.random.fold_in(key, step)
        batch = train_dataset.sample(config['batch_size'])
        agent, update_info = agent.update(batch)
        
        # Log metrics.
        if step % FLAGS.log_interval == 0 or step == 1:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                latent_z = agent.sample_mixed_z(batch, config['z_dim'], key)
                _, val_info = agent.total_loss(val_batch, latent_z, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
                
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=step)
            train_logger.log(train_metrics, step=step)
        
        # Evaluate agent.
        if step == 1 or step % FLAGS.eval_interval == 0:
            renders = []
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            
            if 'ogbench' in FLAGS.env_name:
                task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
                num_tasks = FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)
                for task_id in tqdm(range(1, num_tasks + 1), leave=False, position=1, colour='blue'):
                    task_name = task_infos[task_id - 1]['task_name']
                    eval_info, trajs, cur_renders = evaluate(
                        agent=agent,
                        env=env,
                        task_id=task_id,
                        config=config,
                        num_eval_episodes=FLAGS.eval_episodes,
                        num_video_episodes=FLAGS.video_episodes,
                        video_frame_skip=FLAGS.video_frame_skip,
                        eval_temperature=FLAGS.eval_temperature,
                        eval_gaussian=FLAGS.eval_gaussian,
                    )
                    renders.extend(cur_renders)
                    metric_names = ['success']
                    eval_metrics.update(
                        {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                    )
                    for k, v in eval_info.items():
                        if k in metric_names:
                            overall_metrics[k].append(v)
                            
                    if FLAGS.env_name.split("-")[1] in ['antmaze', 'pointmaze']:
                        observation, info = eval_env.reset(options=dict(task_id=task_id, render_goal=True))
                        goal = info.get('goal')
                        start = eval_env.get_xy()
                        #goal_rendered = info.get('goal_rendered')
                        latent_z = jax.device_get(agent.infer_z(goal)[None])
                        N, M = 14, 20
                        latent_z = np.tile(latent_z, (N * M, 1))
                        pred_value_img = value_image(eval_env, example_batch, N=N, M=M,
                                                    value_fn=partial(agent.predict_q, z=latent_z),
                                                    action_fn=partial(supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32))), latent_z=latent_z, temperature=0.05),
                                                    goal=goal)
                        pred_policy_img = policy_image(eval_env, example_batch, N=N, M=M,
                                                    action_fn=partial(supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32))), latent_z=latent_z, temperature=0.05),
                                                    goal=goal, start=start)
                        eval_metrics[f'draw_Q/draw_value_task_{task_id}'] = wandb.Image(pred_value_img)
                        eval_metrics[f'draw_policy/draw_policy_task_{task_id}'] = wandb.Image(pred_policy_img)
                
                for k, v in overall_metrics.items():
                    eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

                if FLAGS.video_episodes > 0:
                    video = get_wandb_video(renders=renders, n_cols=num_tasks)
                    eval_metrics['video'] = video

                wandb.log(eval_metrics, step=step)
                eval_logger.log(eval_metrics, step=step)
        
            if 'fourrooms' in FLAGS.env_name:
                num_tasks = 4
                for task_id in tqdm(range(1, num_tasks + 1), leave=False, position=1, colour='blue'):
                    eval_info, trajs, cur_renders = evaluate_fourrooms(
                        agent=agent,
                        env=env,
                        task_id=task_id,
                        config=config,
                        num_eval_episodes=FLAGS.eval_episodes,
                        num_video_episodes=FLAGS.video_episodes,
                        video_frame_skip=FLAGS.video_frame_skip,
                        eval_temperature=FLAGS.eval_temperature,
                        eval_gaussian=FLAGS.eval_gaussian,
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
                                                action_fn=partial(supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32))), latent_z=latent_z, temperature=0.3),
                                                goal=goal, start=start)
                    eval_metrics[f'draw_Q/draw_value_task_{task_id}'] = wandb.Image(pred_value_img)
                    eval_metrics[f'draw_policy/draw_policy_task_{task_id}'] = wandb.Image(pred_policy_img)
                
                for k, v in overall_metrics.items():
                    eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

                if FLAGS.video_episodes > 0:
                    video = get_wandb_video(renders=renders, n_cols=num_tasks)
                    eval_metrics['video'] = video

                wandb.log(eval_metrics, step=step)
                eval_logger.log(eval_metrics, step=step)
            
    train_logger.close()
    eval_logger.close()
    
    env.close()
    eval_env.close()

def entry(_):
    try:
        if FLAGS.disable_jit:
            with jax.disable_jit():
                main()
        else:
            main()
    except KeyboardInterrupt:
        wandb.finish()
        print(Fore.GREEN +"Finished!" + Style.RESET_ALL)

if __name__ == "__main__":
    app.run(entry)