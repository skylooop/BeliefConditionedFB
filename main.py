import os
os.environ['MUJOCO_GL']='egl'
os.environ['CUDA_VISIBLE_DEVICES']='1'

import rootutils
ROOT = rootutils.setup_root(search_from=__file__, cwd=True, pythonpath=True)

import json
import random
import time
from rich.pretty import pprint

import jax
import numpy as np
from tqdm.auto import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags
from collections import defaultdict

from agents import agents
from envs.env_utils import make_env_and_datasets

from utils.datasets import Dataset, ReplayBuffer, GCDataset
from utils.evaluation import evaluate, flatten
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-double-play-singletask-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('online_steps', 0, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')

config_flags.DEFINE_config_file('agent', 'agents/fb.py', lock_config=False)



def main(_):
    config = FLAGS.agent
    pprint(config)
    
    exp_name = get_exp_name(FLAGS.seed)
    env, train_dataset, val_dataset = make_env_and_datasets(dataset_name=FLAGS.env_name, frame_stack=FLAGS.frame_stack)
    
    # setup_wandb(project='OGBench', group=FLAGS.run_group, name=exp_name)
    
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    
    example_batch = train_dataset.sample(1)
    if config['discrete']:
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)
    
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

    for step in tqdm(range(1, FLAGS.train_steps + 1), colour='green', dynamic_ncols=True):
        batch = train_dataset.sample(config['batch_size'])
        agent, update_info = agent.update(batch)
        
        # Log metrics.
        if step % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
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
                for task_id in tqdm.trange(1, num_tasks + 1):
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
            for k, v in overall_metrics.items():
                eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders, n_cols=num_tasks)
                eval_metrics['video'] = video

            wandb.log(eval_metrics, step=step)
            eval_logger.log(eval_metrics, step=step)

    train_logger.close()
    eval_logger.close()
    
    
if __name__ == "__main__":
    app.run(main)
