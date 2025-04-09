from envs.custom_mazes.env_utils import policy_image_fourrooms, value_image_fourrooms
from functools import partial
from utils.evaluation import supply_rng
from envs.minigrid.env_utils import random_exploration_fourrooms
import jax
import jax.numpy as jnp
import numpy as np

def visualize_value_image(env, agent, key, example_batch, layout_type, task_num):
    env.reset()
    observation, info = env.unwrapped.setup_goals(seed=None, task_num=task_num)
    goal = info.get("goal_pos", None)
    mdp_type=None
    
    dataset_inference, env = random_exploration_fourrooms(env, num_episodes=1, layout_type=layout_type, num_mdp=100)
    dynamics_embedding_mean, dynamics_mean_std = agent.network.select('dynamic_transformer')(dataset_inference['observations'][None], dataset_inference['actions'][None, :, None],
                                                                                dataset_inference['next_observations'][None], train=False, return_embedding=True)
    dynamics_embedding = dynamics_embedding_mean + jax.random.normal(key=key, shape=dynamics_embedding_mean.shape) * jnp.exp(dynamics_mean_std)
    dynamics_embedding = dynamics_embedding.squeeze()

    latent_z = jax.device_get(agent.infer_z(goal, mdp_num=None, dynamics_embedding=dynamics_embedding)[None])
    N, M = env.unwrapped.maze.size
    pred_value_img = value_image_fourrooms(env.unwrapped, example_batch, N=N, M=M,
                                value_fn=partial(agent.predict_q, z=latent_z, mdp_num=mdp_type[None] if mdp_type is not None else None,
                                                dynamics_embedding=dynamics_embedding[None]),
                                action_fn=None, goal=goal)
    return pred_value_img

def visualize_policy(env, agent, key, whole_dataset, layout_type, task_num):
    env.reset()
    observation, info = env.unwrapped.setup_goals(seed=None, task_num=task_num)
    goal = info.get("goal_pos", None)
    dataset_inference, env = random_exploration_fourrooms(env, num_episodes=1, layout_type=layout_type, num_mdp=100)
    dynamics_embedding_mean, dynamics_mean_std = agent.network.select('dynamic_transformer')(dataset_inference['observations'][None], dataset_inference['actions'][None,:,None],
                                                                                dataset_inference['next_observations'][None], train=False, return_embedding=True)
    dynamics_embedding = dynamics_embedding_mean + jax.random.normal(key=key, shape=dynamics_embedding_mean.shape) * jnp.exp(dynamics_mean_std)
    dynamics_embedding=dynamics_embedding.squeeze()
    mdp_type=None
    
    latent_z = agent.infer_z(goal, mdp_num=mdp_type, dynamics_embedding=dynamics_embedding)
    start = info.get("start_pos", None)
    example_batch = whole_dataset.sample(1)
    mdp_type=None
    N, M = env.unwrapped.maze.size
    pred_policy_img = policy_image_fourrooms(env.unwrapped, example_batch, N=N, M=M,
                                                    action_fn=partial(supply_rng(agent.sample_actions,
                                                                                rng=jax.random.PRNGKey(np.random.randint(0, 2**32))), latent_z=latent_z,
                                                                    mdp_num=mdp_type[None] if mdp_type is not None else None, dynamics_embedding=dynamics_embedding[None], temperature=0.0),
                                                    goal=goal)
    return pred_policy_img