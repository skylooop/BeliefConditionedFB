import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import ml_collections
import optax
from functools import partial

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import FBActor, FBDiscreteActor, FBValue, LengthNormalize


class ForwardBackwardAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def fb_loss(self, batch, z_latent, grad_params, rng):
        rng, sample_rng = jax.random.split(rng)
        
        # Target M
        next_dist = self.network.select('actor')(batch['next_observations'], z_latent)
        next_actions = next_dist.sample(seed=sample_rng)
        target_F, target_b =\
            self.network.select('target_fb')(batch['next_observations'], next_actions, z_latent,
                                             goal=batch['next_observations'])
        target_M = jnp.einsum("sd, td -> st", target_F, target_b)
        
        # Cur M
        F, b =\
            self.network.select('fb')(batch['observations'], batch['actions'], z_latent,
                                      goal=batch['next_observations'], params=grad_params)
        M = jnp.einsum("sd, td -> st", F, b)
        I = np.eye(batch['observations'].shape[0], dtype=bool)
        off_diag = ~I
        #M_ensemble = jnp.stack([M1, M2])
        # fb_loss = jax.vmap(
        #     lambda M: optax.sigmoid_binary_cross_entropy(logits=(M-self.config['discount'] * target_M), labels=I),
        # )(M_ensemble)
        # fb_loss = jnp.mean(fb_loss)
        # diff = M1-self.config['discount'] * target_M
        diff = M - self.config['discount'] * target_M
        fb_offdiag = ((diff[off_diag]) ** 2).mean()
        fb_diag = -jnp.diag(diff).mean()
        fb_loss = fb_diag + fb_offdiag
        
        # Orthonormality loss
        cov_b = b @ b.T
        ort_loss_diag = -2 * jnp.diag(cov_b).mean()
        ort_loss_offdiag = 0.5 * ((cov_b[off_diag])**2).sum()/ off_diag.sum()
        ort_b_loss = ort_loss_diag + ort_loss_offdiag
        total_loss = fb_loss + ort_b_loss
        
        correct_fb = jnp.argmax(M, axis=1) == jnp.argmax(I, axis=1)
        correct_ort = jnp.argmax(cov_b, axis=1) == jnp.argmax(I, axis=1)
        
        return total_loss, {
            "contrastive_fb_loss": fb_loss, 
            "fb_loss": total_loss,
            "z_norm": jnp.linalg.norm(z_latent, axis=-1).mean(),
            # ORTHONORMALITY METRICS
            "ort_b_loss": ort_b_loss,
            # "ort_loss_diag": ort_loss_diag,
            # "ort_loss_offdiag": ort_loss_offdiag,
            "correct_ort": jnp.mean(correct_ort),
            # FB LOSS
            "categorical_accuracy_M": jnp.mean(correct_fb),
            "fb_offdiag_loss": fb_offdiag,
            "fb_diag_loss": fb_diag,
            "target_M": target_M.mean(),
            "M": M.mean(),
        }

    def actor_loss(self, batch, z_latent, grad_params, rng):
        rng, sample_rng = jax.random.split(rng)
    
        dist = self.network.select('actor')(batch['observations'], z_latent, params=grad_params)
        actions, log_probs = dist.sample_and_log_prob(seed=sample_rng)

        F, B = self.network.select('fb')(batch['observations'], actions=actions, latent_z=z_latent, goal=batch['actor_goals'])
        Q = jnp.einsum('bz, bz -> b', F, z_latent)
    
        actor_loss = (0.1 * log_probs - Q).mean()
        return actor_loss, {
            'mean_action': actions.mean(),
            'total_loss': actor_loss,
        }

    @jax.jit
    def total_loss(self, batch, latent_z, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, fb_recon_rng = jax.random.split(rng, 3)

        fb_loss, fb_info = self.fb_loss(batch, latent_z, grad_params, fb_recon_rng)
        for k, v in fb_info.items():
            info[f'fb/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, latent_z, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = fb_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            network.params[f'modules_{module_name}'],
            network.params[f'modules_target_{module_name}'],
        )
        return new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)
        z = self.sample_mixed_z(batch, self.config['z_dim'], new_rng)
        
        def loss_fn(grad_params):
            return self.total_loss(batch, z, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        new_target_params = self.target_update(new_network, 'fb')
        new_network.params['modules_target_fb'] = new_target_params
        
        return self.replace(network=new_network, rng=new_rng), info

    def project_z(self, z):
        return z * jnp.sqrt(z.shape[-1]) / jnp.linalg.norm(z, axis=-1, keepdims=True)
    
    @partial(jax.jit, static_argnums=(1, 2))
    def sample_z(self, batch_size, latent_dim, key):
        z = jax.random.normal(shape=(batch_size, latent_dim), key=key)
        return self.project_z(z)
    
    def sample_mixed_z(self, batch, latent_dim, key):
        batch_size = batch['observations'].shape[0]
        z = self.sample_z(batch_size, latent_dim, key)
        b_goals = self.network.select('fb')(goal=batch['actor_goals'], get_backward=True)
        mask = jax.random.uniform(key, shape=(batch_size, 1)) < self.config['z_mix_ratio']
        z = jnp.where(mask, b_goals, z)
        return z
        
    def infer_z(self, obs, rewards=None):
        """
        If reards are None -> treat as goal-conditioned
        """    
        z = self.network.select('fb')(goal=obs, get_backward=True)
        if rewards is not None:
            z = (rewards.T @ z)
        z = self.project_z(z)
        return z
        
    @jax.jit
    def sample_actions(
        self,
        observations,
        latent_z=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        dist = self.network.select('actor')(observations, latent_z, temperature=temperature)
        actions = dist.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions. For discrete must contain max action value.
            config: Configuration dictionary.
        """
        rng = jax.random.key(seed)
        rng, init_rng = jax.random.split(rng, 2)
        
        ex_goals = ex_observations
        
        if config['discrete']:
            action_dim = ex_actions
            ex_actions = jnp.atleast_1d(jnp.array(ex_actions))
        else:
            action_dim = ex_actions.shape[-1]
        
        # Define networks.
        fb_def = FBValue(
            latent_z_dim=config['z_dim'],
            fb_forward_hidden_dims=config['fb_forward_hidden_dims'],
            fb_forward_layer_norm=config['fb_forward_layer_norm'],
            fb_forward_preprocessor_hidden_dims=config['fb_forward_preprocessor_hidden_dims'],
            fb_backward_hidden_dims=config['fb_backward_hidden_dims'],
            fb_backward_layer_norm=config['fb_backward_layer_norm'],
            grid_world=config['discrete']
        )
        
        if config['discrete']:
            actor_def = FBDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                layer_norm=config['actor_layer_norm'],
                grid_world=True
            )
        else:
            actor_def = FBActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                tanh_squash=config['tanh_squash'],
                state_dependent_std=config['state_dependent_std'],
                const_std=False,
                final_fc_init_scale=config['actor_fc_scale'],
            )
        latent_z = jax.random.normal(init_rng, shape=(1, config['z_dim']))
        latent_z = latent_z * jnp.sqrt(latent_z.shape[-1]) / jnp.linalg.norm(latent_z, axis=-1, keepdims=True)
        network_info = dict(
            fb=(fb_def, (ex_observations, ex_actions, latent_z, ex_goals)),
            target_fb=(copy.deepcopy(fb_def), (ex_observations, ex_actions, latent_z, ex_goals)),
            actor=(actor_def, (ex_observations, latent_z)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.chain(
                optax.clip_by_global_norm(max_norm=config['z_dim']),
                optax.adam(learning_rate=config['lr'])
        )
        #network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        
        params = network_params
        params['modules_target_fb'] = params['modules_fb']
        
        network = TrainState.create(network_def, params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Most of hyperparams from zero-shot rl from high-quality data
            # https://arxiv.org/pdf/2309.15178
            agent_name='fb',  # Agent name.
            discrete=False,
            lr=1e-4,  # Learning rate.
            batch_size=512,  # Batch size.
            
            # FB Specific
            z_dim=50, # 100 for maze env, 50 for others
            fb_forward_hidden_dims=(1024, 1024),  # Value network hidden dimensions.
            fb_forward_layer_norm=True,  # Whether to use layer normalization.
            fb_forward_preprocessor_hidden_dims=(1024, 1024, 512),
            fb_backward_hidden_dims=(256, 256, 256),  # Value network hidden dimensions.
            fb_backward_layer_norm=True,  # Whether to use layer normalization.
            z_mix_ratio=0.5,
            # Actor
            actor_hidden_dims=(1024, 1024),  # Actor network hidden dimensions.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor. 0.99 - for maze, 0.98 others
            tau=0.005,  # Target network update rate.
            tanh_squash=True,  # Whether to squash actions with tanh.
            state_dependent_std=True,  # Whether to use state-dependent standard deviations for actor.
            actor_fc_scale=0.01,  # Final layer initialization scale for actor.
    
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.0,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=0.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=1.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
