import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, FBDiscreteActor, FBValue, LengthNormalize


class ForwardBackwardAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def fb_loss(self, batch, grad_params, rng):
        rng, sample_rng = jax.random.split(rng)
        
        # Target M
        z_latent = self.sample_z(key=sample_rng, batch_size=self.config['batch_size'], latent_dim=self.config['z_dim'])
        next_dist = self.network.select('actor')(batch['next_observations'], z_latent)
        next_actions = next_dist.sample(seed=sample_rng)
        target_next_fb, target_f, target_b = self.network.select('target_fb')(batch['next_observations'], next_actions, z_latent, batch['fb_goals'], info=True)
        
        # Cur M
        fb_M, cur_f, cur_b = self.network.select('fb')(batch['observations'], batch['actions'], z_latent, batch['fb_goals'], info=True, params=grad_params)
        
        diff = fb_M - self.config['discount']*target_next_fb
        I = jnp.eye(*fb_M.shape, dtype=bool)
        off_diagonal = ~I

        fb_off_diag_loss = 0.5 * diff[off_diagonal].pow(2).mean()
        fb_diag_loss = -fb_M.diag().mean()
        fb_loss = fb_off_diag_loss + fb_diag_loss
        
        # Orthonormality loss
        cov = cur_b @ cur_b.T
        ort_loss_diag = -2 * cov.diag().mean()
        ort_loss_offdiag = cov[off_diagonal].pow(2).mean()
        ort_loss = ort_loss_diag + ort_loss_offdiag
        total_loss = fb_loss + ort_loss
        
        return total_loss, {
            'fb_loss': total_loss,
        }

    def actor_loss(self, batch, grad_params, rng):
        rng, sample_rng = jax.random.split(rng)
        
        z_latent = self.sample_z(key=sample_rng, batch_size=self.config['batch_size'], latent_dim=self.config['z_dim'])
        dist = self.network.select('actor')(batch['observations'], z_latent, params=grad_params)
        actions, log_probs = dist.sample_and_log_prob(seed=rng)

        # Actor loss.
        M, F, B = self.network.select('fb')(batch['observations'], actions, z_latent, batch['actor_goals'])
        Q = -(F @ z_latent).mean()
        
        return Q, {
            'total_loss': Q,
            'entropy': -log_probs.mean()
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, fb_recon_rng = jax.random.split(rng, 3)

        fb_loss, fb_info = self.fb_loss(batch, grad_params, fb_recon_rng)
        for k, v in fb_info.items():
            info[f'fb/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = fb_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    def sample_z(self, batch_size, latent_dim, key):
        z = jax.random.normal(shape=(batch_size, latent_dim), key=key)
        return z / jnp.linalg.norm(z, axis=-1, keepdims=True) * jnp.sqrt(z.shape[-1])
        
    def infer_z(self, obs, rewards=None):
        """
        If reards are None -> treat as goal-conditioned
        """    
        _, _, z = self.network.select('target_fb')(obs, None, None, obs, info=True)
        if rewards is not None:
            z = (rewards.T @ z)
        # TODO: Normalize?
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
        
        if config['discrete']:
            action_dim = ex_actions
            ex_actions = jnp.atleast_1d(jnp.array(ex_actions))
        else:
            action_dim = ex_actions.shape[-1]
        
        # Define networks.
        fb_def = FBValue(
            hidden_dims=config['fb_hidden_dims'],
            layer_norm=config['fb_layer_norm'],
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
            actor_def = Actor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                layer_norm=config['actor_layer_norm'],
                tanh_squash=config['tanh_squash'],
                state_dependent_std=config['state_dependent_std'],
                const_std=False,
                final_fc_init_scale=config['actor_fc_scale'],
            )
        latent_z = jax.random.normal(init_rng, shape=(config['z_dim']))
        network_info = dict(
            fb=(fb_def, (ex_observations, ex_actions, latent_z)),
            target_fb=(copy.deepcopy(fb_def), (ex_observations, ex_actions, latent_z)),
            actor=(actor_def, (ex_observations, latent_z)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_fb'] = params['modules_fb']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='forward_backward',  # Agent name.
            discrete=True, # not used right now,
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            z_dim=50,
            actor_hidden_dims=(512, 512, 512, 256),  # Actor network hidden dimensions.
            fb_hidden_dims=(512, 512, 512, 256),  # Value network hidden dimensions.
            fb_layer_norm=False,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.95,  # Discount factor.
            tau=0.005,  # Target network update rate.
            target_entropy=ml_collections.config_dict.placeholder(float),  # Target entropy (None for automatic tuning).
            target_entropy_multiplier=0.5,  # Multiplier to dim(A) for target entropy.
            tanh_squash=True,  # Whether to squash actions with tanh.
            state_dependent_std=True,  # Whether to use state-dependent standard deviations for actor.
            actor_fc_scale=0.01,  # Final layer initialization scale for actor.
            q_agg='min',  # Aggregation function for target Q values.
            backup_entropy=False,  # Whether to back up entropy in the critic loss.
            
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.0,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
