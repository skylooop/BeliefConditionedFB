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
        """Compute the SAC critic loss."""
        rng, sample_rng = jax.random.split(rng)
        z_latent = self.sample_z(key=sample_rng, batch_size=self.config['batch_size'], latent_dim=self.config['z_dim'])
        
        next_dist = self.network.select('actor')(batch['next_observations'], z_latent)
        next_actions, next_log_probs = next_dist.sample_and_log_prob(seed=sample_rng)

        next_fb = self.network.select('target_fb')(batch['next_observations'], next_actions)
        # Add gcdataset
        q = self.network.select('critic')(batch['observations'], batch['actions'], params=grad_params)
        
        critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the SAC actor loss."""
        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        actions, log_probs = dist.sample_and_log_prob(seed=rng)

        # Actor loss.
        qs = self.network.select('critic')(batch['observations'], actions)
        q = jnp.mean(qs, axis=0)

        actor_loss = (log_probs * self.network.select('alpha')() - q).mean()

        # Entropy loss.
        alpha = self.network.select('alpha')(params=grad_params)
        entropy = -jax.lax.stop_gradient(log_probs).mean()
        alpha_loss = (alpha * (entropy - self.config['target_entropy'])).mean()

        total_loss = actor_loss + alpha_loss

        if self.config['tanh_squash']:
            action_std = dist._distribution.stddev()
        else:
            action_std = dist.stddev().mean()

        return total_loss, {
            'total_loss': total_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': alpha,
            'entropy': -log_probs.mean(),
            'std': action_std.mean(),
            'q': q.mean(),
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
            grid_world=True
        )
        
        actor_def = FBDiscreteActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            grid_world=True
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
        )
    )
    return config
