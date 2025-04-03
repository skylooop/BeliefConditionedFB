import copy
from typing import Any

import flax
import flax.struct
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import RecurrentValue, RecurrentActor

class RecurrentPPO(flax.struct.PyTreeNode):
    """
    Recurrent PPO
    """
    rng: Any
    network: Any
    config: Any = nonpytree_field()
    
    def policy_loss(self, batch, grad_params, rng):
        initial_hidden = batch['initial_hidden_actor']
        dist, _ = self.network.select('actor')(batch['observations'], initial_hidden, params=grad_params)
        new_log_probs = dist.log_prob(batch['actions'])
        ratio = jnp.exp(new_log_probs - batch['old_log_probs'])
        clipped_ratio = jnp.clip(ratio, 1 - self.config['clip_eps'], 1 + self.config['clip_eps'])
        surrogate_loss = jnp.minimum(ratio * batch['advantages'], clipped_ratio * batch['advantages'])
        policy_loss = -jnp.mean(surrogate_loss)

        approx_kl = jnp.mean(batch['old_log_probs'] - new_log_probs)
        clip_fraction = jnp.mean((ratio > 1 + self.config['clip_eps']) | (ratio < 1 - self.config['clip_eps']))

        return policy_loss, {
            'policy_loss': policy_loss,
            'approx_kl': approx_kl,
            'clip_fraction': clip_fraction,
            'ratio_mean': jnp.mean(ratio)
        }
    
    def entropy_bonus(self, batch, grad_params, rng):
        initial_hidden = batch['initial_hidden_actor']
        dist, _ = self.network.select('actor')(batch['observations'], initial_hidden, params=grad_params)
        entropy = jnp.mean(dist.entropy())
        return entropy, {'entropy': entropy}
    
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, policy_rng, value_rng, entropy_rng = jax.random.split(rng, 4)

        policy_loss, policy_info = self.policy_loss(batch, grad_params, policy_rng)
        value_loss, value_info = self.value_loss(batch, grad_params, value_rng)
        entropy, entropy_info = self.entropy_bonus(batch, grad_params, entropy_rng)

        total_loss = (
            policy_loss +
            self.config['vf_coef'] * value_loss -
            self.config['ent_coef'] * entropy
        )

        info.update({f'policy/{k}': v for k, v in policy_info.items()})
        info.update({f'value/{k}': v for k, v in value_info.items()})
        info.update({f'entropy/{k}': v for k, v in entropy_info.items()})
        info['total_loss'] = total_loss

        return total_loss, info
    
    def value_loss(self, batch, grad_params, rng):
        initial_hidden = batch['initial_hidden_value']
        values_pred, _ = self.network.select('value')(batch['observations'], initial_hidden, params=grad_params)
        value_loss = jnp.mean(jnp.square(values_pred - batch['returns']))
        return value_loss, {
            'value_loss': value_loss,
            'value_pred': jnp.mean(values_pred)
        }
        
    @jax.jit
    def sample_actions(self, observations, hidden_state, seed=None):
        rng = seed if seed is not None else self.rng
        rng, sample_rng = jax.random.split(rng)
        dist, new_hidden = self.network.select('actor')(observations[None], hidden_state)
        action = dist.sample(seed=sample_rng).squeeze(0)
        return action, new_hidden
        
    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info
    
    @classmethod
    def create(cls,seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        action_dim = ex_actions.shape[-1]

        # Initialize hidden states
        if config['rnn_type'] == 'lstm':
            initial_hidden_actor = jnp.zeros((1, config['actor_hidden_dim']), jnp.zeros((1, config['actor_hidden_dim'])))
            initial_hidden_value = jnp.zeros((1, config['value_hidden_dim']), jnp.zeros((1, config['value_hidden_dim'])))
        else:
            initial_hidden_actor = jnp.zeros((1, config['actor_hidden_dim']))
            initial_hidden_value = jnp.zeros((1, config['value_hidden_dim']))

        actor_def = RecurrentActor(
            hidden_dim=config['actor_hidden_dim'],
            action_dim=action_dim,
            rnn_type=config['rnn_type'],
            layer_norm=config['actor_layer_norm'],
            tanh_squash=config['tanh_squash'],
            state_dependent_std=config['state_dependent_std']
        )
        value_def = RecurrentValue(
            hidden_dim=config['value_hidden_dim'],
            rnn_type=config['rnn_type'],
            layer_norm=config['layer_norm']
        )

        network_info = dict(
            actor=(actor_def, (ex_observations, initial_hidden_actor)),
            value=(value_def, (ex_observations, initial_hidden_value)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(config))
    
    
def get_config():
    config = ml_collections.ConfigDict({
        'agent_name': 'recurrent_ppo',
        'lr': 3e-4,
        'batch_size': 256,
        'actor_hidden_dim': 256,
        'value_hidden_dim': 256,
        'rnn_type': 'lstm',
        'layer_norm': True,
        'actor_layer_norm': False,
        'discount': 0.99,
        'gae_lambda': 0.95,
        'clip_eps': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'tanh_squash': True,
        'state_dependent_std': True,
        'target_entropy': None,
        'target_entropy_multiplier': 0.5,
        'actor_fc_scale': 0.01
    })
    return config