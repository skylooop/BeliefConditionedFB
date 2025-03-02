import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
# from utils.networks import FBActor, FValue, BValue, FValueDiscrete, LengthNormalize

class PSMAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def psm_loss(self, batch, z_latent, grad_params, rng):
        rng, sample_rng = jax.random.split(rng)
        
        pass

    def actor_loss(self, batch, z_latent, grad_params, rng):
        rng, sample_rng = jax.random.split(rng)

        pass

    @jax.jit
    def total_loss(self, batch, latent_z, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, fb_recon_rng = jax.random.split(rng, 3)

        fb_loss, fb_info = self.fb_loss(batch, latent_z, grad_params, fb_recon_rng)
        for k, v in fb_info.items():
            info[f'fb/{k}'] = v

        actor_loss = 0.0
        if not self.config['discrete']:
            actor_loss, actor_info = self.actor_loss(batch, latent_z, grad_params, actor_rng)
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
        
        return self.replace(network=new_network, rng=new_rng), info
        
    @jax.jit
    def sample_actions(
        self,
        observations,
        latent_z,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        if not self.config['discrete']:
            dist = self.network.select('actor')(observations, latent_z, temperature=temperature)
            actions = dist.sample(seed=seed)
            actions = jnp.clip(actions, -1, 1)
        else:
            latent_z = jnp.atleast_2d(latent_z)
            Q = self.predict_q(observations, latent_z)
            actions = jnp.argmax(Q, axis=-1)
            
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
            action_dim = int(ex_actions.max() + 1)
        else:
            action_dim = ex_actions.shape[-1]
        
        # Define networks.
        forward_def = FValue(
            latent_z_dim=config['z_dim'],
            preprocess=True,
            f_layer_norm=config['f_layer_norm'],
            f_preprocessor_hidden_dims=config['f_preprocessor_hidden_dims'],
            f_hidden_dims=config['f_hidden_dims'],
            activate_final=config['f_activate_final'],
        )
        backward_def = BValue(
            latent_z_dim=config['z_dim'],
            b_layer_norm=config['b_layer_norm'],
            b_hidden_dims=config['b_hidden_dims'],
        )
        actor_def = None
        if not config['discrete']:
            actor_def = FBActor(
                hidden_dims=config['actor_hidden_dims'],
                actor_preprocessor_layer_norm=config['actor_preprocessor_layer_norm'],
                actor_preprocessor_activate_final=config['actor_preprocessor_activate_final'],
                actor_preprocessor_hidden_dims=config['actor_preprocessor_hidden_dims'],
                action_dim=action_dim,
                tanh_squash=config['tanh_squash'],
                state_dependent_std=config['state_dependent_std'],
                const_std=config['const_std'],
                final_fc_init_scale=config['actor_fc_scale'],
            )
        network_info = dict(
            f_value=(forward_def, (ex_observations, ex_actions, latent_z)) if not config['discrete'] \
                else (forward_def, (ex_observations, latent_z)),
            target_f_value=(copy.deepcopy(forward_def), (ex_observations, ex_actions, latent_z)) if not config['discrete'] \
                else (copy.deepcopy(forward_def), (ex_observations, latent_z)),
            b_value=(backward_def, (ex_goals, )),
            target_b_value = (copy.deepcopy(backward_def), (ex_goals, )),
        )
        if actor_def is not None:
            network_info.update({"actor": (actor_def, (ex_observations, latent_z))})
            
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.chain(optax.clip_by_global_norm(1.0) if config['clip_by_global_norm'] else optax.identity(),
                                 optax.adam(learning_rate=config['lr']))
        
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)
        
        params = network.params
        params['modules_target_f_value'] = params['modules_f_value']
        params['modules_target_b_value'] = params['modules_b_value']
        
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))
