import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCDiscreteActor, GCDiscreteCritic, GCValue
from functools import partial
from utils.encoders import encoder_modules


class GCIQLAgent(flax.struct.PyTreeNode):
    """Goal-conditioned implicit Q-learning (GCIQL) agent.

    This implementation supports both AWR (actor_loss='awr') and DDPG+BC (actor_loss='ddpgbc') for the actor loss.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params, train_transformer, batch_context=None):
        """Compute the IQL value loss."""
        dynamics_embedding=None
        stop_grad_dynamics_embedding=None
        if self.config['use_context']:
            # dynamics_embedding = jnp.eye(10)[batch_context['layout_type']][:, 0, :].squeeze()
            # stop_grad_dynamics_embedding = jnp.eye(10)[batch_context['layout_type']][:, 0, :].squeeze()
            if train_transformer:
                dynamics_embedding, _ = self.network.select('dynamic_transformer')(batch_context['observations'], batch_context['actions'],
                                                                                batch_context['next_observations'], train=True, return_embedding=True, params=grad_params)
            else:
                dynamics_embedding, _ = self.network.select('dynamic_transformer')(batch_context['observations'], batch_context['actions'],
                                                                                batch_context['next_observations'], train=False, return_embedding=True)
            stop_grad_dynamics_embedding = jax.lax.stop_gradient(dynamics_embedding)
        q1, q2 = self.network.select('target_critic')(batch['observations'], batch['value_goals'], batch['actions'], stop_grad_dynamics_embedding)
        q = jnp.minimum(q1, q2)
        v = self.network.select('value')(batch['observations'], batch['value_goals'], dynamics_embedding=dynamics_embedding, params=grad_params)
        value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def critic_loss(self, batch, grad_params, train_transformer, batch_context=None):
        """Compute the IQL critic loss."""
        dynamics_embedding=None
        stop_grad_dynamics_embedding=None
        if self.config['use_context']:
            # dynamics_embedding = jnp.eye(10)[batch_context['layout_type']][:, 0, :].squeeze()
            # stop_grad_dynamics_embedding = jnp.eye(10)[batch_context['layout_type']][:, 0, :].squeeze()
            if train_transformer:
                dynamics_embedding, _ = self.network.select('dynamic_transformer')(batch_context['observations'], batch_context['actions'],
                                                                                batch_context['next_observations'], train=True, return_embedding=True, params=grad_params)
            else:
                dynamics_embedding, _ = self.network.select('dynamic_transformer')(batch_context['observations'], batch_context['actions'],
                                                                                batch_context['next_observations'], train=False, return_embedding=True)
            stop_grad_dynamics_embedding = jax.lax.stop_gradient(dynamics_embedding)
        next_v = self.network.select('value')(batch['next_observations'], batch['value_goals'], dynamics_embedding=stop_grad_dynamics_embedding)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v

        q1, q2 = self.network.select('critic')(
            batch['observations'], batch['value_goals'], batch['actions'], params=grad_params, dynamics_embedding=dynamics_embedding
        )
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, train_transformer, batch_context=None, rng=None):
        """Compute the actor loss (AWR or DDPG+BC)."""
        dynamics_embedding=None
        stop_grad_dynamics_embedding=None
        if self.config['use_context']:
            # dynamics_embedding = jnp.eye(10)[batch_context['layout_type']][:, 0, :].squeeze()
            # stop_grad_dynamics_embedding = jnp.eye(10)[batch_context['layout_type']][:, 0, :].squeeze()
            if train_transformer:
                dynamics_embedding, _ = self.network.select('dynamic_transformer')(batch_context['observations'], batch_context['actions'],
                                                                                batch_context['next_observations'], train=True, return_embedding=True, params=grad_params)
            else:
                dynamics_embedding, _ = self.network.select('dynamic_transformer')(batch_context['observations'], batch_context['actions'],
                                                                                batch_context['next_observations'], train=False, return_embedding=True)
            stop_grad_dynamics_embedding = jax.lax.stop_gradient(dynamics_embedding)
        if self.config['actor_loss'] == 'awr':
            # AWR loss.
            v = self.network.select('value')(batch['observations'], batch['actor_goals'], dynamics_embedding=stop_grad_dynamics_embedding)
            q1, q2 = self.network.select('critic')(batch['observations'], batch['actor_goals'], batch['actions'], dynamics_embedding=stop_grad_dynamics_embedding)
            q = jnp.minimum(q1, q2)
            adv = q - v

            exp_a = jnp.exp(adv * self.config['alpha'])
            exp_a = jnp.minimum(exp_a, 100.0)

            dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], dynamics_embedding=dynamics_embedding, params=grad_params)
            log_prob = dist.log_prob(batch['actions'])

            actor_loss = -(exp_a * log_prob).mean()

            actor_info = {
                'actor_loss': actor_loss,
                'adv': adv.mean(),
                'bc_log_prob': log_prob.mean(),
            }
            if not self.config['discrete']:
                actor_info.update(
                    {
                        'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                        'std': jnp.mean(dist.scale_diag),
                    }
                )

            return actor_loss, actor_info
        elif self.config['actor_loss'] == 'ddpgbc':
            # DDPG+BC loss.
            assert not self.config['discrete']

            dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
            if self.config['const_std']:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
            q1, q2 = self.network.select('critic')(batch['observations'], batch['actor_goals'], q_actions)
            q = jnp.minimum(q1, q2)

            # Normalize Q values by the absolute mean to make the loss scale invariant.
            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
            log_prob = dist.log_prob(batch['actions'])

            bc_loss = -(self.config['alpha'] * log_prob).mean()

            actor_loss = q_loss + bc_loss

            return actor_loss, {
                'actor_loss': actor_loss,
                'q_loss': q_loss,
                'bc_loss': bc_loss,
                'q_mean': q.mean(),
                'q_abs_mean': jnp.abs(q).mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }
        else:
            raise ValueError(f'Unsupported actor loss: {self.config["actor_loss"]}')

    def total_loss(self, batch, grad_params, train_transformer, batch_context=None, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params, train_transformer=train_transformer, batch_context=batch_context)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params, train_transformer=train_transformer, batch_context=batch_context)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, train_transformer=train_transformer, rng=actor_rng, batch_context=batch_context)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @partial(jax.jit, static_argnames=('train_transformer'))
    def update(self, batch, batch_context=None, train_transformer=True):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, train_transformer=train_transformer, batch_context=batch_context, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        dynamics_embedding=None,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        dist = self.network.select('actor')(observations, goals, dynamics_embedding=dynamics_embedding, temperature=temperature)
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
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        # if config['encoder'] is not None:
        #     encoder_module = encoder_modules[config['encoder']]
        #     encoders['value'] = GCEncoder(concat_encoder=encoder_module())
        #     encoders['critic'] = GCEncoder(concat_encoder=encoder_module())
        #     encoders['actor'] = GCEncoder(concat_encoder=encoder_module())

        # Define value and actor networks.
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=False,
            gc_encoder=encoders.get('value'),
        )

        if config['discrete']:
            critic_def = GCDiscreteCritic(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                gc_encoder=encoders.get('critic'),
                action_dim=action_dim,
            )
        else:
            critic_def = GCValue(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                gc_encoder=encoders.get('critic'),
            )

        if config['discrete']:
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=encoders.get('actor'),
            )
        else:
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoders.get('actor'),
            )
        if config['use_context']:
            dynamics_embedding = jnp.zeros((1, config['h_dim'])) # jnp.zeros((1, config['number_of_meta_envs']))#
        network_info = dict(
            value=(value_def, (ex_observations, ex_goals) if not config['use_context'] else (ex_observations, ex_goals, None, dynamics_embedding)),
            critic=(critic_def, (ex_observations, ex_goals, ex_actions) if not config['use_context'] else (ex_observations, ex_goals, ex_actions, dynamics_embedding)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_goals, ex_actions) if not config['use_context'] else (ex_observations, ex_goals, ex_actions, dynamics_embedding)),
            actor=(actor_def, (ex_observations, ex_goals) if not config['use_context'] else (ex_observations, ex_goals, dynamics_embedding)),
        )
        
        if config['use_context']:
            # Meaning meta-training on different dynamics
            from utils.transformer_nets import DynamicsTransformer, LayoutClassifier
            
            # layout_classifier_def = LayoutClassifier(hidden_dims=(512, 512, 512), num_layouts=config['number_of_meta_envs'])
            dynamics_def = DynamicsTransformer(
                num_layers=config['n_blocks'],
                num_heads=config['n_heads'],
                causal=False, # make permutation-equivariant
                emb_dim=config['h_dim'],
                mlp_dim=config['h_dim'],
                dropout_rate=0.0,
                attention_dropout_rate=0.0,
                action_dim=action_dim,
                num_layouts=config['number_of_meta_envs']
            )
            network_info.update(
                dynamic_transformer=(dynamics_def, (ex_observations[None], jnp.atleast_3d(ex_actions),
                                                    ex_observations[None], None, True, True))
            )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_critic'] = params['modules_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))
