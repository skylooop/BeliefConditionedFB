import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import FBActor, FValue, BValue, FValueDiscrete

class ForwardBackwardAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def fb_loss(self, batch, z_latent, grad_params, rng):
        rng, sample_rng = jax.random.split(rng)
    
        # z_latent = jax.lax.stop_gradient(z_latent)
        # Target M for continuous actor
        if not self.config['discrete']:
            # Target M
            next_dist = self.network.select('actor')(batch['next_observations'], z_latent)
            next_actions = next_dist.sample(seed=sample_rng)
            target_F1, target_F2 = self.network.select('target_f_value')(batch['next_observations'], next_actions, z_latent)
            target_B = self.network.select('target_b_value')(batch['next_observations'])
            target_M1 = target_F1 @ target_B.T
            target_M2 = target_F2 @ target_B.T
            target_M = jnp.minimum(target_M1, target_M2)
            
            # Cur M
            F1, F2 = self.network.select('f_value')(batch['observations'], batch['actions'], z_latent, params=grad_params)
            B = self.network.select('b_value')(batch['next_observations'], params=grad_params)
            M1 = F1 @ B.T
            M2 = F2 @ B.T
        else:
            target_F1, target_F2 = self.network.select('target_f_value')(batch['next_observations'], z_latent)
            next_Q1 = jnp.einsum('sda, sd -> sa', target_F1, z_latent)
            next_Q2 = jnp.einsum('sda, sd -> sa', target_F2, z_latent)
            next_Q = jnp.minimum(next_Q1, next_Q2)
            
            if self.config['boltzmann']:
                pass
                # pi = F.softmax(next_Q / self.cfg.temp, dim=-1)
                # target_F1, target_F2 = [torch.einsum("sa, sda -> sd", pi, Fi) for Fi in [target_F1, target_F2]] # batch x z_dim
                # next_Q = torch.einsum("sa, sa -> s", pi, next_Q)
            else:
                next_action = next_Q.argmax(-1, keepdims=True)
                next_idx = next_action[:, None, :].repeat(repeats=z_latent.shape[-1], axis=1).astype('int8')
                target_F1 = jnp.take_along_axis(target_F1, next_idx, axis=-1).squeeze()
                target_F2 = jnp.take_along_axis(target_F2, next_idx, axis=-1).squeeze()
                target_B = self.network.select('target_b_value')(batch['next_observations'])
                target_M1 = target_F1 @ target_B.T
                target_M2 = target_F2 @ target_B.T
                target_M = jnp.minimum(target_M1, target_M2)
                
                cur_idx = batch['actions'][:, None, :].repeat(repeats=z_latent.shape[-1], axis=1).astype('int8')
                F1, F2 = self.network.select('f_value')(batch['observations'], z_latent, params=grad_params)
                F1 = jnp.take_along_axis(F1, cur_idx, axis=-1).squeeze()
                F2 = jnp.take_along_axis(F2, cur_idx, axis=-1).squeeze()
                B = self.network.select('b_value')(batch['next_observations'], params=grad_params)
                M1 = F1 @ B.T
                M2 = F2 @ B.T
        
        I = np.eye(batch['observations'].shape[0], dtype=bool)
        off_diag = ~I

        fb_offdiag = 0.5 * sum(jnp.pow((M - self.config['discount'] * target_M)[off_diag], 2).mean() for M in [M1, M2])
        #fb_diag = -sum(jnp.diag((M - self.config['discount'] * target_M)).mean() for M in [M1, M2])
        fb_diag = -sum(jnp.diag(M).mean() for M in [M1, M2])
        fb_loss = fb_diag + fb_offdiag
        
        # Orthonormality loss
        cov_b = B @ jax.lax.stop_gradient(B.T)
        ort_loss_diag = -2 * jnp.diag(cov_b).mean()
        ort_loss_offdiag = jnp.pow(cov_b[off_diag], 2).mean()
        ort_b_loss = ort_loss_diag + ort_loss_offdiag
        if len(B.shape) == 2:
            B = B[None, ...]
        logits = jnp.einsum('eik,ejk->ije', B, B) / jnp.sqrt(B.shape[-1]) 
        contrastive_loss = jax.vmap(
            lambda _logits: optax.sigmoid_binary_cross_entropy(logits=_logits, labels=I),
            in_axes=-1,
            out_axes=-1,
        )(logits)
        ort_b_loss = jnp.mean(contrastive_loss)
        total_loss = fb_loss + ort_b_loss
        
        correct_fb = jnp.argmax(M1, axis=1) == jnp.argmax(I, axis=1)
        correct_ort = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        
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
            "M": M1.mean(),
        }

    def actor_loss(self, batch, z_latent, grad_params, rng):
        rng, sample_rng = jax.random.split(rng)
    
        dist = self.network.select('actor')(batch['observations'], z_latent, params=grad_params)
        actions, log_probs = dist.sample_and_log_prob(seed=sample_rng)

        F1, F2 = self.network.select('f_value')(batch['observations'], actions, z_latent)
        Q1 = (F1 * z_latent).sum(-1)
        Q2 = (F2 * z_latent).sum(-1)
        Q = jnp.minimum(Q1, Q2)
        actor_loss = (0.1 * log_probs - Q).mean()
        #actor_loss = -Q.mean()
        return actor_loss, {
            'mean_action': actions.mean(),
            'actor_q': Q.mean(),
            'actor_loss': actor_loss,
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

        actor_loss = 0.0
        if not self.config['discrete']:
            actor_loss, actor_info = self.actor_loss(batch, latent_z, grad_params, actor_rng)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v

        loss = fb_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree.map(
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
        
        new_target_params = self.target_update(new_network, 'f_value')
        new_network.params['modules_target_f_value'] = new_target_params
        new_target_params = self.target_update(new_network, 'b_value')
        new_network.params['modules_target_b_value'] = new_target_params
        
        return self.replace(network=new_network, rng=new_rng), info

    def project_z(self, z):
        return z * jnp.sqrt(z.shape[-1]) / jnp.linalg.norm(z, axis=-1, keepdims=True)
    
    def sample_z(self, batch_size, latent_dim, key):
        z = jax.random.normal(shape=(batch_size, latent_dim), key=key)
        return self.project_z(z)
    
    def sample_mixed_z(self, batch, latent_dim, key):
        batch_size = batch['observations'].shape[0]
        z = self.sample_z(batch_size, latent_dim, key)
        b_goals = self.network.select('b_value')(goal=batch['next_observations']) # batch['value_goals'] - First exp
        mask = jax.random.uniform(key, shape=(batch_size, 1)) < self.config['z_mix_ratio']
        z = jnp.where(mask, b_goals, z)
        return z
    
    @jax.jit
    def infer_z(self, obs, rewards=None):
        """
        If reards are None -> treat as goal-conditioned
        """    
        z = self.network.select('b_value')(goal=obs)
        # if rewards is not None:
        #     z = (rewards.T @ z)
        #z = self.project_z(z)
        return z
        
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
            actions = jnp.argmax(Q, -1)
            
        return actions

    def predict_q(
        self, observation, z, action=None
    ):
        if not self.config['discrete']:
            F1, F2 = self.network.select('f_value')(observation, action, z)
            Q1 = (F1 * z).sum(-1)
            Q2 = (F2 * z).sum(-1)
        else:
            observation = jnp.atleast_2d(observation)
            F1, F2 = self.network.select('f_value')(observation, z)
            Q1 = jnp.einsum('sda, sd -> sa', F1, z)
            Q2 = jnp.einsum('sda, sd -> sa', F2, z)
        Q = jnp.minimum(Q1, Q2)

        return Q
    
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
        if not config['discrete']:
            forward_def = FValue(
                latent_z_dim=config['z_dim'],
                preprocess=True,
                layer_norm_only_first=config['fb_layer_norm_only_first'],
                fb_forward_hidden_dims=config['fb_forward_hidden_dims'],
                fb_forward_preprocessor_hidden_dims=config['fb_forward_preprocessor_hidden_dims'],
                fb_forward_layer_norm=config['fb_forward_layer_norm'],
                fb_preprocessor_layer_norm=config['fb_preprocessor_layer_norm']
        )
        else:
            forward_def = FValueDiscrete(
                action_dim=action_dim,
                latent_z_dim=config['z_dim'],
                fb_forward_hidden_dims=config['fb_forward_hidden_dims'],
                fb_hidden_dims=config['fb_forward_preprocessor_hidden_dims'],
                fb_forward_layer_norm=config['fb_forward_layer_norm'],
            )
        backward_def = BValue(
            latent_z_dim=config['z_dim'],
            fb_backward_layer_norm=config['fb_backward_layer_norm'],
            fb_backward_hidden_dims=config['fb_backward_hidden_dims'],
        )
        actor_def = None
        if not config['discrete']:
            actor_def = FBActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                tanh_squash=config['tanh_squash'],
                state_dependent_std=config['state_dependent_std'],
                actor_preprocessor_hidden_dims=config['actor_preprocessor_hidden_dims'],
                const_std=config['const_std'],
                final_fc_init_scale=config['actor_fc_scale'],
            )
        latent_z = jax.random.normal(init_rng, shape=(1, config['z_dim']))
        latent_z = latent_z * jnp.sqrt(latent_z.shape[-1]) / jnp.linalg.norm(latent_z, axis=-1, keepdims=True)
        
        network_info = dict(
            f_value=(forward_def, (ex_observations, ex_actions, latent_z)) if not config['discrete'] \
                else (forward_def, (ex_observations, latent_z)),
            target_f_value=(copy.deepcopy(forward_def), (ex_observations, ex_actions, latent_z)) if not config['discrete'] \
                else (copy.deepcopy(forward_def), (ex_observations, latent_z)),
            b_value=(backward_def, (ex_goals, )),
            target_b_value = (copy.deepcopy(backward_def), (ex_goals, )),
            #actor=(actor_def, (ex_observations, latent_z))
        )
        if not config['discrete']:
            network_info.update({"actor": (actor_def, (ex_observations, latent_z))})
            
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)
        params = network.params
        params['modules_target_f_value'] = params['modules_f_value']
        params['modules_target_b_value'] = params['modules_b_value']
        network = network.replace(params=params)
        
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))

def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Most of hyperparams from zero-shot rl from high-quality data
            # https://arxiv.org/pdf/2309.15178
            agent_name='fb',  # Agent name.
            discrete=True,
            lr=1e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            # FB Specific
            z_dim=50, # 100 for maze env, 50 for others
            fb_forward_hidden_dims=(1024, 1024),  # Value network hidden dimensions.
            fb_layer_norm_only_first=True, # use tanh + layernorm in first layer
            fb_forward_layer_norm=True,  # Whether to use layer normalization.
            fb_preprocessor_layer_norm=False,
            fb_forward_preprocessor_hidden_dims=(512, 512),
            fb_backward_hidden_dims=(256, 256, 256),  # Value network hidden dimensions.
            fb_backward_layer_norm=False,  # Whether to use layer normalization.
            z_mix_ratio=0.5,
            # Actor
            actor_hidden_dims=(1024, 1024),  # Actor network hidden dimensions.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            boltzmann=False, # TODO: add later maybe?
            # MISC
            discount=0.99,  # Discount factor. 0.99 - for maze, 0.98 others
            tau=0.005,  # Target network update rate.
            tanh_squash=True,  # Whether to squash actions with tanh.
            state_dependent_std=False,  # Whether to use state-dependent standard deviations for actor.
            actor_fc_scale=0.01,  # Final layer initialization scale for actor.
            const_std=True,
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.5,  # Probability of using a random state as the value goal.
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
