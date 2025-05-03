import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import FBActor, FValue, BValue, FValueDiscrete
from functools import partial

class ForwardBackwardAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def fb_loss(self, batch, z_latent, dynamics_embedding, grad_params, rng):
        rng, sample_rng = jax.random.split(rng)
        
        target_F1, target_F2 = self.network.select('target_f_value')(batch['next_observations'], z_latent, dynamics_embedding=dynamics_embedding)
        next_Q1 = jnp.einsum('sda, sd -> sa', target_F1, z_latent)
        next_Q2 = jnp.einsum('sda, sd -> sa', target_F2, z_latent)
        next_Q = jnp.minimum(next_Q1, next_Q2)
        
        if self.config['boltzmann']:
            pi = jax.nn.softmax(next_Q / 300, axis=-1)
            target_F1 = jnp.einsum("sa, sda -> sd", pi, target_F1) # batch x z_dim
            target_F2 = jnp.einsum("sa, sda -> sd", pi, target_F2)
            next_Q = jnp.einsum("sa, sa -> s", pi, next_Q)
        else:
            next_action = next_Q.argmax(-1, keepdims=True)
            next_idx = next_action[:, None, :].repeat(repeats=z_latent.shape[-1], axis=1).astype(jnp.int16)
            target_F1 = jnp.take_along_axis(target_F1, next_idx, axis=-1).squeeze()
            target_F2 = jnp.take_along_axis(target_F2, next_idx, axis=-1).squeeze()
            next_Q = next_Q.max(-1)
            
        target_B = self.network.select('target_b_value')(batch['next_observations'], dynamics_embedding=None)
        target_M1 = target_F1 @ target_B.T
        target_M2 = target_F2 @ target_B.T
        target_M = jnp.minimum(target_M1, target_M2)
        
        cur_idx = batch['actions'][..., None].repeat(repeats=z_latent.shape[-1], axis=1).astype(jnp.int16)[:, :, None]
        F1, F2 = self.network.select('f_value')(batch['observations'], z_latent, dynamics_embedding=dynamics_embedding, params=grad_params)
        F1 = jnp.take_along_axis(F1, cur_idx, axis=-1).squeeze()
        F2 = jnp.take_along_axis(F2, cur_idx, axis=-1).squeeze()
        B = self.network.select('b_value')(batch['next_observations'], dynamics_embedding=None, params=grad_params)
        M1 = F1 @ B.T
        M2 = F2 @ B.T
        
        I = np.eye(batch['observations'].shape[0], dtype=bool)
        off_diag = ~I

        fb_offdiag = 0.5 * sum(((M - self.config['discount'] * target_M)[off_diag] ** 2).mean() for M in [M1, M2])
        fb_diag = -sum(jnp.diag(M).mean() for M in [M1, M2])
        fb_loss = fb_diag + fb_offdiag
        
        # Orthonormality loss
        cov_b = (B @ B.T)
        ort_loss_diag = -2 * jnp.diag(cov_b).mean()
        ort_loss_offdiag = (cov_b[off_diag] ** 2).mean()
        ort_b_loss = ort_loss_diag + ort_loss_offdiag
        total_loss = fb_loss + ort_b_loss
        
        correct_ort = jnp.argmax(cov_b, axis=-1) == jnp.argmax(I, axis=-1)
        
        return total_loss, {
            "fb_loss": total_loss,
            "z_norm": jnp.linalg.norm(z_latent, axis=-1).mean(),
            "correct_b_ort": correct_ort.sum(),
            # ORTHONORMALITY METRICS
            "mean_diag": jnp.diag(cov_b).mean(), # should increase
            "mean_off_diag": cov_b[off_diag].mean(), # should decrease
            "fb_offdiag_loss": fb_offdiag,
            "fb_diag_loss": fb_diag,
        }
    
    def context_encoder_loss(self, batch, grad_params):
        dynamics_embedding_mean, dynamics_embedding_std = self.network.select('dynamic_transformer')(batch['traj_states'], batch['traj_actions'],
                                                                                batch['traj_next_states'], train=True, params=grad_params)
        dynamics_embedding = dynamics_embedding_mean + jax.random.normal(key=self.rng, shape=dynamics_embedding_mean.shape) * jnp.exp(dynamics_embedding_std)
        dynamics_embedding = jnp.tile(dynamics_embedding[:, None], reps=(1, batch['traj_states'].shape[1], 1))
        next_state_pred = self.network.select('next_state_pred')(batch['traj_states'], batch['traj_actions'], dynamics_embedding, params=grad_params)
        loss = optax.squared_error(next_state_pred, batch['traj_next_states']).mean()
        return loss, {"context_embedding_loss": loss}
    
    def total_loss(self, batch, latent_z, grad_params, train_context_embedding, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, fb_recon_rng = jax.random.split(rng, 3)
        actor_loss = 0.0
        fb_loss = 0.0
        
        dynamics_embedding_mean, dynamics_embedding_log_std = self.network.select('dynamic_transformer')(batch['traj_states'],
                                                                                                        batch['traj_actions'],
                                                                                                        batch['traj_next_states'], train=False)
        dynamics_embedding = dynamics_embedding_mean + jax.random.normal(key=self.rng, shape=dynamics_embedding_mean.shape) * jnp.exp(dynamics_embedding_log_std)

        if not train_context_embedding:
            fb_loss, fb_info = self.fb_loss(batch, latent_z, jax.lax.stop_gradient(dynamics_embedding), grad_params, fb_recon_rng)
            for k, v in fb_info.items():
                info[f'fb/{k}'] = v

            if not self.config['discrete']:
                actor_loss, actor_info = self.actor_loss(batch, latent_z, dynamics_embedding, grad_params, actor_rng)
                for k, v in actor_info.items():
                    info[f'actor/{k}'] = v

        trans_loss = 0.0
        if train_context_embedding:
            trans_loss, trans_info = self.context_encoder_loss(batch, grad_params)
            for k, v in trans_info.items():
                info[f'context_encoder_loss/{k}'] = v
        
        loss = fb_loss + actor_loss + trans_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params
    
    @partial(jax.jit, static_argnames=('train_context_embedding'))
    def update(self, batch, train_context_embedding=True):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)
        z = self.sample_mixed_z(batch, self.config['z_dim'], new_rng)
        
        def loss_fn(grad_params):
            return self.total_loss(batch, z, grad_params, train_context_embedding=train_context_embedding, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)        
        self.target_update(new_network, 'f_value')
        self.target_update(new_network, 'b_value')
        
        return self.replace(network=new_network, rng=new_rng), info

    def project_z(self, z):
        return z * jnp.sqrt(z.shape[-1]) / jnp.linalg.norm(z, axis=-1, keepdims=True)

    def sample_z(
            self,
            batch_size: int,
            dim: int,
            key: jax.random.PRNGKey,
            custom_anchor: jnp.ndarray,
            kappa: float = 20.0
        ) -> jnp.ndarray:
            """
            Sample z vectors by first drawing VMF around north pole, then reflecting to align
            with the batch of custom_anchor vectors.
            custom_anchor: shape (batch_size, dim)
            Returns z_post of shape (batch_size, dim).
            """
            # Normalize anchor per-sample
            anchor = custom_anchor / (jnp.linalg.norm(custom_anchor, axis=-1, keepdims=True) + 1e-6)
            # Draw VMF samples around [1,0,..]
            samples = self._vmf_sample_north_pole(key, batch_size, dim, kappa)
            # Compute one Householder matrix per-sample
            rotation = jax.vmap(self._nd_householder_rotation)(anchor)  # (batch_size, dim, dim)
            # Apply rotation to each sample
            z_post = jnp.einsum('bij,bj->bi', rotation, samples)
            # Project to sphere
            return self.project_z(z_post)

    def _vmf_sample_north_pole(self, key, batch_size, dim, kappa):
        # Wood's algorithm for dimensions >= 3
        v_key, w_key = jax.random.split(key)
        
        # Step 1: Sample uniform from S^{d-2}
        uniform_samples = jax.random.normal(v_key, (batch_size, dim-1))
        uniform_samples /= jnp.linalg.norm(uniform_samples, axis=-1, keepdims=True)
        
        # Step 2: Sample radial component (corrected implementation)
        b = (-2*kappa + jnp.sqrt(4*kappa**2 + (dim-1)**2)) / (dim-1)
        t = jax.random.beta(w_key, (dim-1)/2, (dim-1)/2, (batch_size,))
        
        # Compute w using full Wood's formula
        numerator = 1 - (1 + b) * t
        denominator = 1 - (1 - b) * t + 1e-6
        w = numerator / denominator
        
        # Combine components
        w = w[..., jnp.newaxis]
        v = jnp.sqrt(1 - w**2) * uniform_samples
        return jnp.concatenate([w, v], axis=-1)

    def _nd_householder_rotation(self, anchor: jnp.ndarray) -> jnp.ndarray:
        """Householder reflector that maps north-pole e=[1,0,..] to anchor"""
        dim = anchor.shape[-1]
        e = jnp.concatenate([jnp.ones((1,)), jnp.zeros((dim - 1,))], axis=0)
        u = e - anchor
        u_norm = jnp.linalg.norm(u)
        def reflect(_):
            uu = u / (u_norm + 1e-6)
            return jnp.eye(dim) - 2 * jnp.outer(uu, uu)
        return jax.lax.cond(u_norm < 1e-6, lambda _: jnp.eye(dim), reflect, operand=None)

    def sample_mixed_z(self, batch, latent_dim, key):
        batch_size = batch['observations'].shape[0]
        dynamics_embedding_mean, dynamics_embedding_std = self.network.select('dynamic_transformer')(batch['traj_states'],
                                                                                                    batch['traj_actions'],
                                                                                                    batch['traj_next_states'], train=False)
        dynamics_embedding = dynamics_embedding_mean + jax.random.normal(key=key, shape=dynamics_embedding_mean.shape) * jnp.exp(dynamics_embedding_std)
        z = self.sample_z(batch_size, latent_dim, key, dynamics_embedding, kappa=self.config['kappa'])
        b_goals = self.network.select('b_value')(goal=batch['actor_goals'], dynamics_embedding=None)
        mask = jax.random.uniform(key, shape=(batch_size, 1)) < self.config['z_mix_ratio']
        z = jnp.where(mask, z, b_goals)
        return z
    
    @jax.jit
    def infer_z(self, obs, mdp_num=None, dynamics_embedding=None, rewards=None):
        z = self.network.select('b_value')(goal=obs, mdp_num=None, dynamics_embedding=None)
        return z
    
    @jax.jit
    def sample_actions(
        self,
        observations,
        latent_z,
        seed=None,
        mdp_num=None,
        dynamics_embedding=None,
        temperature=1.0,
    ):
        latent_z = jnp.atleast_2d(latent_z)
        Q = self.predict_q(observations, latent_z, mdp_num=None, dynamics_embedding=dynamics_embedding)
        actions = jnp.argmax(Q, axis=-1)
        return actions

    def predict_q(
        self, observation, z, action=None, mdp_num=None, dynamics_embedding=None
    ):
        observation = jnp.atleast_2d(observation)
        F1, F2 = self.network.select('f_value')(observation, z, mdp_num=None, dynamics_embedding=dynamics_embedding)
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
        rng = jax.random.key(seed)
        rng, init_rng = jax.random.split(rng, 2)
        
        ex_goals = ex_observations
        action_dim = int(ex_actions.max() + 1)
        
        forward_def = FValueDiscrete(
            action_dim=action_dim,
            latent_z_dim=config['z_dim'],
            f_hidden_dims=config['f_hidden_dims'],
            f_layer_norm=config['f_layer_norm'],
        )
        backward_def = BValue(
            latent_z_dim=config['z_dim'],
            b_layer_norm=config['b_layer_norm'],
            b_hidden_dims=config['b_hidden_dims'],
        )
        latent_z = jax.random.normal(init_rng, shape=(1, config['z_dim']))
        latent_z = latent_z * jnp.sqrt(latent_z.shape[-1]) / jnp.linalg.norm(latent_z, axis=-1, keepdims=True)
        
        mdp_layout_one_hot = None
        dynamics_embedding = jnp.zeros((1, config['output_dim']))
        network_info = dict(
            f_value=(forward_def, (ex_observations, latent_z, None, mdp_layout_one_hot, dynamics_embedding)),
            target_f_value=(copy.deepcopy(forward_def), (ex_observations, latent_z, None, mdp_layout_one_hot, dynamics_embedding)),
            b_value=(backward_def, (ex_goals, None, mdp_layout_one_hot, None)),
            target_b_value = (copy.deepcopy(backward_def), (ex_goals, None, mdp_layout_one_hot, None)),
        )
        if config['use_context']:
            from utils.transformer_nets import DynamicsTransformer, NextStatePrediction
            from utils.networks import AnchorProjector
            
            next_state_pred_def = NextStatePrediction(hidden_dims=config['world_pred_hidden'], out_dim=ex_observations.shape[-1])
            dynamics_def = DynamicsTransformer(
                num_layers=config['n_blocks'],
                num_heads=config['n_heads'],
                out_dim=config['output_dim'],
                action_dim=action_dim,
                causal=False,
                emb_dim=config['emb_dim'],
                mlp_dim=config['mlp_dim'],
                dropout_rate=0.0,
                attention_dropout_rate=0.0,
                context_len=config['context_len']
            )
            network_info.update(
                dynamic_transformer=(dynamics_def, (jnp.zeros((1, 1, ex_observations.shape[-1])), jnp.zeros((1, 1, ex_actions.shape[-1])),
                                                    jnp.zeros((1, 1, ex_observations.shape[-1])), True, True))
            )
            network_info.update(
                next_state_pred=(next_state_pred_def, (jnp.zeros((1, 1, ex_observations.shape[-1])), jnp.zeros((1, 1, ex_actions.shape[-1])),
                                                    jnp.zeros((1, 1, config['output_dim']))))
            )#####
            # anchor_proj_def = AnchorProjector(
            #         latent_z_dim=config['z_dim'],
            #         b_hidden_dims=[256, 256, 256]
            # )
            # network_info.update(
            #     anchor_proj=(anchor_proj_def, (dynamics_embedding,))
            # )
            # vq_layer = VectorQuantizer(
            #         num_embeddings=config['agent']['num_codebook_entries'],
            #         embedding_dim=config['agent']['dynamics_embed_dim'],
            #         commitment_cost=config['agent']['commitment_cost']
            # )
            # network_info.update(
            #     vq_layer=(vq_layer, )
            # )
            
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.chain(optax.clip_by_global_norm(1.0),
                                optax.adam(learning_rate=config['lr']))
        
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)
        
        params = network.params
        params['modules_target_f_value'] = params['modules_f_value']
        params['modules_target_b_value'] = params['modules_b_value']
        
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))