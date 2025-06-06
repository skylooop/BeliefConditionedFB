from typing import Any, Optional, Sequence

import jax
import flax
import distrax
import flax.linen as nn
import jax.numpy as jnp

from typing import Any, Optional, Sequence, Callable
from flax.linen.initializers import lecun_normal, zeros
from jaxtyping import Array, Float
from flax.linen.initializers import orthogonal, constant

def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')

def orthogonal_scaling(scale=1.0):
    return nn.initializers.orthogonal(scale)

def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0},
        split_rngs={'params': True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )

class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x

def mish(x):
    return x * jnp.tanh(nn.softplus(x))

class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = orthogonal_scaling()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if i == 0:
                    x = nn.tanh(x)
                    x = nn.LayerNorm()(x)
                else:
                    x = self.activations(x)
                    if self.layer_norm:
                        x = nn.LayerNorm()(x)
        return x

class LengthNormalize(nn.Module):
    """Length normalization layer.

    It normalizes the input along the last dimension to have a length of sqrt(dim).
    """

    @nn.compact
    def __call__(self, x):
        return x / jnp.linalg.norm(x, axis=-1, keepdims=True) * jnp.sqrt(x.shape[-1])

class AnchorProjector(nn.Module):
    latent_z_dim: int
    b_hidden_dims: tuple
    
    @nn.compact
    def __call__(self, dynamics_embedding):
        x = dynamics_embedding
        for dim in self.b_hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.swish(x)  # Better activation than ReLU
        x = nn.Dense(self.latent_z_dim)(x)
        # q, _ = jnp.linalg.qr(x[None, :], mode='reduced')
        return x #q.squeeze()

class Param(nn.Module):
    """Scalar parameter module."""

    init_value: float = 0.0

    @nn.compact
    def __call__(self):
        return self.param('value', init_fn=lambda key: jnp.full((), self.init_value))


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class RunningMeanStd(flax.struct.PyTreeNode):
    """Running mean and standard deviation.

    Attributes:
        eps: Epsilon value to avoid division by zero.
        mean: Running mean.
        var: Running variance.
        clip_max: Clip value after normalization.
        count: Number of samples.
    """

    eps: Any = 1e-6
    mean: Any = 1.0
    var: Any = 1.0
    clip_max: Any = 10.0
    count: int = 0

    def normalize(self, batch):
        batch = (batch - self.mean) / jnp.sqrt(self.var + self.eps)
        batch = jnp.clip(batch, -self.clip_max, self.clip_max)
        return batch

    def unnormalize(self, batch):
        return batch * jnp.sqrt(self.var + self.eps) + self.mean

    def update(self, batch):
        batch_mean, batch_var = jnp.mean(batch, axis=0), jnp.var(batch, axis=0)
        batch_count = len(batch)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        return self.replace(mean=new_mean, var=new_var, count=total_count)

class MLPWithFiLM(nn.Module):
    """MLP with FiLM conditioning applied to all hidden layers.
    
    Attributes:
        hidden_dims: List of hidden layer sizes.
        activations: Activation function (default: GELU).
        kernel_init: Kernel initializer for dense layers.
        layer_norm: Whether to apply layer normalization after activation.
    """
    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    kernel_init: Any = default_init()
    layer_norm: bool = False
    activate_final: bool = False
    
    @nn.compact
    def __call__(self, x, conditioning):
        """Apply the MLP with FiLM conditioning.
        
        Args:
            x: Input tensor (batch_size, input_dim), typically observations and goals.
            conditioning: Conditioning tensor (batch_size, conditioning_dim), e.g., dynamics_embedding.
            
        Returns:
            Output tensor (batch_size, hidden_dims[-1]).
        """
        def film_bias_init(rng, shape, dtype):
            """Initialize bias for FiLM parameters: ones for gamma, zeros for beta."""
            assert len(shape) == 1
            size = shape[0] // 2
            return jnp.concatenate([jnp.ones((size,), dtype), jnp.zeros((size,), dtype)])
        
        for i, size in enumerate(self.hidden_dims):
            # Generate FiLM parameters (gamma and beta) from conditioning
            film_params = nn.Dense(
                2 * size,
                kernel_init=nn.initializers.normal(stddev=0.02),
                bias_init=film_bias_init
            )(conditioning)  # Shape: (batch_size, 2 * size)
            gamma, beta = jnp.split(film_params, 2, axis=-1)  # Each: (batch_size, size)
            
            # Apply dense layer transformation
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)  # Shape: (batch_size, size)
            
            # Apply FiLM: gamma * x + beta
            x = gamma * x + beta
            
            # Apply activation
            if self.activate_final:
                x = self.activations(x)
            
            # Apply layer normalization if enabled
            if self.layer_norm:
                x = nn.LayerNorm()(x)
        return x

class GCActor(nn.Module):
    """Goal-conditioned actor.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        temperature=1.0,
    ):
        """Return the action distribution.

        Args:
            observations: Observations.
            goals: Goals (optional).
            goal_encoded: Whether the goals are already encoded.
            temperature: Scaling factor for the standard deviation.
        """
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution

class GCDiscreteActor(nn.Module):
    """Goal-conditioned actor for discrete actions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None
    layer_norm: bool = False
    use_film: bool = False
    
    def setup(self):
        if not self.use_film:
            self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        else:
            self.actor_net = MLPWithFiLM(
                    hidden_dims=self.hidden_dims,
                    activations=nn.gelu,
                    kernel_init=lecun_normal(),
                    layer_norm=self.layer_norm
                )
        self.logit_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))

    def __call__(
        self,
        observations,
        goals=None,
        mdp_num=None,
        dynamics_embedding=None,
        goal_encoded=False,
        temperature=1.0,
    ):
        """Return the action distribution.

        Args:
            observations: Observations.
            goals: Goals (optional).
            goal_encoded: Whether the goals are already encoded.
            temperature: Inverse scaling factor for the logits (set to 0 to get the argmax).
        """
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
        if goals is not None:
            inputs.append(goals)
        if dynamics_embedding is not None:
            inputs.append(dynamics_embedding)
        if mdp_num is not None:
            inputs.append(mdp_num)
        inputs = jnp.concatenate(inputs, axis=-1)
        if not self.use_film:
            outputs = self.actor_net(inputs)
        else:
            outputs = self.actor_net(inputs, dynamics_embedding)
        logits = self.logit_net(outputs)

        distribution = distrax.Categorical(logits=logits / jnp.maximum(1e-6, temperature))

        return distribution

class FValue(nn.Module):
    latent_z_dim: int
    f_hidden_dims: Sequence[int]
    f_preprocessor_hidden_dims: Sequence[int]
    preprocess: bool = True
    f_layer_norm: bool = False
    activate_final: bool = False
    
    def setup(self):
        mlp_module = MLP
        forward_mlp_module = ensemblize(MLP, 2)
        self.forward_map = forward_mlp_module((*self.f_hidden_dims, self.latent_z_dim), activate_final=False, layer_norm=self.f_layer_norm)
        
        self.forward_preprocessor_sa = mlp_module(hidden_dims=self.f_preprocessor_hidden_dims, layer_norm=self.f_layer_norm,
                                                activate_final=self.activate_final)
        self.forward_preprocessor_sz = mlp_module(hidden_dims=self.f_preprocessor_hidden_dims, layer_norm=self.f_layer_norm,
                                                activate_final=self.activate_final)
    
    def __call__(self, observations, actions, latent_z, context_z=None, mdp_num=None):
        if self.preprocess:
            processed_sa = self.forward_preprocessor_sa(jnp.concatenate([observations, actions], -1))
            processed_sz = self.forward_preprocessor_sz(jnp.concatenate([observations, latent_z], -1))
        else:
            processed_sa = jnp.concatenate([observations, actions], -1)
            processed_sz = jnp.concatenate([observations, latent_z], -1)
        input = [processed_sa, processed_sz]
        if mdp_num is not None:
            input.append(mdp_num)
        if context_z is not None:
            input.append(context_z)
        f1, f2 = self.forward_map(jnp.concatenate(input, -1))
        
        return f1, f2

class FValueDiscrete(nn.Module):
    latent_z_dim: int
    action_dim: int
    f_hidden_dims: Sequence[int]
    f_layer_norm: bool = False
    
    def setup(self):
        forward_mlp_module = ensemblize(MLP, 2)
        self.forward_map = forward_mlp_module((*self.f_hidden_dims, self.latent_z_dim * self.action_dim), activate_final=False,
                                    layer_norm=self.f_layer_norm)
        self.anchor_projection = nn.Dense(features=self.latent_z_dim)
        
    def __call__(self, observations, latent_z, context_z=None, mdp_num=None, dynamics_embedding=None):
        input = [observations, latent_z]
        if context_z is not None:
            input.append(context_z)
        if mdp_num is not None:
            input.append(mdp_num)
        if dynamics_embedding is not None:
            input.append(dynamics_embedding)
        f1, f2 = self.forward_map(jnp.concatenate(input, -1))        
        return f1.reshape(-1, self.latent_z_dim, self.action_dim), f2.reshape(-1, self.latent_z_dim, self.action_dim)

class BValue(nn.Module):
    latent_z_dim: int
    b_hidden_dims: Sequence[int]
    b_layer_norm: bool = False
    
    def setup(self):
        self.backward_map = MLP((*self.b_hidden_dims, self.latent_z_dim), activate_final=False, layer_norm=self.b_layer_norm,
                                kernel_init=orthogonal_scaling())
        self.project_onto = LengthNormalize()
        
    def __call__(self, goal, context_z=None, mdp_num=None, dynamics_embedding=None):
        input = [goal]
        if context_z is not None:
            input.append(context_z)
        if mdp_num is not None:
            input.append(mdp_num)
        if dynamics_embedding is not None:
            input.append(dynamics_embedding)
        backward = self.backward_map(jnp.concatenate(input, -1))
        project = self.project_onto(backward)
        return project
    
class FBDiscreteActor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    final_fc_init_scale: float = 1e-2
    layer_norm: bool = False
    
    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.logit_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))

    def __call__(
        self,
        observations,
        z_latent,
        temperature=1.0,
    ):
        inputs = jnp.concatenate([observations, z_latent], axis=-1)
        outputs = self.actor_net(inputs)

        logits = self.logit_net(outputs)

        distribution = distrax.Categorical(logits=logits / jnp.maximum(1e-6, temperature))

        return distribution

class FBActor(nn.Module):
    """

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    actor_preprocessor_hidden_dims: Sequence[int]
    actor_preprocessor_layer_norm: bool
    actor_preprocessor_activate_final: bool
    action_dim: int
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = True
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    
    def setup(self):
        self.forward_preprocessor_s = MLP((*self.actor_preprocessor_hidden_dims, ),
                                        activate_final=self.actor_preprocessor_activate_final, layer_norm=self.actor_preprocessor_layer_norm)
        self.forward_preprocessor_sz = MLP((*self.actor_preprocessor_hidden_dims, ),
                                        activate_final=self.actor_preprocessor_activate_final, layer_norm=self.actor_preprocessor_layer_norm)
        
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=False)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        latent_z,
        temperature=1.0,
    ):
        processed_s = self.forward_preprocessor_s(observations)
        processed_sz = self.forward_preprocessor_sz(jnp.concatenate([observations, latent_z], -1))
        inputs = jnp.concatenate([processed_s, processed_sz], axis=-1)
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means) * 0.3
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution
    
    
import flax.linen as nn
import jax.numpy as jnp
import distrax

class RecurrentActor(nn.Module):
    hidden_dim: int
    action_dim: int
    rnn_type: str = 'lstm'
    layer_norm: bool = False
    tanh_squash: bool = True
    state_dependent_std: bool = True

    @nn.compact
    def __call__(self, inputs, hidden_state):
        embedding = nn.Dense(
            128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(inputs)
        embedding = jax.nn.relu(embedding)
        
        if self.rnn_type == 'lstm':
            lstm = nn.LSTMCell(self.hidden_dim)
            carry, outputs = nn.scan(
                nn.LSTMCell.call,
                variable_broadcast="params",
                split_rngs={"params": False}
            )(lstm, hidden_state, inputs)
        elif self.rnn_type == 'gru':
            gru = nn.GRUCell(self.hidden_dim)
            carry, outputs = nn.scan(
                nn.GRUCell.call,
                variable_broadcast="params",
                split_rngs={"params": False}
            )(gru, hidden_state, inputs)
        else:
            raise ValueError(f"Unsupported RNN type: {self.rnn_type}")

        if self.layer_norm:
            outputs = nn.LayerNorm()(outputs)

        mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.xavier_uniform())(outputs)
        if self.state_dependent_std:
            log_std = nn.Dense(self.action_dim, kernel_init=nn.initializers.xavier_uniform())(outputs)
        else:
            log_std = self.param('log_std', nn.initializers.zeros, (self.action_dim,))

        log_std = jnp.clip(log_std, -20, 2)
        std = jnp.exp(log_std)

        if self.tanh_squash:
            distribution = distrax.Transformed(
                distrax.MultivariateNormalDiag(mean, std),
                distrax.Block(distrax.Tanh(), 1)
            )
        else:
            distribution = distrax.MultivariateNormalDiag(mean, std)

        return distribution, carry

class RecurrentValue(nn.Module):
    hidden_dim: int
    rnn_type: str = 'lstm'
    layer_norm: bool = False

    @nn.compact
    def __call__(self, inputs, hidden_state):
        if self.rnn_type == 'lstm':
            lstm = nn.LSTMCell(self.hidden_dim)
            carry, outputs = nn.scan(
                nn.LSTMCell.call,
                variable_broadcast="params",
                split_rngs={"params": False}
            )(lstm, hidden_state, inputs)
        elif self.rnn_type == 'gru':
            gru = nn.GRUCell(self.hidden_dim)
            carry, outputs = nn.scan(
                nn.GRUCell.call,
                variable_broadcast="params",
                split_rngs={"params": False}
            )(gru, hidden_state, inputs)
        else:
            raise ValueError(f"Unsupported RNN type: {self.rnn_type}")

        if self.layer_norm:
            outputs = nn.LayerNorm()(outputs)

        value = nn.Dense(1, kernel_init=nn.initializers.xavier_uniform())(outputs)
        return value.squeeze(-1), carry    
    
class GCValue(nn.Module):
    """Goal-conditioned value/critic function.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        ensemble: Whether to ensemble the value function.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    ensemble: bool = True
    gc_encoder: nn.Module = None
    use_film: bool = False
    
    def setup(self):
        mlp_module = MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)
        if self.use_film:
            mlp_module = ensemblize(MLPWithFiLM, 2)
        value_net = mlp_module((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)
        self.value_net = value_net

    def __call__(self, observations, goals=None, actions=None, mdp_num=None, dynamics_embedding=None):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals (optional).
            actions: Actions (optional).
        """
        if self.gc_encoder is not None:
            inputs = [self.gc_encoder(observations, goals)]
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
        if actions is not None:
            inputs.append(actions)
        if mdp_num is not None:
            inputs.append(mdp_num)
        if not self.use_film:
            if dynamics_embedding is not None:
                inputs.append(dynamics_embedding)
            inputs = jnp.concatenate(inputs, axis=-1)
            v = self.value_net(inputs).squeeze(-1)
        else:
            inputs = jnp.concatenate(inputs, axis=-1)
            v = self.value_net(inputs, dynamics_embedding)
        return v


class GCDiscreteCritic(GCValue):
    """Goal-conditioned critic for discrete actions."""

    action_dim: int = None

    def __call__(self, observations, goals=None, actions=None, mdp_num=None, dynamics_embedding=None):
        actions = jnp.eye(self.action_dim)[actions]
        return super().__call__(observations, goals, actions, mdp_num, dynamics_embedding)


class GCBilinearValue(nn.Module):
    """Goal-conditioned bilinear value/critic function.

    This module computes the value function as V(s, g) = phi(s)^T psi(g) / sqrt(d) or the critic function as
    Q(s, a, g) = phi(s, a)^T psi(g) / sqrt(d), where phi and psi output d-dimensional vectors.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        layer_norm: Whether to apply layer normalization.
        ensemble: Whether to ensemble the value function.
        value_exp: Whether to exponentiate the value. Useful for contrastive learning.
        state_encoder: Optional state encoder.
        goal_encoder: Optional goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    ensemble: bool = True
    value_exp: bool = False
    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None

    def setup(self) -> None:
        mlp_module = MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)

        self.phi = mlp_module((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)
        self.psi = mlp_module((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, goals, actions=None, info=False):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals.
            actions: Actions (optional).
            info: Whether to additionally return the representations phi and psi.
        """
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)

        if actions is None:
            phi_inputs = observations
        else:
            phi_inputs = jnp.concatenate([observations, actions], axis=-1)

        phi = self.phi(phi_inputs)
        psi = self.psi(goals)

        v = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)

        if self.value_exp:
            v = jnp.exp(v)

        if info:
            return v, phi, psi
        else:
            return v


class GCDiscreteBilinearCritic(GCBilinearValue):
    """Goal-conditioned bilinear critic for discrete actions."""

    action_dim: int = None

    def __call__(self, observations, goals=None, actions=None, info=False):
        actions = jnp.eye(self.action_dim)[actions]
        return super().__call__(observations, goals, actions, info)


class GCMRNValue(nn.Module):
    """Metric residual network (MRN) value function.

    This module computes the value function as the sum of a symmetric Euclidean distance and an asymmetric
    L^infinity-based quasimetric.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional state/goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    encoder: nn.Module = None

    def setup(self) -> None:
        self.phi = MLP((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, goals, is_phi=False, info=False):
        """Return the MRN value function.

        Args:
            observations: Observations.
            goals: Goals.
            is_phi: Whether the inputs are already encoded by phi.
            info: Whether to additionally return the representations phi_s and phi_g.
        """
        if is_phi:
            phi_s = observations
            phi_g = goals
        else:
            if self.encoder is not None:
                observations = self.encoder(observations)
                goals = self.encoder(goals)
            phi_s = self.phi(observations)
            phi_g = self.phi(goals)

        sym_s = phi_s[..., : self.latent_dim // 2]
        sym_g = phi_g[..., : self.latent_dim // 2]
        asym_s = phi_s[..., self.latent_dim // 2 :]
        asym_g = phi_g[..., self.latent_dim // 2 :]
        squared_dist = ((sym_s - sym_g) ** 2).sum(axis=-1)
        quasi = jax.nn.relu((asym_s - asym_g).max(axis=-1))
        v = jnp.sqrt(jnp.maximum(squared_dist, 1e-12)) + quasi

        if info:
            return v, phi_s, phi_g
        else:
            return v


class GCIQEValue(nn.Module):
    """Interval quasimetric embedding (IQE) value function.

    This module computes the value function as an IQE-based quasimetric.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        dim_per_component: Dimension of each component in IQE (i.e., number of intervals in each group).
        layer_norm: Whether to apply layer normalization.
        encoder: Optional state/goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    dim_per_component: int
    layer_norm: bool = True
    encoder: nn.Module = None

    def setup(self) -> None:
        self.phi = MLP((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)
        self.alpha = Param()

    def __call__(self, observations, goals, is_phi=False, info=False):
        """Return the IQE value function.

        Args:
            observations: Observations.
            goals: Goals.
            is_phi: Whether the inputs are already encoded by phi.
            info: Whether to additionally return the representations phi_s and phi_g.
        """
        alpha = jax.nn.sigmoid(self.alpha())
        if is_phi:
            phi_s = observations
            phi_g = goals
        else:
            if self.encoder is not None:
                observations = self.encoder(observations)
                goals = self.encoder(goals)
            phi_s = self.phi(observations)
            phi_g = self.phi(goals)

        x = jnp.reshape(phi_s, (*phi_s.shape[:-1], -1, self.dim_per_component))
        y = jnp.reshape(phi_g, (*phi_g.shape[:-1], -1, self.dim_per_component))
        valid = x < y
        xy = jnp.concatenate(jnp.broadcast_arrays(x, y), axis=-1)
        ixy = xy.argsort(axis=-1)
        sxy = jnp.take_along_axis(xy, ixy, axis=-1)
        neg_inc_copies = jnp.take_along_axis(valid, ixy % self.dim_per_component, axis=-1) * jnp.where(
            ixy < self.dim_per_component, -1, 1
        )
        neg_inp_copies = jnp.cumsum(neg_inc_copies, axis=-1)
        neg_f = -1.0 * (neg_inp_copies < 0)
        neg_incf = jnp.concatenate([neg_f[..., :1], neg_f[..., 1:] - neg_f[..., :-1]], axis=-1)
        components = (sxy * neg_incf).sum(axis=-1)
        v = alpha * components.mean(axis=-1) + (1 - alpha) * components.max(axis=-1)

        if info:
            return v, phi_s, phi_g
        else:
            return v

class Actor(nn.Module):
    """Gaussian actor network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        temperature=1.0,
    ):
        """Return action distributions.

        Args:
            observations: Observations.
            temperature: Scaling factor for the standard deviation.
        """
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class Value(nn.Module):
    """Value/critic network.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    def __call__(self, observations, actions=None):
        """Return values or critic values.

        Args:
            observations: Observations.
            actions: Actions (optional).
        """
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        return v


class ActorVectorField(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None

    def setup(self) -> None:
        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)

    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False):
        """Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            times: Times (optional).
            is_encoded: Whether the observations are already encoded.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        v = self.mlp(inputs)

        return v
    
# TRANSFORMER

class MaskedCausalAttention(nn.Module):
    h_dim: int
    max_T: int
    n_heads: int
    drop_p: float = 0.1
    dtype: Any = jnp.float32
    kernel_init: Callable[..., Any] = lecun_normal()
    bias_init: Callable[..., Any] = zeros
    deterministic: bool = False if drop_p > 0.0 else True
    use_mask: bool = False
    
    def setup(self):
        self.mask = jnp.tril(
            jnp.ones((self.max_T, self.max_T))).reshape(1, 1, self.max_T, self.max_T)

    @nn.compact
    def __call__(self, input):
        B, T, C = input.shape # batch size, seq length, h_dim * n_heads
        N, D = self.n_heads, C // self.n_heads # N = num heads, D = attention dim

        q = nn.Dense(features=self.h_dim, kernel_init=self.kernel_init)(input).reshape(B, T, N, D).transpose(0, 2, 1, 3)
        k = nn.Dense(features=self.h_dim, kernel_init=self.kernel_init)(input).reshape(B, T, N, D).transpose(0, 2, 1, 3)
        v = nn.Dense(features=self.h_dim, kernel_init=self.kernel_init)(input).reshape(B, T, N, D).transpose(0, 2, 1, 3)
        weights = q @ k.transpose(0, 1, 3, 2) / jnp.sqrt(D)
        if self.use_mask:
            weights = jnp.where(self.mask[..., :T, :T], weights, -jnp.inf)
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = jax.nn.softmax(weights, axis=-1)
        
        attention = normalized_weights @ v
        # attention = nn.Dropout(
        #     rate=self.drop_p,
        #     deterministic=self.deterministic)(normalized_weights @ v)
        attention = attention.transpose(0, 2, 1, 3).reshape(B, T, N*D)
        projection = nn.Dense(self.h_dim, kernel_init=self.kernel_init)(attention)
        #out = nn.Dropout(rate=self.drop_p, deterministic=self.deterministic)(projection)
        return projection
        
class Block(nn.Module):
    h_dim: int
    max_T: int
    n_heads: int
    drop_p: float = 0.9
    dtype: Any = jnp.float32
    kernel_init: Callable[..., Any] = lecun_normal()
    bias_init: Callable[..., Any] = zeros
    deterministic: bool = False if drop_p > 0.0 else True
    use_mask: bool = False
    
    @nn.compact
    def __call__(self, input: Float[Array, "bs {2}*context_len h_dim"]):
        input = input + MaskedCausalAttention(h_dim=self.h_dim,
            max_T=self.max_T,
            n_heads=self.n_heads,
            drop_p=self.drop_p,
            use_mask=self.use_mask)(input)
        input = nn.LayerNorm()(input)
        
        mlp1 = nn.Dense(self.h_dim * 4, kernel_init=self.kernel_init)(input)
        mlp1 = nn.gelu(mlp1)
        mlp2 = nn.Dense(self.h_dim, kernel_init=self.kernel_init)(mlp1)
        
        out = mlp2 + input
        out = nn.LayerNorm()(out)
        return out
    
class DynamicsTransformer(nn.Module):
    """
    The goal of context-identifier transformer is to provide a latent conditional variable,
    which is passed into F and B modules
    """
    n_blocks: int
    h_dim: int  # h_dim = z_dim
    context_len: int
    n_heads: int
    drop_p: float
    num_layouts: int
    dtype: Any = jnp.float32
    kernel_init: Callable[..., Any] = lecun_normal()
    use_masked_attention: bool = False
    use_mean_embedding: bool = False
    
    def setup(self):
        self.input_seq_len = 3 * self.context_len # state x action; maybe later add z
        self.project_obs = nn.Dense(features=self.h_dim, kernel_init=self.kernel_init)
        self.project_acts = nn.Dense(features=self.h_dim, kernel_init=self.kernel_init)
        self.project_layout = nn.Dense(features=self.h_dim, kernel_init=self.kernel_init)
        self.project_next_states = nn.Dense(features=self.h_dim, kernel_init=self.kernel_init)
        self.pre_layernorm = nn.LayerNorm()
        self.context_final_emb = nn.Dense(self.h_dim)
        self.block_module = Block(h_dim=self.h_dim,
                max_T=self.input_seq_len,
                n_heads=self.n_heads,
                drop_p=self.drop_p,
                use_mask=self.use_masked_attention)
        self.classification_head = MLP([512, 512, self.num_layouts])#nn.Dense(self.num_layouts)
        
    def __call__(self,
                states: Float[Array, "bs context_len dim"],
                actions: Float[Array, "bs context_len dim"],
                next_states: Float[Array, "bs context_len dim"],
                layout_type=None,
                predict_type: bool = False,
                return_last_layer: bool = False):
        
        assert states.ndim == 3
        assert states.shape[-1] == self.h_dim
        
        B, T, _ = states.shape
        states = self.project_obs(states)
        acts = self.project_acts(actions)
        next_states = self.project_next_states(next_states)
        h = jnp.stack(
            (states, acts, next_states), axis=1
        ).transpose(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)
        
        h = self.pre_layernorm(h)
        for _ in range(self.n_blocks):
            h = self.block_module(h)
    
        h = h.reshape(B, T, 3, self.h_dim).transpose(0, 2, 1, 3)
            
        if self.use_mean_embedding:
            context_embedding = jnp.mean(h[:, 1], axis=1)
        else:
            context_embedding = h[:, 1, -1]
        # context_embedding = self.context_final_emb(h[:, 1]) # context is s_0, a_0, ... and predict based on context and s_t, a_t
        if predict_type:
            return self.classification_head(context_embedding)
        if return_last_layer:
            return h[:, 1]
        return context_embedding
        