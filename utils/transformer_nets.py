from typing import Any, Callable, Optional, Tuple, Type

import flax.linen as nn
import jax.numpy as jnp
from typing import *

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')

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
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
        return x

class IdentityLayer(nn.Module):
    """Identity layer, convenient for giving a name to an array."""

    @nn.compact
    def __call__(self, x):
        return x


class AddPositionEmbs(nn.Module):
    # Need to define function that adds the poisition embeddings to the input.
    context_len: int
    posemb_init: Callable[[PRNGKey, Shape, Dtype], Array]

    @nn.compact
    def __call__(self, inputs):
        """
            inputs.shape is (batch_size, timesteps, emb_dim).
            Output tensor with shape `(batch_size, timesteps, in_dim)`.
        """
        assert inputs.ndim == 3, ('Number of dimensions should be 3, but it is: %d' % inputs.ndim)

        position_ids = jnp.arange(self.context_len)[None] # (1, timesteps)
        pos_embeddings = nn.Embed(
            self.context_len, # Max Positional Embeddings
            inputs.shape[2],
            embedding_init=self.posemb_init,
            dtype=inputs.dtype,
        )(position_ids)
        return inputs + pos_embeddings

class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int
    dtype: Dtype = jnp.float32
    out_dim: Optional[int] = None
    dropout_rate: float = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(stddev=1e-6)

    @nn.compact
    def __call__(self, inputs, *, deterministic):
        """It's just an MLP, so the input shape is (batch, len, emb)."""
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
                features=self.mlp_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init)(inputs)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        output = nn.Dense(
                features=actual_out_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init)(x)
        output = nn.Dropout(
                rate=self.dropout_rate)(output, deterministic=deterministic)
        return output


class Encoder1DBlock(nn.Module):
    """Transformer encoder layer.
    Given a sequence, it passes it through an attention layer, then through a mlp layer.
    In each case it is a residual block with a layer norm.
    """

    mlp_dim: int
    num_heads: int
    causal: bool
    dropout_rate: float
    attention_dropout_rate: float
    dtype: Dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, inputs, *, deterministic, train=True):

        if self.causal:
            causal_mask = nn.make_causal_mask(
                jnp.ones((inputs.shape[0], inputs.shape[1])),
                dtype="bool"
            )
        else:
            causal_mask = None

        # Attention block.
        assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
        x = nn.LayerNorm(dtype=self.dtype)(inputs)
        x = nn.MultiHeadDotProductAttention(
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            broadcast_dropout=False,
            deterministic=deterministic,
            dropout_rate=self.attention_dropout_rate,
            decode=False,
            num_heads=self.num_heads)(x, x, mask=None)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + inputs

        # MLP block. This does NOT change the embedding dimension!
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = MlpBlock(mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate)(y, deterministic=deterministic)

        return x + y

class LayoutClassifier(nn.Module):
    hidden_dims: Sequence[int]
    num_layouts: int 
    
    def setup(self):
        self.classifier = MLP((*self.hidden_dims, self.num_layouts))
        
    def __call__(self, context_embedding, states):
        z_expand = jnp.expand_dims(context_embedding, axis=1) # [batch, 1, emb_dim]
        z_expand = jnp.repeat(z_expand, repeats=states.shape[1], axis=1)
        decoder_input = jnp.concatenate([z_expand, states], axis=-1)
        layout_pred = self.classifier(decoder_input)
        return layout_pred

class NextStatePrediction(nn.Module):
    hidden_dims: Sequence[int]
    out_dim: int
    
    def setup(self):
        self.state_predictor = MLP((*self.hidden_dims, self.out_dim))
        # self.state_no_context_pred = MLP((*self.hidden_dims, self.out_dim))
        
    def __call__(self, states, actions, dynamics_embedding):
        # pred_next_no_context = self.state_predictor(jnp.concatenate([states, actions], -1))
        pred_next_context = self.state_predictor(jnp.concatenate([states, actions, dynamics_embedding], -1))
        return pred_next_context

class DynamicsTransformer(nn.Module):
    num_layers: int
    num_heads: int
    emb_dim: int
    out_dim: int
    mlp_dim: int
    dropout_rate: float
    attention_dropout_rate: float
    action_dim: int
    context_len: int
    causal: bool = False

    
    @nn.compact
    def __call__(self, states, actions, next_states, valid_transition=None, train=False, return_embedding=True):
        B, T, _ = states.shape
        
        # 1. Embed Individual Components
        # state_emb = nn.Dense(self.emb_dim, name='state_embed')(states)
        # action_emb = nn.Embed(self.action_dim, self.emb_dim, name='action_embed')(actions.squeeze(-1))
        # next_state_emb = nn.Dense(self.emb_dim, name='next_state_embed')(next_states)
        
        # Testing
        state_emb = states
        action_emb = actions
        next_states_emb = next_states
        transitions = jnp.concatenate([state_emb, action_emb, next_states_emb], axis=-1)
        transition_emb = nn.Dense(self.emb_dim, name='token_embedding')(transitions)
    
        # # 3. Add Positional Encoding
        # transitions = AddPositionEmbs(
        #     context_len=self.context_len,
        #     posemb_init=nn.initializers.normal(stddev=0.02),
        #     name='pos_embed'
        # )(transitions)

        # 4. Transformer Processing
        for _ in range(self.num_layers):
            transition_emb = Encoder1DBlock(
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                causal=self.causal
            )(transition_emb, deterministic=not train)
          
        context_embedding = nn.Dense(self.out_dim)(transition_emb.mean(1))
        return context_embedding
        # return context_embedding, transitions
        # emb_mean = nn.Dense(self.emb_dim)(context_embedding)
        # emb_log_std = nn.Dense(self.emb_dim)(context_embedding)
        
        # return emb_mean, emb_log_std
