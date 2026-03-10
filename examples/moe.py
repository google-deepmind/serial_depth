# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Computes the serial depth of a MoE (Mixture of Experts) transformer.

This example demonstrates depth computation on a complete MoE transformer
architecture with the following components:
- Token embeddings with learned vocabulary
- Multi-head self-attention with Rotary Positional Embeddings (RoPE)
- Mixture of Experts (MoE) feedforward layers with top-k routing
- RMSNorm (Root Mean Square Normalization) for layer normalization
- Pre-normalization architecture (norm before attention/FFN)
- Residual connections

**MoE Layer Details:**
- Configurable number of experts (default: 8)
- Top-k expert selection per token (default: 2 experts per token)
- Router network for computing expert selection logits
- SwiGLU activation in expert networks (gated linear units)
- Weighted combination of selected expert outputs

This architecture is representative of modern MoE transformers and works
entirely on CPU using standard JAX/Flax operations, making it easy to run
the depth calculation without any additional dependencies.
"""

from absl import app
from absl import flags
from flax import linen as nn
import jax
import jax.numpy as jnp

from serial_depth.serial_depth import depth

_NUM_EXPERTS = flags.DEFINE_integer(
    'num_experts', 64, 'Number of experts in each MoE layer.'
)
_NUM_EXPERTS_PER_TOKEN = flags.DEFINE_integer(
    'experts_per_token', 8, 'Number of experts to route each token to.'
)
_HIDDEN_DIM = flags.DEFINE_integer(
    'hidden_dim', 2048, 'Hidden dimension / model dimension.'
)
_NUM_HEADS = flags.DEFINE_integer('num_heads', 16, 'Number of attention heads.')
_NUM_LAYERS = flags.DEFINE_integer(
    'num_layers', 28, 'Number of transformer layers.'
)
_VOCAB_SIZE = flags.DEFINE_integer('vocab_size', 151936, 'Vocabulary size.')
_SEQUENCE_LENGTH = flags.DEFINE_integer(
    'sequence_length', 512, 'Sequence length.'
)
_VERBOSE = flags.DEFINE_boolean(
    'verbose',
    False,
    'Whether to enable verbose output during depth computation.',
)
_USE_MOE = flags.DEFINE_boolean(
    'use_moe',
    True,
    'Whether to use MoE layers. If False, uses standard dense FFN instead.',
)


class RMSNorm(nn.Module):
  """Root Mean Square Layer Normalization."""

  epsilon: float = 1e-6

  @nn.compact
  def __call__(self, x):
    scale = self.param('scale', nn.initializers.ones, (x.shape[-1],))
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    x = x * jax.lax.rsqrt(variance + self.epsilon)
    return scale * x


def apply_rotary_emb(x, positions):
  """Apply rotary positional embeddings (RoPE)."""
  dim = x.shape[-1]
  freqs = 1.0 / (10000 ** (jnp.arange(0, dim, 2).astype(jnp.float32) / dim))
  pos_freqs = jnp.outer(positions, freqs)
  cos = jnp.cos(pos_freqs)
  sin = jnp.sin(pos_freqs)

  x_even = x[..., 0::2]
  x_odd = x[..., 1::2]
  x_rotated_even = x_even * cos - x_odd * sin
  x_rotated_odd = x_even * sin + x_odd * cos

  x_rotated = jnp.stack([x_rotated_even, x_rotated_odd], axis=-1)
  return x_rotated.reshape(x.shape)


class MultiHeadAttention(nn.Module):
  """Multi-head self-attention with rotary positional embeddings."""

  hidden_dim: int
  num_heads: int

  @nn.compact
  def __call__(self, x):
    batch_size, seq_len, _ = x.shape
    head_dim = self.hidden_dim // self.num_heads

    q = nn.Dense(self.hidden_dim, name='q_proj')(x)
    k = nn.Dense(self.hidden_dim, name='k_proj')(x)
    v = nn.Dense(self.hidden_dim, name='v_proj')(x)

    q = q.reshape(batch_size, seq_len, self.num_heads, head_dim)
    k = k.reshape(batch_size, seq_len, self.num_heads, head_dim)
    v = v.reshape(batch_size, seq_len, self.num_heads, head_dim)

    positions = jnp.arange(seq_len)
    q = jax.vmap(
        lambda x: apply_rotary_emb(x, positions), in_axes=2, out_axes=2
    )(q)
    k = jax.vmap(
        lambda x: apply_rotary_emb(x, positions), in_axes=2, out_axes=2
    )(k)

    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))

    scale = jnp.sqrt(head_dim).astype(x.dtype)
    attn_weights = jnp.einsum('bhqd,bhkd->bhqk', q, k) / scale
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)
    attn_output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)

    attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
    attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_dim)

    output = nn.Dense(self.hidden_dim, name='o_proj')(attn_output)
    return output


class Expert(nn.Module):
  """Expert network with SwiGLU activation."""

  hidden_dim: int

  @nn.compact
  def __call__(self, x):
    gate = nn.Dense(self.hidden_dim * 4, name='gate')(x)
    up = nn.Dense(self.hidden_dim * 4, name='up')(x)
    hidden = jax.nn.silu(gate) * up
    output = nn.Dense(self.hidden_dim, name='down')(hidden)
    return output


class DenseFFN(nn.Module):
  """Standard dense feedforward network with SwiGLU activation."""

  hidden_dim: int

  @nn.compact
  def __call__(self, x):
    gate = nn.Dense(self.hidden_dim * 4, name='gate')(x)
    up = nn.Dense(self.hidden_dim * 4, name='up')(x)
    hidden = jax.nn.silu(gate) * up
    output = nn.Dense(self.hidden_dim, name='down')(hidden)
    return output


class MoELayer(nn.Module):
  """Mixture of Experts layer with top-k routing."""

  num_experts: int
  num_experts_per_token: int
  hidden_dim: int

  @nn.compact
  def __call__(self, x):
    batch_size, seq_len, dim = x.shape

    # Router: compute logits for expert selection
    router_logits = nn.Dense(self.num_experts, name='router')(x)

    # Select top-k experts per token
    top_k_logits, top_k_indices = jax.lax.top_k(
        router_logits, self.num_experts_per_token
    )
    routing_weights = jax.nn.softmax(top_k_logits, axis=-1)

    # Simplified approach: apply all experts and use one-hot selection
    # This is not efficient but fine for depth calculation
    expert_outputs = []
    for i in range(self.num_experts):
      expert = Expert(self.hidden_dim, name=f'expert_{i}')
      expert_out = expert(x)  # (batch, seq, dim)
      expert_outputs.append(expert_out)

    # Stack all expert outputs: (num_experts, batch, seq, dim)
    all_expert_outputs = jnp.stack(expert_outputs, axis=0)

    # For each token, select and combine outputs from chosen experts
    indices_flat = top_k_indices.reshape(-1, self.num_experts_per_token)
    weights_flat = routing_weights.reshape(-1, self.num_experts_per_token)

    def combine_experts(expert_indices, expert_weights):
      one_hot = jax.nn.one_hot(expert_indices, self.num_experts)
      weighted_mask = one_hot * expert_weights[:, None]
      final_weights = jnp.sum(weighted_mask, axis=0)
      return final_weights

    final_weights = jax.vmap(combine_experts)(indices_flat, weights_flat)

    expert_outs_reshaped = all_expert_outputs.reshape(
        self.num_experts, batch_size * seq_len, dim
    )

    output = jnp.einsum('te,etd->td', final_weights, expert_outs_reshaped)

    return output.reshape(batch_size, seq_len, dim)


class TransformerBlock(nn.Module):
  """A single transformer block with attention and feedforward layer."""

  hidden_dim: int
  num_heads: int
  num_experts: int
  num_experts_per_token: int
  use_moe: bool = True

  @nn.compact
  def __call__(self, x):
    attn_input = RMSNorm(name='attn_norm')(x)
    attn_output = MultiHeadAttention(
        hidden_dim=self.hidden_dim,
        num_heads=self.num_heads,
        name='attention',
    )(attn_input)
    x = x + attn_output

    ffn_input = RMSNorm(name='ffn_norm')(x)
    if self.use_moe:
      ffn_output = MoELayer(
          num_experts=self.num_experts,
          num_experts_per_token=self.num_experts_per_token,
          hidden_dim=self.hidden_dim,
          name='moe',
      )(ffn_input)
    else:
      ffn_output = DenseFFN(
          hidden_dim=self.hidden_dim,
          name='ffn',
      )(ffn_input)
    x = x + ffn_output

    return x


class MoETransformer(nn.Module):
  """Full MoE Transformer."""

  vocab_size: int
  hidden_dim: int
  num_heads: int
  num_layers: int
  num_experts: int
  num_experts_per_token: int
  use_moe: bool = True

  @nn.compact
  def __call__(self, tokens):
    x = nn.Embed(
        num_embeddings=self.vocab_size,
        features=self.hidden_dim,
        name='token_embeddings',
    )(tokens)

    for i in range(self.num_layers):
      x = TransformerBlock(
          hidden_dim=self.hidden_dim,
          num_heads=self.num_heads,
          num_experts=self.num_experts,
          num_experts_per_token=self.num_experts_per_token,
          use_moe=self.use_moe,
          name=f'layer_{i}',
      )(x)

    # Final norm
    x = RMSNorm(name='final_norm')(x)

    # Output projection to vocabulary
    logits = nn.Dense(self.vocab_size, name='output')(x)

    return logits


def main(argv):
  """Creates a MoE transformer and computes its depth."""
  del argv  # Unused.

  model = MoETransformer(
      vocab_size=_VOCAB_SIZE.value,
      hidden_dim=_HIDDEN_DIM.value,
      num_heads=_NUM_HEADS.value,
      num_layers=_NUM_LAYERS.value,
      num_experts=_NUM_EXPERTS.value,
      num_experts_per_token=_NUM_EXPERTS_PER_TOKEN.value,
      use_moe=_USE_MOE.value,
  )

  batch_size = 1
  tokens = jnp.ones((batch_size, _SEQUENCE_LENGTH.value), dtype=jnp.int32)

  # Use abstract evaluation to skip actual weight initialization
  abstract_tokens = jax.ShapeDtypeStruct(tokens.shape, tokens.dtype)
  abstract_vars = jax.eval_shape(
      lambda: model.init(jax.random.PRNGKey(0), abstract_tokens)
  )
  closed_jaxpr = jax.make_jaxpr(model.apply)(abstract_vars, abstract_tokens)

  result = depth.compute_depth(closed_jaxpr.jaxpr, verbose=_VERBOSE.value)

  # Calculate total parameters
  def count_params(pytree):
    return sum(x.size for x in jax.tree.leaves(pytree))

  total_params = count_params(abstract_vars)
  params_millions = total_params / 1e6
  params_billions = total_params / 1e9

  print('MoE Transformer Depth Calculation:')
  print(f'  Vocabulary size: {_VOCAB_SIZE.value}')
  print(f'  Hidden dim: {_HIDDEN_DIM.value}')
  print(f'  Num heads: {_NUM_HEADS.value}')
  print(f'  Num layers: {_NUM_LAYERS.value}')
  print(f'  Experts per layer: {_NUM_EXPERTS.value}')
  print(f'  Experts per token: {_NUM_EXPERTS_PER_TOKEN.value}')
  print(f'  Sequence length: {_SEQUENCE_LENGTH.value}')
  if params_billions >= 1.0:
    print(f'  Total parameters: {params_billions:.2f}B ({total_params:,})')
  else:
    print(f'  Total parameters: {params_millions:.2f}M ({total_params:,})')

  # Calculate active parameters for MoE
  if _USE_MOE.value:
    activation_ratio = _NUM_EXPERTS_PER_TOKEN.value / _NUM_EXPERTS.value
    active_params = total_params * activation_ratio
    active_millions = active_params / 1e6
    active_billions = active_params / 1e9
    if active_billions >= 1.0:
      print(
          f'  Active parameters: {active_billions:.2f}B ({active_params:,.0f},'
          f' {activation_ratio*100:.1f}% activation)'
      )
    else:
      print(
          f'  Active parameters: {active_millions:.2f}M ({active_params:,.0f},'
          f' {activation_ratio*100:.1f}% activation)'
      )

  print(f'\nSerial depth: {result}')


if __name__ == '__main__':
  app.run(main)
