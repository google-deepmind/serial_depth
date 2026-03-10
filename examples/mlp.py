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

"""Computes the serial depth of a simple MLP."""

from absl import app
from absl import flags
from flax import linen as nn
import jax
import jax.numpy as jnp
from serial_depth.serial_depth import depth

_NUM_HIDDEN = flags.DEFINE_integer(
    'num_hidden', 8, 'Number of hidden units in the MLP.'
)
_NUM_OUTPUTS = flags.DEFINE_integer(
    'num_outputs', 1, 'Number of output units in the MLP.'
)
_USE_BIAS = flags.DEFINE_boolean(
    'use_bias', False, 'Whether to use a bias in the Dense layers.'
)
_VERBOSE = flags.DEFINE_boolean(
    'verbose',
    False,
    'Whether to enable verbose output during depth computation.',
)


class MLP(nn.Module):
  """A simple MLP."""

  num_hidden: int
  num_outputs: int
  use_bias: bool = True

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(features=self.num_hidden, use_bias=self.use_bias)(x)
    x = nn.relu(x)
    x = nn.Dense(features=self.num_hidden, use_bias=self.use_bias)(x)
    x = nn.relu(x)
    x = nn.Dense(features=self.num_outputs, use_bias=self.use_bias)(x)
    return x


def main(argv):
  """Initializes an MLP, traces it to a jaxpr, and computes its serial depth."""
  del argv  # Unused.
  key = jax.random.PRNGKey(0)
  model = MLP(
      num_hidden=_NUM_HIDDEN.value,
      num_outputs=_NUM_OUTPUTS.value,
      use_bias=_USE_BIAS.value,
  )

  abstract_input = jax.ShapeDtypeStruct((1,), jnp.float32)
  abstract_vars = jax.eval_shape(model.init, key, abstract_input)

  closed_jaxpr = jax.make_jaxpr(model.apply)(abstract_vars, abstract_input)
  jaxpr = closed_jaxpr.jaxpr

  result = depth.compute_depth(jaxpr, verbose=_VERBOSE.value)
  print(f'The serial depth of the MLP is: {result}')


if __name__ == '__main__':
  app.run(main)
