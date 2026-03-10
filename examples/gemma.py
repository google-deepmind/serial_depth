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

"""Computes the serial depth of a Gemma model."""

from absl import app
from absl import flags
from gemma import gm
import jax
import numpy as np
from serial_depth.serial_depth import depth

_MODEL = flags.DEFINE_enum(
    'model',
    'Gemma3_1B',
    [
        'Gemma2_2B',
        'Gemma2_9B',
        'Gemma2_27B',
        'Gemma3_1B',
        'Gemma3_4B',
        'Gemma3_12B',
        'Gemma3_27B',
    ],
    'Which Gemma model to load.',
)
_SEQUENCE_LENGTH = flags.DEFINE_integer(
    'sequence_length', 32768, 'The sequence length to use for the calculation.'
)
_VERBOSE = flags.DEFINE_boolean(
    'verbose',
    False,
    'Whether to enable verbose output during depth computation.',
)


def main(argv):
  """Loads a Gemma model and computes its depth."""
  del argv  # Unused.

  model_class = getattr(gm.nn, _MODEL.value)
  model = model_class()
  rng = jax.random.PRNGKey(0)

  # Create a dummy input tensor to get the correct shape and dtype for tracing
  vocab_size = model.config.num_embed
  tokens = np.random.randint(0, vocab_size, size=(1, _SEQUENCE_LENGTH.value))
  abstract_tokens = jax.ShapeDtypeStruct(tokens.shape, tokens.dtype)

  # Get abstract variables and the jaxpr for the model's apply function
  abstract_vars = jax.eval_shape(
      lambda: model.init(rng, tokens=abstract_tokens)
  )
  closed_jaxpr = jax.make_jaxpr(model.apply)(
      abstract_vars, tokens=abstract_tokens
  )
  jaxpr = closed_jaxpr.jaxpr

  # Compute the depth of the model
  result = depth.compute_depth(jaxpr, verbose=_VERBOSE.value)

  print(f'The serial depth of the Gemma model is: {result}')


if __name__ == '__main__':
  app.run(main)
