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

"""Primitive sets for depth computation."""

DEPTH_0_PRIMITIVES = {
    'broadcast_in_dim',
    'gather',
    'stop_gradient',
    'reshape',
    'transpose',
    'slice',
    'squeeze',
    'split',
    'concatenate',
    'named_tensor',
    'copy',
    'random_wrap',
    'random_unwrap',
    'pad',
    'name',
    'optimization_barrier',
    # assigning values to specific indices, doesn't do computation
    'scatter',
}

DEPTH_1_PRIMITIVES = {
    'add',
    'div',
    'sub',
    'mul',
    'rem',
    'relu',
    'max',
    'ne',
    'lt',
    'ge',
    'gt',
    'iota',
    'square',
    'sqrt',
    'rsqrt',
    'exp',
    'integer_pow',
    'tanh',
    'sin',
    'cos',
    'log',
    'pow',
    'convert_element_type',
    'and',
    'not',
    'or',
    'eq',
    'reduce_precision',
    'neg',
    'abs',
    'min',
    # PRNG algorithms are possibly not low depth but we can treat their output
    # as a random bit input to the circuit we're computing depth of
    'random_seed',
    'random_fold_in',
    'random_split',
    'random_clone',
    'logistic',
    'is_finite',
    'sign',
}

REDUCE_PRIMITIVES = {
    'reduce_sum',
    'reduce_max',
    'reduce_min',
    'reduce_and',
    'reduce_or',
    'argmax',
    'argmin',
}
