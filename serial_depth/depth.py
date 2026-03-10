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

"""Provides functionality to compute the serial depth of JAX jaxprs."""

import dataclasses
import math
from typing import Callable

import jax

from . import primitives

core = jax._src.core  # pylint: disable=protected-access
HandlerType = Callable[[core.JaxprEqn, int, int], None]


@dataclasses.dataclass
class SerialDepthConfig:
  """Configuration for serial depth computation.

  Attributes:
    handle_unsupported_as_depth_0: If True, unsupported primitives are treated
      as depth-0 operations. If False, they raise an error.
    verbose: If True, enables verbose output during computation.
    max_recursion_depth: Maximum allowed recursion depth for nested jaxprs.
  """

  handle_unsupported_as_depth_0: bool = True
  verbose: bool = False
  max_recursion_depth: int = 1000


@dataclasses.dataclass
class SerialDepthResult:
  """Result of serial depth computation.

  Attributes:
    depth: The resulting serial depth of the computation.
    unsupported_primitives: Set of primitive names that were not recognized.
    total_operations: Total number of operations in the computation graph.
  """

  depth: int
  unsupported_primitives: set[str]
  total_operations: int


class SerialDepthCalculator:
  """Computes the serial depth of a JAX model by traversing its jaxpr.

  Attributes:
    config: Configuration object for the computation.
    depths: Dictionary of variable depths used during recursion.
    unsupported_primitives: Set used to track encountered unsupported
      primitives.
    total_operations: Counter for total operations processed.
    primitive_handlers: Dispatch table for primitive-specific handlers.
  """

  def __init__(self, config: SerialDepthConfig = SerialDepthConfig()):
    self.config = config
    self.depths: dict[core.Var, int] = {}
    self.unsupported_primitives: set[str] = set()
    self.total_operations = 0
    self.primitive_handlers = self._register_handlers()

  def _get_depth(self, var: core.Var) -> int:
    """Gets depth for a variable, returning 0 for Literal objects."""
    if isinstance(var, core.Literal):
      return 0
    return self.depths.get(var, 0)

  def _register_handlers(self) -> dict[str, HandlerType]:
    """Creates a dispatch dictionary for primitive handlers."""
    handlers: dict[str, HandlerType] = {
        'dot_general': self._handle_dot_general,
        'cumsum': self._handle_cumsum,
        'select_n': self._handle_select_n,
        'top_k': self._handle_top_k,
        'cond': self._handle_cond,
        'scan': self._handle_scan,
        'pjit': self._get_nested_jaxpr_handler('jaxpr'),
        'jit': self._get_nested_jaxpr_handler('jaxpr'),
        'remat2': self._get_nested_jaxpr_handler('jaxpr'),
        'custom_vjp_call': self._get_nested_jaxpr_handler('call_jaxpr'),
        'custom_jvp_call': self._get_nested_jaxpr_handler('call_jaxpr'),
        **{p: self._handle_depth_0 for p in primitives.DEPTH_0_PRIMITIVES},
        **{p: self._handle_depth_1 for p in primitives.DEPTH_1_PRIMITIVES},
        **{p: self._handle_reduce for p in primitives.REDUCE_PRIMITIVES},
    }
    return handlers

  @staticmethod
  def _get_eqn_label(eqn: core.JaxprEqn) -> str:
    if eqn.source_info and eqn.source_info.name_stack:
      names = [x.name for x in eqn.source_info.name_stack.stack]
      return '/'.join(names) + ':' + eqn.primitive.name
    return eqn.primitive.name

  @staticmethod
  def _get_dot_general_input_size(eqn: core.JaxprEqn) -> int:
    if isinstance(eqn.invars[0], core.Literal):
      return 0
    lhs_contracting_dims = eqn.params['dimension_numbers'][0][0]
    lhs_aval = eqn.invars[0].aval
    assert isinstance(lhs_aval, core.ShapedArray)
    return math.prod([lhs_aval.shape[i] for i in lhs_contracting_dims])

  def _process_jaxpr(
      self,
      name: str,
      jaxpr: core.Jaxpr,
      current_recursion_depth: int = 0,
  ):
    """Processes a jaxpr by recursively calling handlers for each primitive.

    Args:
      name: Name of the jaxpr for debugging output.
      jaxpr: The JAX expression to process.
      current_recursion_depth: Current recursion depth.

    Raises:
      RuntimeError: If recursion depth exceeds max_recursion_depth.
    """
    if current_recursion_depth > self.config.max_recursion_depth:
      raise RuntimeError(
          f'Recursion depth {current_recursion_depth} exceeds maximum'
          f' {self.config.max_recursion_depth}'
      )

    if self.config.verbose:
      print(f"{' ' * (current_recursion_depth * 2)}{name}")

    for eqn in jaxpr.eqns:
      self.total_operations += 1
      max_input_depth = 0
      if eqn.invars:
        input_depths = [self._get_depth(v) for v in eqn.invars]
        if input_depths:
          max_input_depth = max(input_depths)

      handler = self.primitive_handlers.get(
          eqn.primitive.name, self._handle_unsupported
      )
      handler(eqn, max_input_depth, current_recursion_depth)

  def _get_nested_jaxpr_handler(self, param_key: str) -> HandlerType:
    """Returns a handler for primitives with nested jaxprs at the given param key."""

    def handler(
        eqn: core.JaxprEqn,
        unused_max_in_serial_depth: int,
        current_recursion_depth: int,
    ):
      nested_jaxpr = eqn.params[param_key]

      for nested_var, outer_var in zip(nested_jaxpr.invars, eqn.invars):
        self.depths[nested_var] = self._get_depth(outer_var)

      self._process_jaxpr(
          eqn.primitive.name, nested_jaxpr, current_recursion_depth + 1
      )

      for outer_var, nested_var in zip(eqn.outvars, nested_jaxpr.outvars):
        self.depths[outer_var] = self._get_depth(nested_var)

    return handler

  def _update_outvars(
      self,
      eqn: core.JaxprEqn,
      output_depth: int,
      current_recursion_depth: int,
  ):
    """Updates output variables with computed depth.

    Args:
      eqn: The JAX equation being processed.
      output_depth: The computed depth for the output variables.
      current_recursion_depth: Current recursion depth for verbose output.
    """
    if self.config.verbose:
      print(
          f"{' ' * (current_recursion_depth * 2 + 2)}{self._get_eqn_label(eqn)}:"
          f' {output_depth}'
      )
    for var in eqn.outvars:
      self.depths[var] = output_depth

  def _handle_depth_0(
      self,
      eqn: core.JaxprEqn,
      max_in_serial_depth: int,
      current_recursion_depth: int,
  ):
    self._update_outvars(eqn, max_in_serial_depth, current_recursion_depth)

  def _handle_depth_1(
      self,
      eqn: core.JaxprEqn,
      max_in_serial_depth: int,
      current_recursion_depth: int,
  ):
    self._update_outvars(eqn, max_in_serial_depth + 1, current_recursion_depth)

  def _handle_unsupported(
      self,
      eqn: core.JaxprEqn,
      max_in_serial_depth: int,
      current_recursion_depth: int,
  ):
    self.unsupported_primitives.add(eqn.primitive.name)
    self._update_outvars(eqn, max_in_serial_depth, current_recursion_depth)

  def _handle_dot_general(
      self,
      eqn: core.JaxprEqn,
      max_in_serial_depth: int,
      current_recursion_depth: int,
  ):
    """Handles dot_general primitive."""
    n = self._get_dot_general_input_size(eqn)
    op_depth = math.ceil(math.log2(n)) + 1 if n > 0 else 0
    self._update_outvars(
        eqn, max_in_serial_depth + op_depth, current_recursion_depth
    )

  def _handle_cumsum(
      self,
      eqn: core.JaxprEqn,
      max_in_serial_depth: int,
      current_recursion_depth: int,
  ):
    """Handles cumsum primitive."""
    if isinstance(eqn.invars[0], core.Literal):
      op_depth = 0
    else:
      aval = eqn.invars[0].aval
      assert isinstance(aval, core.ShapedArray)
      axis_length = aval.shape[eqn.params['axis']]
      op_depth = math.ceil(math.log2(axis_length))
    self._update_outvars(
        eqn, max_in_serial_depth + op_depth, current_recursion_depth
    )

  def _handle_reduce(
      self,
      eqn: core.JaxprEqn,
      max_in_serial_depth: int,
      current_recursion_depth: int,
  ):
    """Handles reduce primitive."""
    if isinstance(eqn.invars[0], core.Literal):
      op_depth = 0
    else:
      aval = eqn.invars[0].aval
      assert isinstance(aval, core.ShapedArray)
      n = math.prod([aval.shape[i] for i in eqn.params['axes']])
      op_depth = math.ceil(math.log2(n))
    self._update_outvars(
        eqn, max_in_serial_depth + op_depth, current_recursion_depth
    )

  def _handle_select_n(
      self,
      eqn: core.JaxprEqn,
      max_in_serial_depth: int,
      current_recursion_depth: int,
  ):
    """Handles select_n primitive."""
    num_choices = len(eqn.invars) - 1  # Subtract 1 for the selector input
    op_depth = math.ceil(math.log2(num_choices))
    self._update_outvars(
        eqn, max_in_serial_depth + op_depth, current_recursion_depth
    )

  def _handle_top_k(
      self,
      eqn: core.JaxprEqn,
      max_in_serial_depth: int,
      current_recursion_depth: int,
  ):
    """Handles top_k primitive.

    top_k selects the k largest elements, which can be computed in O(log n)
    parallel depth using sorting or selection algorithms.

    Args:
      eqn: The JAX equation being processed.
      max_in_serial_depth: The maximum serial depth of the input variables.
      current_recursion_depth: Current recursion depth.
    """
    if isinstance(eqn.invars[0], core.Literal):
      op_depth = 0
    else:
      aval = eqn.invars[0].aval
      assert isinstance(aval, core.ShapedArray)
      # top_k operates on the last dimension
      n = aval.shape[-1]
      op_depth = math.ceil(math.log2(n))
    self._update_outvars(
        eqn, max_in_serial_depth + op_depth, current_recursion_depth
    )

  def _handle_scan(
      self,
      eqn: core.JaxprEqn,
      max_in_serial_depth: int,
      current_recursion_depth: int,
  ):
    """Handles scan primitive.

    scan applies a function over the leading axis of an array, while carrying
    along state. This requires O(n) depth, where n is the size of the leading
    axis.

    Args:
      eqn: The JAX equation being processed.
      max_in_serial_depth: The maximum serial depth of the input variables.
      current_recursion_depth: Current recursion depth.
    """
    jaxpr = eqn.params['jaxpr'].jaxpr
    length = eqn.params['length']

    for nested_var, outer_var in zip(jaxpr.invars, eqn.invars):
      self.depths[nested_var] = self._get_depth(outer_var)

    self._process_jaxpr(eqn.primitive.name, jaxpr, current_recursion_depth + 1)

    for nested_var, outer_var in zip(jaxpr.outvars, eqn.outvars):
      depth_increment = max(
          0, self._get_depth(nested_var) - max_in_serial_depth
      )
      self.depths[outer_var] = max_in_serial_depth + length * depth_increment

  def _handle_cond(
      self,
      eqn: core.JaxprEqn,
      unused_max_in_serial_depth: int,
      current_recursion_depth: int,
  ):
    """Handles cond primitive."""
    branches = eqn.params['branches']
    for branch in branches:
      for nested_var, outer_var in zip(branch.jaxpr.invars, eqn.invars):
        self.depths[nested_var] = self._get_depth(outer_var)

      self._process_jaxpr(
          eqn.primitive.name, branch.jaxpr, current_recursion_depth + 1
      )

      for nested_var, outer_var in zip(branch.jaxpr.outvars, eqn.outvars):
        current_var_depth = self._get_depth(outer_var)
        branch_depth = self._get_depth(nested_var)
        self.depths[outer_var] = max(current_var_depth, branch_depth)

  def compute(self, jaxpr: core.Jaxpr) -> SerialDepthResult:
    """Computes the depth of a jaxpr."""
    self.depths = {
        var: 0 for var in jaxpr.invars if not isinstance(var, core.Literal)
    }
    self.unsupported_primitives = set()
    self.total_operations = 0

    self._process_jaxpr('root', jaxpr, 0)

    depth = (
        max(self._get_depth(v) for v in jaxpr.outvars) if jaxpr.outvars else 0
    )

    if self.config.verbose:
      print(f'Processed {self.total_operations} operations.')

    return SerialDepthResult(
        depth=depth,
        unsupported_primitives=self.unsupported_primitives.copy(),
        total_operations=self.total_operations,
    )


def compute_depth(
    jaxpr: core.Jaxpr,
    verbose: bool = False,
) -> int:
  """Computes the serial depth of a JAX model from its jaxpr.

  Args:
    jaxpr: The JAX expression to analyze.
    verbose: If True, enables verbose output.

  Returns:
    The computed serial depth as an integer.
  """
  config = SerialDepthConfig(
      handle_unsupported_as_depth_0=True, verbose=verbose
  )
  calculator = SerialDepthCalculator(config=config)
  result = calculator.compute(jaxpr)

  if result.unsupported_primitives:
    print(
        'Warning: There were unsupported primitives, that were assumed to'
        f' have depth 0: {result.unsupported_primitives}'
    )

  return result.depth
