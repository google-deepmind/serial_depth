# JAX Serial Depth Calculator

Automated computation of serial depth for JAX neural networks by analyzing their
jaxpr (JAX expression) computational graphs.

## Quick Start

```bash
uv run examples/gemma.py --model Gemma3_4B --sequence_length 32768
```

## Rules for computing serial depth

When computing the serial depth of a JAX program, we follow the following rules:

*   The depth of initial input variables and constants is 0.
*   **Depth 0 primitives:** Data movement and reshaping operations like
    `transpose` or `reshape` add 0 to the depth.
*   **Depth 1 primitives:** Element-wise operations like `add`, `mul`, or `sin`
    add 1 to the depth.
*   **Logarithmic depth primitives:** Parallelizable reductions over `N`
    elements, like `reduce_sum`, `dot_general`, or `cumsum`, add $\log N$ to the
    depth.
*   **Linear depth primitives:** Inherently sequential operations like `scan`
    add `num_iterations * body_depth` to the depth.
*   **Control flow primitives:** The depth of a `cond` is the maximum depth of
    its branches.
*   **Nested computations:** Nested computations like `jit` or `pjit` are
    handled by recursively computing their serial depth.
*   **Final depth:** The depth of the entire program is the maximum depth among
    its output variables.

## Example: MLP

To understand the computation, let's manually compute the depth of a simple two
layer MLP (see `examples/mlp.py`). We have a single input dimension and a single
output dimension, and we have two hidden layers with dimension 8.

We can compute the serial depth layer by layer:

* The first hidden layer has 8 neurons, and each neuron computes a product with
  the input. These can happen in parallel which gives depth 1. This is followed
  by a ReLU activation, so the total depth contribution is `1 + 1 = 2`.
* The second hidden layer has 8 neurons, and computes a product with each input
  (depth 1), a sum over 8 inputs (depth 3), and by a ReLU activation (depth 1).
  The total depth contribution is `1 + ceil(log(8)) + 1 = 5`.
* The output layer has 1 output, and computes a product for each input and then
  a sum over 8 inputs. The depth contribution is `1 + ceil(log(8)) = 4`.

The total depth of the MLP is the sum of these contributions: `2 + 5 + 4 = 11`.

We can run this computation using the example in `examples/mlp.py` and verify
that the code computes the same depth:

```bash
> uv run examples/mlp.py

The serial depth of the MLP is: 11
```
