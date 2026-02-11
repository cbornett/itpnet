import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
import flax

from typing import Dict, Callable, Sequence

from functools import partial

# taken from MLFF package
class Residual(nn.Module):
    num_blocks: int = 2
    activation_fn: Callable = jax.nn.silu
    use_bias: bool = True

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = inputs
        feat = x.shape[-1]
        for i in range(self.num_blocks):
            x = self.activation_fn(x)
            x = nn.Dense(feat, use_bias=self.use_bias, name=f"layers_{i}")(x)
        return e3x.nn.add(x, inputs)

### Equivariant Layer Norm and Auxiliary Functions

def safe_mask(
        mask: jnp.ndarray,
        fn: Callable[..., jnp.ndarray],
        operand: jnp.ndarray,
        placeholder: float = 0.0,
):
    """Gradient safe mask.

  Args:
    mask:
    fn:
    operand:
    placeholder:

  Returns:

  """
    masked = jnp.where(mask, operand, 0.0)
    return jnp.where(mask, fn(masked), placeholder)


def safe_norm(x, axis=0, keepdims=False) -> jnp.ndarray:
    """Take gradient safe norm.

  Args:
    x: Tensor.
    axis: Axis along which norm is taken.
    keepdims: If dimension should be kept.

  Returns: Tensor.

  """
    u = (x ** 2).sum(axis=axis, keepdims=keepdims)
    return safe_mask(mask=u > 0, fn=jnp.sqrt, operand=u, placeholder=0.0)


def promote_to_e3x(x: jnp.ndarray) -> jnp.ndarray:
    """
  Promote an invariant node representation to a tensor that matches the shape
  convention of e3x, i.e. adding an axis for parity and irreps.

    Args:
      x: Tensor of shape (n, F)

    Returns: Tensor of shape (n, 1, 1, F)
  """
    assert x.ndim == 2
    return x[:, None, None, :]


def make_degree_repeat_fn(degrees: Sequence[int], axis: int = -1):
    repeats = np.array([2 * y + 1 for y in degrees])
    repeat_fn = partial(np.repeat, repeats=repeats, axis=axis)
    return repeat_fn


class EquivariantLayerNorm(nn.Module):
    use_scale: bool = True
    use_bias: bool = True

    bias_init: Callable = nn.initializers.zeros
    scale_init: Callable = nn.initializers.ones

    epsilon: float = 1e-6
    param_dtype = jnp.float32

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        """
            x.shape: (N, 1 or 2, (max_degree + 1)^2, num_features)
        """
        assert x.ndim == 4

        max_degree = int(np.rint(np.sqrt(x.shape[-2]))) - 1
        num_features = x.shape[-1]
        num_atoms = x.shape[-4]

        has_pseudotensors = x.shape[-3] == 2
        has_ylms = x.shape[-2] > 1

        if has_pseudotensors or has_ylms:
            plm_axes = x.shape[-3:-1]

            y = x.reshape(num_atoms, -1, num_features)  # (N, plm, num_features)
            y00, ylm = jnp.split(
                y,
                axis=1,
                indices_or_sections=np.array([1])
            )  # (N, 1, num_features), (N, plm - 1, num_features)

            # Construct the segment sum indices for summing over degree and parity channels.
            repeat_fn_even = make_degree_repeat_fn(degrees=list(range(1, max_degree + 1)))
            sum_idx_even = repeat_fn_even(np.arange(max_degree))

            if has_pseudotensors:
                repeat_fn_odd = make_degree_repeat_fn(degrees=list(range(max_degree + 1)))
                sum_idx_odd = repeat_fn_odd(np.arange(max_degree, 2 * max_degree + 1))
            else:
                sum_idx_odd = np.array([], dtype=sum_idx_even.dtype)

            sum_idx = np.concatenate([sum_idx_even, sum_idx_odd], axis=0)

            ylm_sum_squared = jax.vmap(
                partial(
                    jax.ops.segment_sum,
                    segment_ids=sum_idx,
                    num_segments=2 * max_degree + 1 if has_pseudotensors else max_degree
                )
            )(
                lax.square(ylm),
            )  # (N, parity * max_degree + 1 or max_degree, num_features)

            ylm_inv = safe_mask(
                ylm_sum_squared > self.epsilon,
                lax.sqrt,
                ylm_sum_squared,
                # self.epsilon # minimum value?
            )

            _, var_lm = nn.normalization._compute_stats(
                ylm_inv,
                axes=-1,
                dtype=None
            )  # (N, parity * max_degree + 1 or max_degree)

            mul_lm = lax.rsqrt(var_lm + jnp.asarray(self.epsilon, dtype=var_lm.dtype))
            # (N, parity * max_degree + 1 or max_degree)

            if self.use_scale:
                scales_lm = self.param(
                    'scales_lm',
                    self.scale_init,
                    (var_lm.shape[-1], ),
                    self.param_dtype
                )  # (parity * max_degree + 1 or max_degree)

                mul_lm = mul_lm * scales_lm  # (N, parity * max_degree + 1 or max_degree)

            mul_lm = jnp.expand_dims(mul_lm, axis=-1)  # (N, parity * max_degree + 1 or max_degree, 1)

            ylm = ylm * mul_lm[:, sum_idx, :]  # (N, plm - 1, num_features)

            y00 = nn.LayerNorm(
                use_scale=self.use_scale,
                use_bias=self.use_bias,
                scale_init=self.scale_init,
                bias_init=self.bias_init
            )(y00)  # (N, 1, num_features)

            y = jnp.concatenate([y00, ylm], axis=1)  # (N, plm, num_features)
            return y.reshape(num_atoms, *plm_axes, num_features)  # (N, 1 or 2, (max_degree + 1)^2, num_features)
        else:
            return nn.LayerNorm(
                use_scale=self.use_scale,
                use_bias=self.use_bias,
                scale_init=self.scale_init,
                bias_init=self.bias_init
            )(x)  # (N, 1, num_features)
        
# immutable dataclass
@flax.struct.dataclass
class InteractionConfig:

    # parameters
    num_features: int # number of features for Dense MLPs
    num_basis_functions: int = 8 # number of basis functions for displacement vector expansion
    cutoff: float = 5.0 # cutoff for message pass
    max_atomic_number: int = 118
    include_pseudotensors: bool = True
    mp_res_block: bool = False
    num_mp_res_block: int = 2 
    init_mp_max_degree: int = 3 # l_max
    itp_max_degree: int = 2 # l_max
    num_itp_features: int = 32
    num_itp_iterations: int = 3 
    itp_res_block: bool = False
    num_itp_res_block: int = 2
    readout: str = 'last'

    # modify parts of the model
    do_eqv_norm: bool = True
    do_feature_basis_align: bool = True
    radial_fn: str = "reciprocal_bernstein"
    shared_embed: bool = True
