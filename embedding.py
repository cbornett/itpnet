import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Dict
from model_utils import Residual

class DelocalizedEmbedSparse(nn.Module):
    num_features: int
    activation_fn: str = 'silu'
    zmax: int = 118
    shared_embed: bool = True

    @nn.compact
    def __call__(self,
                 atomic_numbers: jnp.ndarray,
                 psi: jnp.ndarray,
                 batch_segments: jnp.ndarray,
                 graph_mask: jnp.ndarray,
                 e_Z: jnp.ndarray,
                 *args,
                 **kwargs) -> jnp.ndarray:
        """
        Create atomic embeddings based on the total charge or the number of unpaired spins in the system, following the
        embedding procedure introduced in SpookyNet. Returns per atom embeddings of dimension F.

        Args:
            z (Array): Atomic types, shape: (N)
            psi (Array): Total charge or number of unpaired spins, shape: (num_graphs)
            batch_segment (Array): (N)
            graph_mask (Array): Mask for atom-wise operations, shape: (num_graphs)
            *args ():
            **kwargs ():

        Returns: Per atom embedding, shape: (n,F)

        """
        assert psi.ndim == 1

        # reuse the nuclear charge embedding
        if self.shared_embed: 
            q = nn.Dense(self.num_features)(e_Z) # shape: (N,1,1,F)
        # or create an invididual embedding
        else:
            q = e3x.nn.Embed(
                num_embeddings=self.zmax + 1,
                features=self.num_features
            )(atomic_numbers)  # shape: (N,1,1,F)

        psi_ = psi // jnp.inf  # -1 if psi < 0 and 0 otherwise
        psi_ = psi_.astype(jnp.int32)  # shape: (num_graphs)

        k = e3x.nn.Embed(
            num_embeddings=2,
            features=self.num_features
        )(psi_)[batch_segments]  # shape: (N, 1,1, F)

        v = e3x.nn.Embed(
            num_embeddings=2,
            features=self.num_features
        )(psi_)[batch_segments]  # shape: (N, 1,1, F)

        q_x_k = (q*k).sum(axis=-1) / jnp.sqrt(self.num_features)  # shape: (N, 1,1)
        y = nn.softplus(q_x_k)  # shape: (N, 1,1)

        denominator = jax.ops.segment_sum(
            jnp.squeeze(y, axis=(-1,-2)),
            segment_ids=batch_segments,
            num_segments=len(graph_mask)
        )  # (num_graphs)

        denominator = jnp.where(
            graph_mask,
            denominator,
            jnp.asarray(1., dtype=q.dtype)
        )  # (num_graphs)

        # we have to expand dims here
        # TODO is jnp.expand_dims faster?
        a = psi[batch_segments].reshape(-1,1,1) * y / denominator[batch_segments].reshape(-1,1,1)  # shape: (N, 1,1)

        e_psi = Residual(
            use_bias=False,
            activation_fn=getattr(jax.nn, self.activation_fn) if self.activation_fn != 'identity' else lambda u: u
        )(jnp.expand_dims(a, axis=-1) * v)  # shape: (N, 1,1, F)

        return e_psi


class ChargeEmbedSparse(nn.Module):
    prop_keys: Dict
    num_features: int
    activation_fn: str = 'silu'
    zmax: int = 118
    module_name: str = 'charge_embed_sparse'

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs):
        """

        Args:
           inputs (Dict):
                atomic_numbers (Array): atomic types, shape: (N)
                total_charge (Array): total charge, shape: (num_graphs)
                graph_mask (Array): (num_graphs)
                batch_segments (Array): (N)
            *args ():
            **kwargs ():

        Returns:

        """
        atomic_numbers = inputs['atomic_numbers']
        Q = inputs['total_charge']
        graph_mask = inputs['graph_mask']
        batch_segments = inputs['batch_segments']

        if Q is None:
            raise ValueError(
                f'ChargeEmbedSparse requires to pass `total_charge != None`.'
            )

        return DelocalizedEmbedSparse(
            zmax=self.zmax,
            num_features=self.num_features,
            activation_fn=self.activation_fn
        )(
            atomic_numbers=atomic_numbers,
            psi=Q,
            batch_segments=batch_segments,
            graph_mask=graph_mask
            )

    def __dict_repr__(self):
        return {self.module_name: {'num_features': self.num_features,
                                   'zmax': self.zmax,
                                   'prop_keys': self.prop_keys}}


class SpinEmbedSparse(nn.Module):
    prop_keys: Dict
    num_features: int
    activation_fn: str = 'silu'
    zmax: int = 118
    module_name: str = 'spin_embed_sparse'

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs):
        """

        Args:
           inputs (Dict):
                atomic_numbers (Array): atomic types, shape: (N)
                num_unpaired_electrons (Array): total charge, shape: (num_graphs)
                graph_mask (Array): (num_graphs)
                batch_segments (Array): (N)
            *args ():
            **kwargs ():

        Returns:

        """
        atomic_numbers = inputs['atomic_numbers']
        S = inputs['num_unpaired_electrons']
        graph_mask = inputs['graph_mask']
        batch_segments = inputs['batch_segments']

        if S is None:
            raise ValueError(
                f'SpinEmbedSparse requires to pass `num_unpaired_electrons != None`.'
            )

        return DelocalizedEmbedSparse(
            zmax=self.zmax,
            num_features=self.num_features,
            activation_fn=self.activation_fn
        )(
            atomic_numbers=atomic_numbers,
            psi=S,
            batch_segments=batch_segments,
            graph_mask=graph_mask
            )

    def __dict_repr__(self):
        return {self.module_name: {'num_features': self.num_features,
                                   'zmax': self.zmax,
                                   'prop_keys': self.prop_keys}}