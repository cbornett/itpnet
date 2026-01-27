import flax.linen as nn
import jax
import jax.numpy as jnp

# property head for energy prediction
class EnergyPrediction(nn.Module):

    max_atomic_number:int = 118
    batch_max_num_graph:int = 10
    use_dense_mlp: bool = False

    @nn.compact
    def get_energy(self, x, batch): 
        
        atomic_numbers = batch['atomic_numbers']
        batch_segments = batch['batch_segments']
        graph_mask = batch['graph_mask']

        batch_max_num_graph = self.batch_max_num_graph

        if batch_segments is None:
            batch_segments = jnp.zeros_like(atomic_numbers)
            assert self.batch_max_num_graph==1

        
        if self.use_dense_mlp:
            # dense transformation of output feature representation
            y = nn.Dense(features=x.shape[-1])(x)
            y = nn.silu(y)

            # aggregate to atom-wise energies
            atomic_energies = nn.Dense(features=1)(y) 
            atomic_energies = jnp.squeeze(atomic_energies, axis=(-1,-2,-3))

        else:
            # linear combination of output feature representation

            # element-wise bias
            element_bias = self.param('element_bias', lambda rng, shape:jnp.zeros(shape), (self.max_atomic_number+1))

            # linear combination
            atomic_energies = nn.Dense(1, use_bias=False, kernel_init=jax.nn.initializers.zeros)(x)

            # aggregate to atom-wise energies, add element-wise bias
            atomic_energies = jnp.squeeze(atomic_energies, axis=(-1,-2,-3))
            atomic_energies += element_bias[atomic_numbers]

        # sum up atom-wise energy contributions
        energy = jax.ops.segment_sum(atomic_energies, segment_ids=batch_segments, num_segments=batch_max_num_graph) #len(graph_mask)

        # set contributions from padding to zero
        energy = jnp.where(graph_mask, energy, jnp.array([0.], dtype=energy.dtype))

        return energy
    
    def __call__(self, x, batch):

        energy = self.get_energy(x, batch)

        return -jnp.sum(energy), energy # the negative sum of the per-graph energy is neccessary for the prediction of the atom-wise forces