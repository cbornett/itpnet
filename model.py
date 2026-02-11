import functools
import e3x
import flax.linen as nn
from model_utils import SharedInteractionConfig, EquivariantLayerNorm, Residual
from embedding import DelocalizedEmbedSparse

class ITPNet(nn.Module):
    config: SharedInteractionConfig

    # computes the forward pass
    # retrieves feature representation from the input
    def forward(self, positions, batch):
        
        ### input & embedding

        # get input from batch
        atomic_numbers = batch['atomic_numbers']
        dst_idx = batch['dst_idx']
        src_idx = batch['src_idx']
        num_unpaired_electrons = batch['num_unpaired_electrons'] 
        total_charge = batch['total_charge'] 
        batch_segments = batch['batch_segments']
        graph_mask = batch['graph_mask']
    
        # calculate displacement vectors
        positions_dst = e3x.ops.gather_dst(positions, dst_idx=dst_idx)
        positions_src = e3x.ops.gather_src(positions, src_idx=src_idx)
        displacements = positions_src - positions_dst

        # basis expansion of distance and angular information
        basis = e3x.nn.basis(
            displacements,
            num=self.config.num_basis_functions,
            max_degree=max(self.config.mp_max_degree, self.config.init_mp_max_degree),
            radial_fn= getattr(e3x.nn, self.config.radial_fn, e3x.nn.reciprocal_bernstein),
            cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.config.cutoff)
        )

        # nuclear charge embedding
        e_Z = e3x.nn.Embed(num_embeddings=self.config.max_atomic_number+1, features=self.config.num_features)(atomic_numbers)

        # total charge embedding
        e_Q = DelocalizedEmbedSparse(
            num_features = self.config.num_features,
            zmax = self.config.max_atomic_number,
            shared_embed = self.config.shared_embed,
        )(
            atomic_numbers=atomic_numbers,
            psi = total_charge, ### ChargeEmbed
            batch_segments=batch_segments,
            graph_mask=graph_mask,
            e_Z=e_Z,
        )

        # total spin embedding
        e_S = DelocalizedEmbedSparse(
            num_features = self.config.num_features,
            zmax = self.config.max_atomic_number,
            shared_embed=self.config.shared_embed,
        )(
            atomic_numbers=atomic_numbers,
            psi = num_unpaired_electrons, ### SpinEmbed
            batch_segments=batch_segments,
            graph_mask=graph_mask,
            e_Z=e_Z,
        )

        assert e_Z.shape == e_Q.shape and e_Z.shape == e_S.shape, f"Embeddings do not have identical shape. {e_Z.shape}, {e_Q.shape} and {e_S.shape} are not equal."

        # combine embeddings
        x = e_Z + e_Q + e_S 

        ### refinement of feature representation

        # dense transformation of basis functions
        if self.config.do_feature_basis_align:
            for _ in range(1):
                basis = e3x.nn.Dense(features=self.config.num_features)(basis)
                basis = e3x.nn.silu(basis)

        # one message passing step
        y = e3x.nn.MessagePass(max_degree=self.config.init_mp_max_degree, include_pseudotensors=self.config.include_pseudotensors)(x, basis, dst_idx=dst_idx, src_idx=src_idx, num_segments=x.shape[0])
        
        # skip connection
        z = e3x.nn.Dense(features=self.config.num_features)(x)
        y = e3x.nn.add(y,z)

        # residual block
        if self.config.mp_res_block:
            y = Residual(num_blocks=self.config.num_mp_res_block)(y)

        # equivariant layer norm
        if self.config.do_eqv_norm: y = EquivariantLayerNorm()(y)

        # dense
        y = e3x.nn.Dense(features=self.config.num_itp_features)(y)
        
        # itp layer
        features = [y]
        for i in range(self.config.num_itp_iterations):
            y_pre_itp = features[-1]
            
            # dense transformation + equivariant tensor product
            if i == (self.config.num_itp_iterations-1):
                # compute scalar (invariant) feature representation for last iteration
                y_itp = e3x.nn.TensorDense(include_pseudotensors=False, max_degree=0,features=self.config.num_itp_features)(y_pre_itp)
            else: 
                y_itp = e3x.nn.TensorDense(include_pseudotensors=self.config.include_pseudotensors, max_degree=self.config.itp_max_degree, features=self.config.num_itp_features)(y_pre_itp)

            # residual block
            if self.config.itp_res_block:
                z = y_itp
                y_itp = Residual(num_blocks=self.config.num_itp_res_block)(z)

            # skip connection
            if i == (self.config.num_itp_iterations-1): 
                # produce only scalars & exclude irreps for last iteration
                y_pre_itp = e3x.nn.change_max_degree_or_type(y_pre_itp, max_degree=0, include_pseudotensors=False)

            # equivariant layer norm 
            if self.config.do_eqv_norm: y_itp = EquivariantLayerNorm()(y_itp)
            
            # skip connection
            y_itp = e3x.nn.add(y_pre_itp, y_itp)
                
            # append to list of itp layer feature representation results
            features.append(y_itp)
        
        ### readout

        if self.config.readout == 'last':
            # retrieve 'last'
            y = features[-1]
            x = e3x.nn.change_max_degree_or_type(y, max_degree=0, include_pseudotensors=False)
        

        elif self.config.readout == 'all':
            x = e3x.nn.change_max_degree_or_type(x, max_degree=0, include_pseudotensors=False) # features[0] CHANGED
            x = e3x.nn.Dense(
                self.config.num_features,
                use_bias=False
            )(x)

            # aggregate features
            for feat in features:

                x_aggr = e3x.nn.change_max_degree_or_type(feat, max_degree=0, include_pseudotensors=False)

                x = e3x.nn.add(
                    x,

                    nn.Dense(
                        self.config.num_features, 
                        use_bias=False,
                    )(x_aggr)
                ) 
            
        else:
            raise ValueError(
                f"Readout must be one of 'last' and 'all'. "
                f"Received {self.config.readout=}."
            )
        return x
        
    @nn.compact
    def __call__(self, positions, batch): 
        
        return self.forward(positions, batch)
    

# class representing the full network of feature extractor & desired property heads
class GeneralModel(nn.Module):
    
    properties: dict # dictionary of property heads with {'property': PropertyHead object}
    feature_extractor = None
    calc_forces: bool = True

    # apply the forward step of the feature extractor and predict properties such as energy and forces
    def forward(self, positions, batch):

        # compute forward pass of the feature extractor
        x = self.feature_extractor(positions, batch)

        # pass computed features to property heads
        y = jax.tree.map(lambda p: p(
                            x,
                            batch,
            ), 
            self.properties)

        energy_sum, energy = y['energy']
        y = y.copy({'energy':energy})

        # returns the negative energy sum and a dictionary with {'property': prediction} which can then be used in the loss
        return energy_sum, y


    @nn.compact
    def __call__(self, positions, batch): 

        # call model forward to get features
        if self.calc_forces:
            (_, y), forces = jax.value_and_grad(self.forward, argnums=0, has_aux=True)(positions, batch)
            y = y.copy({'forces':forces})

        else:
            _, y = self.forward(positions, batch)

        return y 
    
