# ITPNet

Provides an implementation of the Iterated Tensor Product Network (ITPNet).

## Overview

This repository contains an implementation of the Iterated Tensor Product Network (ITPNet) with JAX, flax and e3x. This implementation is part of my master thesis. It provides an overview, not a working example. The ITPNet is a strictly local machine learning force field (MLFF). It utilizes various types of information about molecular systems, such as interatomic distances and nuclear charges, as input and predicts energies and forces. In this implementation, other work was used. The total charge and spin embeddings are taken from Unke et al. (2019), other parts, such as the graph mask and loss calculation are taken from Frank et al. (2022).

## Requirements

jax >= 0.5

flax >= 0.10.1

e3x >= 1.0.2

numpy >= 2.0.2


## References
- Unke, O. T., Chmiela, S., Gastegger, M., Schütt, K. T., Sauceda, H. E., & Müller, K. R. (2021). SpookyNet: Learning force fields with electronic degrees of freedom and nonlocal effects. Nature communications, 12(1), 7273.

- Frank, T., Unke, O., & Müller, K. R. (2022). So3krates: Equivariant attention for interactions on arbitrary length-scales in molecular systems. Advances in Neural Information Processing Systems, 35, 29400-29413.

