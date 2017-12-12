from numpy import ndarray
import theano.tensor as T
import keras.backend as K

def temporal_padding(x, paddings=(1, 0), padvalue=0):
    '''Pad the middle dimension of a 3D tensor
    with `padding[0]` values left and `padding[1]` values right.

    Modified from keras.backend.temporal_padding
    https://github.com/fchollet/keras/blob/3bf913d/keras/backend/theano_backend.py#L590

    TODO: Implement for tensorflow (supposebly more easy)
    '''
    if not isinstance(paddings, (tuple, list, ndarray)):
        paddings = (paddings, paddings)

    input_shape = x.shape
    output_shape = (input_shape[0],
                    input_shape[1] + sum(paddings),
                    input_shape[2])
    output = T.zeros(output_shape)

    # Set pad value and set subtensor of actual tensor
    output = T.set_subtensor(output[:, :paddings[0], :], padvalue)
    output = T.set_subtensor(output[:, paddings[1]:, :], padvalue)
    output = T.set_subtensor(output[:, paddings[0]:x.shape[1] + paddings[0], :], x)
    return output

def neighbour_lookup(atoms, edges, maskvalue=0, include_self=False):
    ''' Looks up the features of an all atoms neighbours, for a batch of molecules.

    # Arguments:
        atoms (K.tensor): of shape (batch_n, max_atoms, num_atom_features)
        edges (K.tensor): of shape (batch_n, max_atoms, max_degree) with neighbour
            indices and -1 as padding value
        maskvalue (numerical): the maskingvalue that should be used for empty atoms
            or atoms that have no neighbours (does not affect the input maskvalue
            which should always be -1!)
        include_self (bool): if True, the featurevector of each atom will be added
            to the list feature vectors of its neighbours

    # Returns:
        neigbour_features (K.tensor): of shape (batch_n, max_atoms(+1), max_degree,
            num_atom_features) depending on the value of include_self

    # Todo:
        - make this function compatible with Tensorflow, it should be quite trivial
            because there is an equivalent of `T.arange` in tensorflow.
    '''

    # The lookup masking trick: We add 1 to all indices, converting the
    #   masking value of -1 to a valid 0 index.
    masked_edges = edges + 1
    # We then add a padding vector at index 0 by padding to the left of the
    #   lookup matrix with the value that the new mask should get
    masked_atoms = temporal_padding(atoms, (1,0), padvalue=maskvalue)


    # Import dimensions
    atoms_shape = K.shape(masked_atoms)
    batch_n = atoms_shape[0]
    lookup_size = atoms_shape[1]
    num_atom_features = atoms_shape[2]

    edges_shape = K.shape(masked_edges)
    max_atoms = edges_shape[1]
    max_degree = edges_shape[2]

    # create broadcastable offset
    offset_shape = (batch_n, 1, 1)
    offset = K.reshape(T.arange(batch_n, dtype=K.dtype(masked_edges)), offset_shape)
    offset *= lookup_size

    # apply offset to account for the fact that after reshape, all individual
    #   batch_n indices will be combined into a single big index
    flattened_atoms = K.reshape(masked_atoms, (-1, num_atom_features))
    flattened_edges = K.reshape(masked_edges + offset, (batch_n, -1))

    # Gather flattened
    flattened_result = K.gather(flattened_atoms, flattened_edges)

    # Unflatten result
    output_shape = (batch_n, max_atoms, max_degree, num_atom_features)
    output = T.reshape(flattened_result, output_shape)

    if include_self:
        return K.concatenate([K.expand_dims(atoms, 2), output], axis=2)
    return output
