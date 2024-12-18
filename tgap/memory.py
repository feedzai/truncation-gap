import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.tree_util import tree_flatten, tree_unflatten


def state_store(num_nodes, init_state, numpy=True):
    sample_state = init_state(1)
    leaves, treedef = tree_flatten(sample_state)
    del sample_state
    shapes = [(-1,) + np.shape(l)[1:] for l in leaves]

    sizes = np.cumsum([0] + [np.size(l) for l in leaves])

    def unvectorize(v):
        v = [v[..., start:stop].reshape(shape) for start, stop, shape in zip(sizes, sizes[1:], shapes)]
        return tree_unflatten(treedef, v)

    def vectorize(u):
        return jnp.concatenate([ui.reshape((ui.shape[0], -1)) for ui in tree_flatten(u)[0]], -1)
    
    jit_vectorize = jit(vectorize)

    def init(*args, **kwargs):
        state = vectorize(init_state(num_nodes, *args, **kwargs))
        return np.array(state) if numpy else state
    
    @jit
    def get(store, indexes):
        return unvectorize(store[indexes])

    if numpy:
        def set(store, indexes, values):
            store[indexes] = jit_vectorize(values)
            return store
    else:
        @jit
        def set(store, indexes, values):
            return store.at[indexes].set(vectorize(values))

    return init, get, set
