import numpy as np
import jax
import jax.numpy as jnp
from functools import wraps
from jax import nn
from jax.tree_util import tree_map
from dataclasses import dataclass
from typing import Callable, ClassVar, Optional, Union


def local_state_initializer(entity_state_sizes: Union[int, list]):
    """Defines decorator for a local state initializer function.

    Args:
        entity_state_sizes (int | list): size of hidden state (if one type of entity) 
        or list of hidden state sizes for each type of entity.

    Returns:
        A decorator to apply to an initializer function.
    """
    if np.isscalar(entity_state_sizes): 
        def _local_state_initializer(initializer):
            @wraps(initializer)
            def wrapped_initializer(batch_size: Union[int, list], entity: None=None):
                shape = [batch_size, entity_state_sizes] if np.isscalar(batch_size) else [*batch_size, entity_state_sizes]
                    
                return initializer(shape)

            return wrapped_initializer
    else:
        def _local_state_initializer(initializer):
            @wraps(initializer)
            def wrapped_initializer(batch_size: Union[int, list], entity: int):
                state_size = entity_state_sizes[entity]
                shape = [batch_size, state_size] if np.isscalar(batch_size) else [*batch_size, state_size]
                    
                return initializer(shape)

            return wrapped_initializer
    
    return _local_state_initializer


def zero_local_state_initializer(entity_state_sizes: Union[int, list]):
    """Defines a local state initializer that outputs a single N-D zero array.

    Args:
        entity_state_sizes (int | list): size of hidden state (if one type of entity) 
        or list of hidden state sizes for each type of entity.

    Returns:
        An initializer function that outputs a N-D zero array of the appropriate size.
    """

    @local_state_initializer(entity_state_sizes)
    def initializer(shape):
        return jnp.zeros(shape)
    
    return initializer


class BaseGRNNCell:
    has_global: ClassVar = False

    @staticmethod
    def init_global():
        return None


@dataclass(frozen=True)
class GRNNCell(BaseGRNNCell):
    init_params: Callable
    init_local: Callable
    step: Callable
    num_entities: int = 2
    symmetric: Optional[bool] = False


def SymmetricGRUCell(state_size, input_size, initializer=nn.initializers.lecun_normal):
    def init_params(rng):
        initializer_instance = initializer()
        
        return {
            'wh_zr': initializer_instance(rng, (2, state_size, 2*state_size)),
            'wh_h': initializer_instance(rng, (2, state_size, state_size)),
            'wi': initializer_instance(rng, (input_size, 3*state_size)),
            'b': jnp.zeros(3*state_size),
        }

    init_state = zero_local_state_initializer(state_size)

    # States shape is (B, 2, H)
    # Inputs shape is (B, I) or (B, 2, I) in case inputs to src and dst are different
    state_axes = 2
    entity_axis = -2
    def step(params, states, inputs):
        winputs = jnp.tensordot(inputs, params['wi'], axes=1) + params['b']
        winputs_zr, winputs_h = winputs[..., :2*state_size], winputs[..., 2*state_size:]

        wstates_zr = (jnp.tensordot(states, params['wh_zr'], axes=state_axes), jnp.tensordot(states, params['wh_zr'][::-1], axes=state_axes))
        wstates_zr = jnp.stack(wstates_zr, axis=entity_axis) #-2
        zr = jax.nn.sigmoid(wstates_zr + winputs_zr) # [B, 2, 2H]
        
        z, r = jnp.split(zr, 2, axis=-1) # [B, 2, H]

        rstates = r*states
        wstates_h = (jnp.tensordot(rstates, params['wh_h'], axes=state_axes), jnp.tensordot(rstates, params['wh_h'][::-1], axes=state_axes))
        wstates_h = jnp.stack(wstates_h, axis=entity_axis) #-2
        h = jnp.tanh(wstates_h + winputs_h)

        next_state = (1 - z)*states + z*h
        return next_state, next_state.reshape(next_state.shape[:-2] + (-1,))

    return GRNNCell(init_params, init_state, step, symmetric=True)
