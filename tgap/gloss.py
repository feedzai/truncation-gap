import jax
import jax.numpy as jnp
from jax import nn
from dataclasses import dataclass
from typing import Callable

from tgap.grnn import BaseGRNNCell, GRNNCell


@dataclass
class FFModel:
    init_params: Callable
    apply: Callable
    has_aux: bool = False

    def init_params(rng: jax.random.KeyArray):
        """Initializes parameters of model.

        Args:
            rng (KeyArray): PRNG key for initializer.

        Returns:
            pytree: Parameters of model.
        """
        raise NotImplementedError
    
    def apply(params, inputs):
        """Applies model.
        """
        raise NotImplementedError


def Dropout(rate=0.1):
    def apply(input, rng):
        if rng is None:
            return input
        u = jax.random.uniform(rng, input.shape)
        return jnp.where(u < rate, 0., input) / (1 - rate)

    return apply if rate > 0 else lambda input, rng: input


def MLP(
        hidden_sizes: list[int], 
        input_size: int, 
        activation: Callable=nn.relu, 
        last_activation: bool=False, 
        scalar_output: bool=False, 
        initializer=nn.initializers.lecun_normal,
        dropout=0
        ) -> FFModel:
    num_layers = len(hidden_sizes)
    apply_dropout = Dropout(dropout)

    def apply(params, inputs, rng=None):
        inputs = apply_dropout(inputs, rng)
        for i, param in enumerate(params):
            inputs = jnp.dot(inputs, param['w']) + param['b']
            if last_activation or i < num_layers - 1:
                inputs = activation(inputs)
            if i < num_layers - 1:
                inputs = apply_dropout(inputs, rng)
        return inputs
    
    def init(rng): 
        initializer_instance = initializer()
        rngs = jax.random.split(rng, num_layers)
        params = [
            {
                'w': initializer_instance(rng, (prev_size, size)),
                'b': jnp.zeros(size)
            }
            for rng, prev_size, size in zip(rngs, [input_size] + hidden_sizes, hidden_sizes)
        ]
        if scalar_output:
            assert hidden_sizes[-1] == 1
            params[-1]['w'] = params[-1]['w'][..., 0]
            params[-1]['b'] = params[-1]['b'][..., 0]
        return tuple(params)
    
    return FFModel(init, apply)


def bce(scale_pos_weight = 1, reduction=jnp.mean):
    def loss(logits, targets):
        abs_logits = jnp.abs(logits)
        losses = jnp.log1p(jnp.exp(-abs_logits))

        # target dependant part
        margin = logits * (1 - 2 * targets)
        losses += jnp.maximum(0, margin)
        if scale_pos_weight != 1:
            losses *= jnp.where(targets, scale_pos_weight, 1)
        return reduction(losses)

    return loss


def mse(preds, targets):
    return jnp.mean((preds - targets) ** 2)


def with_loss(mlp: FFModel, loss: Callable, mlp_output: bool=False) -> FFModel:
    def apply_with_loss(params, inputs, targets, rng=None):
        output = mlp.apply(params, inputs, rng=rng)
        l = loss(output, targets)
        return (l, output) if mlp_output else l
    
    return FFModel(mlp.init_params, apply_with_loss, mlp_output)


def with_feedforward_loss(cell: BaseGRNNCell, loss: FFModel) -> GRNNCell:
    def init_params(rng):
        cell_rng, loss_rng = jax.random.split(rng, 2)
        return {
            'cell': cell.init_params(cell_rng),
            'loss': loss.init_params(loss_rng)
        }
    
    def step_with_loss(params, states, inputs, targets, rng=None):
        new_states, outputs = cell.step(params['cell'], states, inputs)
        return new_states, loss.apply(params['loss'], outputs, targets, rng=rng)
    
    return GRNNCell(init_params, cell.init_local, step_with_loss)


def reverse_outputs(f: Callable):
    return lambda *args, **kwargs: f(*args, **kwargs)[::-1]


def with_truncated_grads(cell: BaseGRNNCell, has_aux=False) -> GRNNCell:
    if has_aux:
        def update_and_loss(*args, **kwargs):
            states, outputs = cell.step(*args, **kwargs)
            return outputs[0], (states, *outputs[1:])
        
        update_and_loss = jax.value_and_grad(update_and_loss, has_aux=True)

        def step_with_loss(params, states, inputs, targets, rng=None):
            (loss, (new_states, *extra_outputs)), grads = update_and_loss(params, states, inputs, targets, rng=rng)
            return new_states, (loss, grads, *extra_outputs)
    else:
        update_and_loss = jax.value_and_grad(reverse_outputs(cell.step), has_aux=True)

        def step_with_loss(params, states, inputs, targets, rng=None):
            (loss, new_states), grads = update_and_loss(params, states, inputs, targets, rng=rng)
            return new_states, (loss, grads)
        
    return GRNNCell(cell.init_params, cell.init_local, step_with_loss)


def with_feedforward_and_truncated_grads(cell: GRNNCell, loss: FFModel) -> GRNNCell:
    return with_truncated_grads(with_feedforward_loss(cell, loss), has_aux=loss.has_aux)


