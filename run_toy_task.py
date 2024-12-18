import argparse
import numpy as np
import jax

import jax.numpy as jnp
import optax
from jax.tree_util import tree_map
from jax import lax
from functools import partial
from pathlib import Path

from tgap.grnn import SymmetricGRUCell
from tgap.gloss import mse, MLP, with_loss, with_feedforward_loss, with_feedforward_and_truncated_grads
from tgap.memory import state_store
from tgap.data.buffer_task import get_sampler_link_regression
from hpt import GridSampler

RESULTS_BASE = ['results', 'toy']

parser = argparse.ArgumentParser('Truncation Gap on Toy Data')
parser.add_argument('-m', '--method', type=str, choices=['FBPTT', 'TBPTT'], help='Method name (FBPTT or TBPTT)')
parser.add_argument('--num_epochs', type=int, default=5000, help='Number of epochs')
parser.add_argument('--num_steps', type=int, default=1000, help='Number of steps per epoch')
parser.add_argument('--num_nodes', type=int, default=100, help='Number of nodes')
args = parser.parse_args()

method = args.method.upper()
NUM_EPOCHS = args.num_epochs
NUM_NODES = args.num_nodes
NUM_STEPS = args.num_steps

base_results_path = Path(*RESULTS_BASE)
base_results_path.mkdir(parents=True, exist_ok=True)

hpt_space = {
    'memory': [1, 2, 3, 4, 5],
    'seed': [5, 11, 42, 123, 1984],
    'learning_rate': 1e-3,
    'beta1': 0.9, 
    'beta2': 0.999,
    'weight_decay': 0.0001,
    'state_size': [32, 64, 128],
}

hpt_samples = GridSampler(hpt_space)

CELL = SymmetricGRUCell


def make_model_step(model):
    init_model_state, get_state, set_state = state_store(NUM_NODES, model.init_local, numpy=False)

    def init_model(_=None):
        return init_model_state()

    def step_model(params, states, edge):
        src, dst, feature, target = edge
        nodes = jnp.array((src, dst))
        
        batch_states = get_state(states, nodes)
        inputs = jnp.array([feature])
        new_batch_states, outputs = model.step(params, batch_states, inputs, target)
        states = set_state(states, nodes, new_batch_states)

        return states, outputs
    
    return init_model, step_model


def make_fbptt_unrolled(step_fun, step_data, optimizer, num_steps):

    def episodic_step(params, states, _=None):
        data_state, model_state = states
        data_state, new_edge = step_data(data_state)
        model_state, loss = step_fun(params, model_state, new_edge)
        return (data_state, model_state), loss

    def unrolled_step(params, state):
        state, losses = lax.scan(partial(episodic_step, params), state, None, num_steps)
        return jnp.mean(losses), state

    unrolled_step = jax.value_and_grad(unrolled_step, has_aux=True)
    
    @jax.jit
    def unrolled_episode(state):
        params, optimizer_state, data_state, model_state = state
        
        (loss, (data_state, model_state)), grads = unrolled_step(params, (data_state, model_state))

        updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
        params = optax.apply_updates(params, updates)
        
        return (params, optimizer_state, data_state, model_state), loss
    
    return unrolled_episode


def make_bptt_unrolled(step_fun, step_data, optimizer, num_steps):
    
    def episodic_step(params, states, _=None):
        accumulator, data_state, model_state = states
        data_state, new_edge = step_data(data_state)
        model_state, (loss, grads) = step_fun(params, model_state, new_edge)
        
        accumulator = tree_map(jnp.add, accumulator, grads)
        
        return (accumulator, data_state, model_state), loss

    @jax.jit
    def unrolled_episode(state):
        params, optimizer_state, data_state, model_state = state
        accumulator = tree_map(jnp.zeros_like, params)
        
        (accumulator, data_state, model_state), loss = lax.scan(partial(episodic_step, params), (accumulator, data_state, model_state), None, num_steps)

        grads = tree_map(lambda a: a/num_steps, accumulator)
        updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
        params = optax.apply_updates(params, updates)
        
        return (params, optimizer_state, data_state, model_state), jnp.mean(loss)
    
    return unrolled_episode


for iter_num, hpt in enumerate(hpt_samples):
    print('Running iteration:', iter_num)
    memory = hpt['memory']
    state_size = hpt['state_size']
    feature_size = 1
    
    # Load data
    init_data, step_data = get_sampler_link_regression(NUM_NODES, 
                                                    delay=memory,
                                                    feedthrough=False)
    
    # define model and optimizer
    gcell = CELL(state_size, feature_size)
    regressor = MLP([state_size, 1], 2*state_size, scalar_output=True)
    loss = with_loss(regressor, mse)

    optimizer = optax.chain(
        #optax.clip(1.0),
        optax.adamw(hpt['learning_rate'], b1=hpt['beta1'], b2=hpt['beta2'], weight_decay=hpt['weight_decay'])
    )

    # make episode step
    if method == 'FBPTT':
        model = with_feedforward_loss(gcell, loss)
        make_unrolled = make_fbptt_unrolled
    else:
        model = with_feedforward_and_truncated_grads(gcell, loss)
        make_unrolled = make_bptt_unrolled

    init_model, step_model = make_model_step(model)
    unrolled_episode = make_unrolled(step_model, step_data, optimizer, NUM_STEPS)

    # initialize model/data/optimizer state/params
    seed = hpt['seed']
    rng_key = jax.random.PRNGKey(seed)
    rng_data, rng_params = jax.random.split(rng_key, 2)
    data_state = init_data(rng_data)
    params = model.init_params(rng_params)
    model_state = init_model()
    optimizer_state = optimizer.init(params)
    
    # training loop
    losses = []
    for i in range(NUM_EPOCHS):
        state = (params, optimizer_state, data_state, model_state)
        
        (params, optimizer_state, data_state, _), l = unrolled_episode(state)

        losses.append(float(l))
        
        data_state = init_data(data_state[-1])  

    losses = np.array(losses)

    results_path = base_results_path / f'memory_{memory}' / f'state_{state_size}' / f'{seed % 5}.pkl'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'wb') as file:
        np.save(file, losses)