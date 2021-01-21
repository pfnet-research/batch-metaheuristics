import numpy as np

from batchopt.optimizers.nuclear_reaction_optimization import (
    NuclearReactionOptimization,
)


def test_nro_init(optimizer_init_test_func):
    optimizer_init_test_func(NuclearReactionOptimization)


def test_nro_nfi_phase(default_optimizer, position, fitness):
    optimizer = default_optimizer(NuclearReactionOptimization)
    optimizer.current_epoch = 0
    optimizer.g_best_pos = np.array([-0.99446959, -0.90140898])
    result = optimizer.nfi_phase(position, fitness)
    assert len(result) == 10
    assert result[0].shape == (2,)


def test_nro_nfu_phase(default_optimizer, position, fitness, g_best_pos):
    optimizer = default_optimizer(NuclearReactionOptimization)
    optimizer.current_epoch = 1
    optimizer.g_best_pos = g_best_pos
    result = optimizer.nfu_phase(position, fitness)
    assert len(result) == 10
    assert result[0].shape == (2,)


def test_nro_calculate_pc(default_optimizer, position, fitness, g_best_pos):
    optimizer = default_optimizer(NuclearReactionOptimization)
    optimizer.current_epoch = 2
    optimizer.g_best_pos = g_best_pos
    result = optimizer.calculate_pc(position, fitness)
    assert len(result) == 10
    assert result[0].shape == (2,)


def test_nro_ask(optimizer_ask_test_func):
    optimizer_ask_test_func(NuclearReactionOptimization)


def test_nro_train(optimizer_optimize_test_func):
    optimizer_optimize_test_func(NuclearReactionOptimization)
