import numpy as np
import pytest

from batchopt.optimizers.ant_lion_optimizer import AntLionOptimizer


def test_alo_init(optimizer_init_test_func):
    optimizer_init_test_func(AntLionOptimizer)


@pytest.mark.parametrize("epoch", [0, 6, 14, 16, 19, 20])
def test_alo__random_walk_around_antlion(epoch, default_optimizer):
    optimizer = default_optimizer(AntLionOptimizer)
    optimizer.g_best_pos = np.array([-0.54253259, -0.4909204])
    result = optimizer._random_walk_around_antlion(optimizer.g_best_pos, epoch)
    assert result.shape == (2, 2)


def test_alo__roulette_wheel_selection(default_optimizer, fitness):
    optimizer = default_optimizer(AntLionOptimizer)
    optimizer.prev_fitness = fitness
    result = optimizer._roulette_wheel_selection(1.0 / optimizer.prev_fitness)
    assert isinstance(result, int)


def test_alo_ask(optimizer_ask_test_func):
    optimizer_ask_test_func(AntLionOptimizer)


def test_alo_train(optimizer_optimize_test_func):
    optimizer_optimize_test_func(AntLionOptimizer)
