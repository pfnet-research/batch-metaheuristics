from copy import deepcopy

import numpy as np
import pytest

from batchopt.benchfunctions import batch_styblinski_tang


@pytest.fixture
def default_optimizer():
    def def_opt(opt):
        return opt(domain_range=[(-1, 1), (-1, 1)], log=False, epoch=2, pop_size=10,)

    return def_opt


@pytest.fixture
def position():
    return np.random.uniform(np.array([-1, -1]), np.array([1, 1]), (10, 2),)


@pytest.fixture
def fitness(position):
    return batch_styblinski_tang(position)


@pytest.fixture
def min_index(fitness):
    return np.argmin(fitness)


@pytest.fixture
def g_best_pos(position, min_index):
    return deepcopy(position[min_index])


@pytest.fixture
def g_best_fit(fitness, min_index):
    return deepcopy(fitness[min_index])


@pytest.fixture
def local_best_position():
    return np.random.uniform(np.array([-1, -1]), np.array([1, 1]), (10, 2),)


@pytest.fixture
def local_best_fitness(local_best_position):
    return batch_styblinski_tang(local_best_position)


@pytest.fixture
def optimizer_init_test_func(default_optimizer):
    def test_init(opt):
        optimizer = default_optimizer(opt)
        assert optimizer.problem_size == 2
        assert optimizer.log is False
        assert optimizer.epoch == 2
        assert optimizer.pop_size == 10

    return test_init


@pytest.fixture
def optimizer_ask_test_func(default_optimizer):
    def test_ask(opt):
        optimizer = default_optimizer(opt)
        result = optimizer.ask()
        assert result.shape == (10, 2)

    return test_ask


@pytest.fixture
def optimizer_optimize_test_func(default_optimizer):
    def test_optimize(opt):
        optimizer = default_optimizer(opt)
        result = optimizer.optimize(batch_styblinski_tang)
        assert result.g_best_pos.shape == (2,)
        assert isinstance(result.g_best_fit, float)
        assert len(result.loss_train) == 2

    return test_optimize
