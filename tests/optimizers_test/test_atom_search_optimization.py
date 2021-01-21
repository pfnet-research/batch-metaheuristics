from copy import deepcopy

import numpy as np
import pytest
from mock import MagicMock

from batchopt.benchfunctions import batch_styblinski_tang
from batchopt.optimizers.atom_search_optimization import AtomSearchOptimization


@pytest.fixture
def optimizer(default_optimizer):
    return default_optimizer(AtomSearchOptimization)


@pytest.fixture
def position(optimizer):
    return np.random.uniform(
        optimizer.domain_range[0],
        optimizer.domain_range[1],
        (optimizer.pop_size, optimizer.problem_size),
    )


@pytest.fixture
def fitness(position, optimizer):
    return batch_styblinski_tang(position)


@pytest.fixture
def min_index(fitness, optimizer):
    return np.argmin(fitness)


@pytest.fixture
def g_best_pos(position, min_index, optimizer):
    return deepcopy(position[min_index])


@pytest.fixture
def g_best_fit(fitness, min_index, optimizer):
    return deepcopy(fitness[min_index])


def test_aso_init():
    optimizer = AtomSearchOptimization(
        domain_range=[(-1, 1), (-1, 1)],
        log=False,
        epoch=750,
        pop_size=100,
        alpha=50,
        beta=0.2,
    )
    assert optimizer.problem_size == 2
    assert optimizer.log is False
    assert optimizer.epoch == 750
    assert optimizer.pop_size == 100
    assert optimizer.alpha == 50
    assert optimizer.beta == 0.2


def test_aso_atom_acc_list(default_optimizer, position, fitness, g_best_pos):
    optimizer = default_optimizer(AtomSearchOptimization)
    optimizer.prev_position = position
    optimizer.prev_fitness = fitness
    optimizer.g_best_pos = g_best_pos
    assert optimizer.atom_acc_list.shape == (10, 2)


def test_aso__update_mass(default_optimizer):
    optimizer = default_optimizer(AtomSearchOptimization)
    mass_list = np.zeros(optimizer.pop_size)
    fitness = np.array(
        [
            -5.29724262,
            -4.6140039,
            -11.86161013,
            -2.46774383,
            -2.9666911,
            -7.68319523,
            0.0830319,
            -8.74812433,
            -7.2708638,
            -6.12842473,
        ]
    )
    expected = np.array(
        [
            -0.03041883,
            -0.03220953,
            -0.01755776,
            -0.03854962,
            -0.03697251,
            -0.02491104,
            -0.04772694,
            -0.02278621,
            -0.02578599,
            -0.02837407,
        ]
    )
    result = optimizer._update_mass(fitness, mass_list)
    assert result.shape == (10,)
    assert np.allclose(result, expected)


def test_aso__find_LJ_potential(default_optimizer):
    optimizer = default_optimizer(AtomSearchOptimization)
    average_dict = np.array(
        [
            [0.50885217],
            [0.68513988],
            [0.89980625],
            [0.98719058],
            [0.95261956],
            [1.12861034],
            [0.67707892],
            [0.49531578],
            [0.59134217],
            [0.66223693],
        ]
    )
    radius = np.array(
        [
            [1.01313107, 0.70639391, 0.93744939, 0.91773673, 0.2061784],
            [1.34804165, 1.64869285, 1.11221259, 1.04119745, 1.34247633],
            [0.30916381, 1.62410494, 0.07110678, 0.0, 1.11089278],
            [1.70147208, 0.0, 1.64306094, 1.62410494, 0.52335886],
            [0.23840666, 1.64306094, 0.0, 0.07110678, 1.12527626],
            [0.0, 1.70147208, 0.23840666, 0.30916381, 1.17823087],
            [1.27658668, 1.65390467, 1.03989994, 0.96880875, 1.32567764],
            [0.73473403, 1.00841291, 0.63766588, 0.61574294, 0.50028853],
            [1.71142209, 0.87184753, 1.54341894, 1.49114669, 0.89167578],
            [1.17823087, 0.52335886, 1.12527626, 1.11089278, 0.0],
        ]
    )
    expected = np.array(
        [
            [0.59874821, 0.59874821, 0.59874821, 0.59874821, 0.59874821],
            [0.59874821, 0.59874821, 0.59874821, 0.59874821, 0.59874821],
            [0.59874821, 0.59874821, 0.59874821, 0.59874821, -0.18430963],
            [0.59874821, 0.44423961, 0.59874821, 0.59874821, 0.44423961],
            [0.44423961, 0.59874821, 0.44423961, 0.44423961, 0.03916258],
            [0.44423961, 0.59874821, 0.44423961, 0.44423961, 0.44423961],
            [0.59874821, 0.59874821, 0.59874821, 0.59874821, 0.59874821],
            [0.59874821, 0.59874821, 0.59874821, 0.59874821, 0.59874821],
            [0.59874821, 0.59874821, 0.59874821, 0.59874821, 0.59874821],
            [0.59874821, 0.59874821, 0.59874821, 0.59874821, 0.59874821],
        ]
    )
    result = optimizer._find_LJ_potential(0, average_dict, radius)
    assert result.shape == (10, 5)
    assert np.allclose(result, expected)


def test_aso__acceleration(default_optimizer, position, fitness, g_best_pos):
    optimizer = default_optimizer(AtomSearchOptimization)
    mock = MagicMock()
    mock._update_mass.return_value = np.array(
        [
            -0.03041883,
            -0.03220953,
            -0.01755776,
            -0.03854962,
            -0.03697251,
            -0.02491104,
            -0.04772694,
            -0.02278621,
            -0.02578599,
            -0.02837407,
        ]
    )
    mock._find_LJ_potential.return_value = np.array(
        [
            [0.59874821, 0.59874821, 0.59874821, 0.59874821, 0.59874821],
            [0.59874821, 0.59874821, 0.59874821, 0.59874821, 0.59874821],
            [0.59874821, 0.59874821, 0.59874821, 0.59874821, -0.18430963],
            [0.59874821, 0.44423961, 0.59874821, 0.59874821, 0.44423961],
            [0.44423961, 0.59874821, 0.44423961, 0.44423961, 0.03916258],
            [0.44423961, 0.59874821, 0.44423961, 0.44423961, 0.44423961],
            [0.59874821, 0.59874821, 0.59874821, 0.59874821, 0.59874821],
            [0.59874821, 0.59874821, 0.59874821, 0.59874821, 0.59874821],
            [0.59874821, 0.59874821, 0.59874821, 0.59874821, 0.59874821],
            [0.59874821, 0.59874821, 0.59874821, 0.59874821, 0.59874821],
        ]
    )
    mass_list = np.zeros(optimizer.pop_size)
    optimizer.prev_position = position
    optimizer.prev_fitness = fitness
    optimizer.g_best_pos = g_best_pos
    result = optimizer._acceleration(
        optimizer.prev_position,
        optimizer.prev_fitness,
        mass_list,
        optimizer.g_best_pos,
        0,
    )
    assert result.shape == (10, 2)


def test_aso_ask(optimizer_ask_test_func):
    optimizer_ask_test_func(AtomSearchOptimization)


def test_aso_train(optimizer_optimize_test_func):
    optimizer_optimize_test_func(AtomSearchOptimization)
