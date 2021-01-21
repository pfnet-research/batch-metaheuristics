import pytest

from batchopt.optimizers.base_optimizer import BaseOptimizer

BaseOptimizer.__abstractmethods__ = set()  # type: ignore


@pytest.fixture
def optimizer(default_optimizer):
    return default_optimizer(BaseOptimizer)


def test_base_init(optimizer_init_test_func):
    optimizer_init_test_func(BaseOptimizer)


def test_base_sort_population_by_fitness(optimizer, position, fitness):
    optimizer.prev_position = position
    optimizer.prev_fitness = fitness
    optimizer.sort_population_by_fitness()
    assert optimizer.prev_position.shape == (10, 2)
    assert optimizer.prev_fitness.shape == (10,)
    for i in range(len(optimizer.prev_fitness) - 1):
        assert optimizer.prev_fitness[i] < optimizer.prev_fitness[i + 1]


def test_base_update(optimizer, position, fitness, g_best_pos, g_best_fit):
    optimizer.min_index = 0
    optimizer.g_best_pos = g_best_pos
    optimizer.g_best_fit = g_best_fit
    g_best_pos, g_best_fit = optimizer.update(position, fitness)
    assert g_best_pos.shape == (2,)
    assert isinstance(g_best_fit, float)


def test_base_update_local_best(
    optimizer, position, fitness, local_best_position, local_best_fitness
):
    optimizer.local_best_position = local_best_position
    optimizer.local_best_fitness = local_best_fitness
    optimizer.update_local_best(position, fitness)
    assert optimizer.local_best_position.shape == (10, 2)
    assert optimizer.local_best_fitness.shape == (10,)


def test_base_update_only_improved_position(optimizer, position, fitness):
    optimizer.prev_position = position
    optimizer.prev_fitness = fitness
    position, fitness = optimizer.update_only_improved_position(position, fitness)
    assert position.shape == (10, 2)
    assert fitness.shape == (10,)


def test_base_output_log(capfd, g_best_fit):
    optimizer = BaseOptimizer(
        # objective_func=batch_styblinski_tang,
        domain_range=[(-1, 1), (-1, 1)],
        log=True,
        epoch=2,
        pop_size=10,
    )
    optimizer.g_best_fit = g_best_fit
    optimizer.output_log()
    out, err = capfd.readouterr()
    assert out == "> Epoch: {}, Best fit: {}\n".format(
        optimizer.current_epoch + 1, optimizer.g_best_fit
    )
    assert err == ""
