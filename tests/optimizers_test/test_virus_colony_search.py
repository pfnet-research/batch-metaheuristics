import numpy as np

from batchopt.optimizers.virus_colony_search import VirusColonySearch


def test_vcs_init(optimizer_init_test_func):
    optimizer_init_test_func(VirusColonySearch)


def test_vcs_diffusion(default_optimizer, position, g_best_pos):
    optimizer = default_optimizer(VirusColonySearch)
    optimizer.g_best_pos = g_best_pos
    result = optimizer.diffusion(position)
    assert len(result) == 10
    assert result[0].shape == (2,)


def test_vcs_host_cells_infection(default_optimizer):
    optimizer = default_optimizer(VirusColonySearch)
    optimizer.x_mean = np.array([-0.02067053, -0.09922654])
    result = optimizer.host_cells_infection()
    assert len(result) == 10
    assert result[0].shape == (2,)


def test_vcs_immune_response(default_optimizer, position, fitness):
    optimizer = default_optimizer(VirusColonySearch)
    result = optimizer.immune_response(position, fitness)
    assert len(result) == 10
    assert result[0].shape == (2,)


def test_vcs_ask(optimizer_ask_test_func):
    optimizer_ask_test_func(VirusColonySearch)


def test_vcs_train(optimizer_optimize_test_func):
    optimizer_optimize_test_func(VirusColonySearch)
