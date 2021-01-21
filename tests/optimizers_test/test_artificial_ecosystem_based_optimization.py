from batchopt.optimizers.artificial_ecosystem_based_optimization import (
    ArtificialEcosystemBasedOptimization,
)


def test_aeo_init(optimizer_init_test_func):
    optimizer_init_test_func(ArtificialEcosystemBasedOptimization)


def test_aeo_ask(optimizer_ask_test_func):
    optimizer_ask_test_func(ArtificialEcosystemBasedOptimization)


def test_aeo_train(optimizer_optimize_test_func):
    optimizer_optimize_test_func(ArtificialEcosystemBasedOptimization)
