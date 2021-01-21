from batchopt.optimizers.electromagnetic_field_optimization import (
    ElectromagneticFieldOptimization,
)


def test_efo_init(optimizer_init_test_func):
    optimizer_init_test_func(ElectromagneticFieldOptimization)


def test_efo_ask(optimizer_ask_test_func):
    optimizer_ask_test_func(ElectromagneticFieldOptimization)


def test_efo_train(optimizer_optimize_test_func):
    optimizer_optimize_test_func(ElectromagneticFieldOptimization)
