from batchopt.optimizers.life_choice_based_optimization import (
    LifeChoiceBasedOptimization,
)


def test_lcbo_init(optimizer_init_test_func):
    optimizer_init_test_func(LifeChoiceBasedOptimization)


def test_lcbo_ask(optimizer_ask_test_func):
    optimizer_ask_test_func(LifeChoiceBasedOptimization)


def test_lcbo_train(optimizer_optimize_test_func):
    optimizer_optimize_test_func(LifeChoiceBasedOptimization)
