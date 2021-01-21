from batchopt.optimizers.teaching_learning_optimization import (
    TeachingLearningOptimization,
)


def test_tlo_init(optimizer_init_test_func):
    optimizer_init_test_func(TeachingLearningOptimization)


def test_tlo_ask(optimizer_ask_test_func):
    optimizer_ask_test_func(TeachingLearningOptimization)


def test_tlo_train(optimizer_optimize_test_func):
    optimizer_optimize_test_func(TeachingLearningOptimization)
