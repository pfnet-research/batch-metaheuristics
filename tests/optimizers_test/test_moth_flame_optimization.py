from batchopt.optimizers.moth_flame_optimization import MothFlameOptimization


def test_mfo_init(optimizer_init_test_func):
    optimizer_init_test_func(MothFlameOptimization)


def test_mfo_train(optimizer_optimize_test_func):
    optimizer_optimize_test_func(MothFlameOptimization)
