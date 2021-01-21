from batchopt.optimizers.grey_wolf_optimizer import GreyWolfOptimizer


def test_gwo_init(optimizer_init_test_func):
    optimizer_init_test_func(GreyWolfOptimizer)


def test_gwo_ask(optimizer_ask_test_func):
    optimizer_ask_test_func(GreyWolfOptimizer)


def test_gwo_train(optimizer_optimize_test_func):
    optimizer_optimize_test_func(GreyWolfOptimizer)
