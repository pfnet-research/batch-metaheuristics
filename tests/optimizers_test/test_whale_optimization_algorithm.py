from batchopt.optimizers.whale_optimization_algorithm import WhaleOptimizationAlgorithm


def test_woa_init(optimizer_init_test_func):
    optimizer_init_test_func(WhaleOptimizationAlgorithm)


def test_woa_ask(optimizer_ask_test_func):
    optimizer_ask_test_func(WhaleOptimizationAlgorithm)


def test_woa_train(optimizer_optimize_test_func):
    optimizer_optimize_test_func(WhaleOptimizationAlgorithm)
