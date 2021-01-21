from batchopt.optimizers.sine_cosine_algorithm import SineCosineAlgorithm


def test_sca_init(optimizer_init_test_func):
    optimizer_init_test_func(SineCosineAlgorithm)


def test_sca_ask(optimizer_ask_test_func):
    optimizer_ask_test_func(SineCosineAlgorithm)


def test_sca_train(optimizer_optimize_test_func):
    optimizer_optimize_test_func(SineCosineAlgorithm)
