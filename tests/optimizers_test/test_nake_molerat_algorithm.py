from batchopt.optimizers.nake_molerat_algorithm import NakeMoleratAlgorithm


def test_nmra_init(optimizer_init_test_func):
    optimizer_init_test_func(NakeMoleratAlgorithm)


def test_nmra_ask(optimizer_ask_test_func):
    optimizer_ask_test_func(NakeMoleratAlgorithm)


def test_nmra_train(optimizer_optimize_test_func):
    optimizer_optimize_test_func(NakeMoleratAlgorithm)
