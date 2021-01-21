from batchopt.optimizers.pigeon_inspired_optimization import PigeonInspiredOptimization


def test_pio_init(optimizer_init_test_func):
    optimizer_init_test_func(PigeonInspiredOptimization)


def test_pio_ask(optimizer_ask_test_func):
    optimizer_ask_test_func(PigeonInspiredOptimization)


def test_pio_train(optimizer_optimize_test_func):
    optimizer_optimize_test_func(PigeonInspiredOptimization)
