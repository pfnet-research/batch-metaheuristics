from batchopt.optimizers.emperor_penguin_optimizer import EmperorPenguinOptimizer


def test_epo_init(optimizer_init_test_func):
    optimizer_init_test_func(EmperorPenguinOptimizer)


def test_epo_ask(optimizer_ask_test_func):
    optimizer_ask_test_func(EmperorPenguinOptimizer)


def test_epo_train(optimizer_optimize_test_func):
    optimizer_optimize_test_func(EmperorPenguinOptimizer)
