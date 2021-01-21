from batchopt.optimizers.bald_eagle_search import BaldEagleSearch


def test_bes_init(optimizer_init_test_func):
    optimizer_init_test_func(BaldEagleSearch)


def test_bes__create_x_and_y(default_optimizer):
    optimizer = default_optimizer(BaldEagleSearch)
    result = optimizer._create_x_and_y()
    assert result.shape == (4,)


def test_bes_ask(optimizer_ask_test_func):
    optimizer_ask_test_func(BaldEagleSearch)


def test_bes_train(optimizer_optimize_test_func):
    optimizer_optimize_test_func(BaldEagleSearch)
