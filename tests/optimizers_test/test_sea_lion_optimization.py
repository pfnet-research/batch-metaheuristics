from batchopt.optimizers.sea_lion_optimization import SeaLionOptimization


def test_slo_init(optimizer_init_test_func):
    optimizer_init_test_func(SeaLionOptimization)


def test_slo_ask(optimizer_ask_test_func):
    optimizer_ask_test_func(SeaLionOptimization)


def test_slo_train(optimizer_optimize_test_func):
    optimizer_optimize_test_func(SeaLionOptimization)
