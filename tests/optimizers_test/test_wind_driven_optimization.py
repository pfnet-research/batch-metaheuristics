from batchopt.optimizers.wind_driven_optimization import WindDrivenOptimization


def test_wdo_init(optimizer_init_test_func):
    optimizer_init_test_func(WindDrivenOptimization)


def test_wdo_ask(optimizer_ask_test_func):
    optimizer_ask_test_func(WindDrivenOptimization)


def test_wdo_train(optimizer_optimize_test_func):
    optimizer_optimize_test_func(WindDrivenOptimization)
