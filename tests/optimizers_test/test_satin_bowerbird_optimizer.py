import numpy as np

from batchopt.optimizers.satin_bowerbird_optimizer import SatinBowerbirdOptimizer


def test_sbo_init():
    optimizer = SatinBowerbirdOptimizer(
        domain_range=[(-1, 1), (-1, 1)],
        log=False,
        epoch=750,
        pop_size=100,
        alpha=0.94,
        pm=0.05,
        z=0.02,
    )
    assert optimizer.problem_size == 2
    assert optimizer.log is False
    assert optimizer.epoch == 750
    assert optimizer.pop_size == 100
    assert optimizer.alpha == 0.94
    assert optimizer.p_m == 0.05
    assert optimizer.z == 0.02


def test_sbo__roulette_wheel_selection(default_optimizer):
    optimizer = default_optimizer(SatinBowerbirdOptimizer)
    fitness_list = np.array(
        [
            0.06574828,
            0.10262336,
            0.13785525,
            0.19278564,
            0.04575697,
            0.16031539,
            0.0374833,
            0.1901998,
            0.03609407,
            0.03113795,
        ]
    )
    result = optimizer._roulette_wheel_selection(fitness_list)
    assert isinstance(result, np.int64)


def test_sbo_ask(optimizer_ask_test_func):
    optimizer_ask_test_func(SatinBowerbirdOptimizer)


def test_sbo_train(optimizer_optimize_test_func):
    optimizer_optimize_test_func(SatinBowerbirdOptimizer)
