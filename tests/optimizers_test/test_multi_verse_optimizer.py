import numpy as np

from batchopt.optimizers.multi_verse_optimizer import MultiVerseOptimizer


def test_mvo_init(optimizer_init_test_func):
    optimizer_init_test_func(MultiVerseOptimizer)


def test_mvo__roulette_wheel_selection(default_optimizer):
    optimizer = default_optimizer(MultiVerseOptimizer)
    weights = np.array(
        [
            -2.89687573,
            -0.02079452,
            -2.73526714,
            -8.47993176,
            -10.58704968,
            -8.50050021,
            -3.42264851,
            0.34152638,
            -8.95118705,
            -9.51461537,
        ]
    )
    result = optimizer._roulette_wheel_selection(weights)
    assert isinstance(result, int)


def test_mvo_ask(optimizer_ask_test_func):
    optimizer_ask_test_func(MultiVerseOptimizer)


def test_mvo_train(optimizer_optimize_test_func):
    optimizer_optimize_test_func(MultiVerseOptimizer)
