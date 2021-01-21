from batchopt.benchfunctions import batch_styblinski_tang
from batchopt.optimizers.manta_ray_foraging_optimization import (
    MantaRayForagingOptimization,
)


def test_mrfo_init():
    optimizer = MantaRayForagingOptimization(
        domain_range=[(-1, 1), (-1, 1)], log=False, epoch=750, pop_size=100, S=2,
    )
    assert optimizer.problem_size == 2
    assert optimizer.log is False
    assert optimizer.epoch == 750
    assert optimizer.pop_size == 100
    assert optimizer.S == 2


def test_mrfo_ask(optimizer_ask_test_func):
    optimizer_ask_test_func(MantaRayForagingOptimization)


def test_mrfo_train():
    optimizer = MantaRayForagingOptimization(
        domain_range=[(-1, 1), (-1, 1)], log=False, epoch=750, pop_size=100,
    )
    result = optimizer.optimize(batch_styblinski_tang)
    assert result.g_best_pos.shape == (2,)
    assert isinstance(result.g_best_fit, float)
    assert len(result.loss_train) == 750
