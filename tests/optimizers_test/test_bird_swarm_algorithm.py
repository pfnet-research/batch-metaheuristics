from batchopt.benchfunctions import batch_styblinski_tang
from batchopt.optimizers.bird_swarm_algorithm import BirdSwarmAlgorithm


def test_bsa_init(optimizer_init_test_func):
    optimizer_init_test_func(BirdSwarmAlgorithm)


def test_bsa_train():
    optimizer = BirdSwarmAlgorithm(
        domain_range=[(-1, 1), (-1, 1)], log=False, epoch=750, pop_size=100,
    )
    result = optimizer.optimize(batch_styblinski_tang)
    assert result.g_best_pos.shape == (2,)
    assert isinstance(result.g_best_fit, float)
    assert len(result.loss_train) == 750
