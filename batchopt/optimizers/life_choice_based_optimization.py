from copy import deepcopy

from numpy import array, ceil, mean, ones, sqrt
from numpy.random import uniform

from batchopt.optimizers.base_optimizer import BaseOptimizer


class LifeChoiceBasedOptimization(BaseOptimizer):
    """
    Batch-version of Life Choice-Based Optimization (LCBO)
    """

    def __init__(
        self, domain_range, log=True, epoch=750, pop_size=100, r1=2.35,
    ):
        super().__init__(domain_range, log, epoch, pop_size)
        self.r1 = r1

    @property
    def is_update_improved(self):
        True

    def optional_initialization(self):
        self.sort_population_by_fitness()

    def generate_new_position(self):
        # Originally, this algorithm use current position and update them.
        # In this functio, use previous ones for parallelize and reduce accuracy.
        new_position_list = []
        for i in range(self.pop_size):
            rand = uniform()
            self.sort_population_by_fitness()
            if rand > 0.875:  # Update using Eq. 1, update from n best solution
                n = int(ceil(sqrt(self.pop_size)))
                temp = array([uniform() * self.prev_position[j] for j in range(n)])
                temp = mean(temp, axis=0)
            elif rand < 0.7:  # Update using Eq. 2-6
                f = (self.current_epoch + 1) / self.epoch
                if i != 0:
                    better_diff = (
                        f
                        * self.r1
                        * (self.prev_position[i - 1] - self.prev_position[i])
                    )
                else:
                    better_diff = (
                        f * self.r1 * (self.g_best_pos - self.prev_position[i])
                    )
                best_diff = (
                    (1 - f) * self.r1 * (self.prev_position[0] - self.prev_position[i])
                )
                temp = (
                    self.prev_position[i]
                    + uniform() * better_diff
                    + uniform() * best_diff
                )
            else:
                x_min = self.domain_range[0] * ones(self.problem_size)
                x_max = self.domain_range[1] * ones(self.problem_size)
                temp = x_max - (self.prev_position[i] - x_min) * uniform(
                    self.domain_range[0], self.domain_range[1], self.problem_size
                )
            new_position_list.append(deepcopy(temp))
        return new_position_list
