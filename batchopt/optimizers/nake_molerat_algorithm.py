from copy import deepcopy

from numpy.random import choice, uniform

from batchopt.optimizers.base_optimizer import BaseOptimizer


class NakeMoleratAlgorithm(BaseOptimizer):
    """
    Naked Mole-rat Algorithm (NMR)
    """

    def __init__(
        self, domain_range, log=True, epoch=750, pop_size=100, bp=0.75,
    ):
        super().__init__(domain_range, log, epoch, pop_size)
        self.size_b = int(self.pop_size / 5)
        self.size_w = self.pop_size - self.size_b
        self.bp = bp  # breeding probability (0.75)

    @property
    def is_update_improved(self):
        return True

    def optional_initialization(self):
        self.sort_population_by_fitness()

    def generate_new_position(self):
        new_position_list = []
        for i in range(self.pop_size):
            temp = deepcopy(self.prev_position[i])
            if i < self.size_b:  # breeding operators
                if uniform() < self.bp:
                    alpha = uniform()
                    temp = (1 - alpha) * self.prev_position[i] + alpha * (
                        self.g_best_pos - self.prev_position[i]
                    )
            else:  # working operators
                t1, t2 = choice(range(self.size_b, self.pop_size), 2, replace=False)
                temp = self.prev_position[i] + uniform() * (
                    self.prev_position[t1] - self.prev_position[t2]
                )
            new_position_list.append(temp)
        return new_position_list
