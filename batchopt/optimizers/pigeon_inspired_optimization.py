import numpy as np
from numpy import exp, sum
from numpy.random import uniform

from batchopt.optimizers.base_optimizer import BaseOptimizer, RandomRelocator


class PigeonInspiredOptimization(BaseOptimizer):
    """
        Pigeon-Inspired Optimization
    """

    def __init__(
        self,
        domain_range,
        log=True,
        epoch=750,
        pop_size=100,
        relocator=RandomRelocator,
        R=0.2,
        n_switch=0.75,
    ):
        super().__init__(domain_range, log, epoch, pop_size, relocator)
        self.R = R
        if n_switch < 1:
            self.n_switch = int(self.epoch * n_switch)
        else:
            self.n_switch = int(n_switch)  # Represent Nc1 and Nc2 in the paper
        self.n_p = int(self.pop_size / 2)

    @property
    def is_update_improved(self):
        return True

    def generate_new_position(self):
        prev_position = self.prev_position
        prev_fitness = self.prev_fitness
        new_position_list = []
        if self.current_epoch < self.n_switch:  # Map and compass operations
            for i in range(self.pop_size):
                v_new = prev_position[i] * exp(
                    -self.R * (self.current_epoch + 1)
                ) + uniform() * (self.g_best_pos - prev_position[i])
                x_new = prev_position[i] + v_new
                new_position_list.append(x_new)

        else:  # Landmark operations
            sorted_index = np.argsort(prev_fitness)
            fitness = prev_fitness[sorted_index]
            position = prev_position[sorted_index]
            list_fit = [fitness[i] for i in range(self.n_p)]
            list_pos = [position[i] for i in range(self.n_p)]
            frac_up = sum([list_fit[i] * list_pos[i] for i in range(self.n_p)], axis=0)
            frac_down = self.n_p * sum(list_fit)
            x_c = frac_up / frac_down

            for i in range(self.pop_size):
                x_new = position[i] + uniform() * (x_c - position[i])
                new_position_list.append(x_new)
        return new_position_list
