from copy import deepcopy

import numpy as np

from batchopt.optimizers.base_optimizer import BaseOptimizer, RandomRelocator


class SineCosineAlgorithm(BaseOptimizer):
    """
    Sine-Cosine Algorithm
    """

    A = 2.0

    def __init__(
        self,
        domain_range,
        log=True,
        epoch=750,
        pop_size=100,
        relocator=RandomRelocator,
    ):
        super().__init__(domain_range, log, epoch, pop_size, relocator)

    @property
    def is_update_improved(self):
        return True

    def generate_new_position(self):
        prev_position = self.prev_position
        r1 = self.A - (self.current_epoch + 1) * (self.A / self.epoch)
        r2 = 2 * np.pi * np.random.uniform(size=(self.pop_size, self.problem_size))
        r3 = 2 * np.random.uniform(size=(self.pop_size, self.problem_size))
        r4 = np.random.uniform(size=(self.pop_size, self.problem_size))
        # Update the position of solutions with respect to destination
        temp1 = deepcopy(prev_position)
        temp2 = deepcopy(prev_position)
        broadcasted_g_best_pos = np.broadcast_to(
            self.g_best_pos, (self.pop_size, self.problem_size)
        )
        temp1 = temp1 + r1 * np.sin(r2) * np.abs(r3 * broadcasted_g_best_pos - temp1)
        temp2 = temp2 + r1 * np.cos(r2) * np.abs(r3 * broadcasted_g_best_pos - temp2)
        new_position_list = np.where(r4 >= 0.5, temp1, temp2)
        # TODO: this is improve idea: using same random in population axis.
        # rand_amend = np.random.uniform(
        #     self.domain_range[0], self.domain_range[1], size=(self.pop_size, 1),
        # )
        # rand_amend = np.broadcast_to(rand_amend, (self.pop_size, self.problem_size))
        return new_position_list
