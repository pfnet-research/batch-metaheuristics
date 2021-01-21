from copy import deepcopy

import numpy as np
from numpy.random import randint, uniform

from batchopt.optimizers.base_optimizer import BaseOptimizer, RandomRelocator


class ElectromagneticFieldOptimization(BaseOptimizer):
    """
    Electromagnetic Field Optimization (EFO)
    """

    phi = (1 + np.sqrt(5)) / 2

    def __init__(
        self,
        domain_range,
        log=True,
        epoch=750,
        pop_size=100,
        relocator=RandomRelocator,
        r_rate=0.3,
        ps_rate=0.85,
        p_field=0.1,
        n_field=0.45,
    ):
        super().__init__(domain_range, log, epoch, pop_size, relocator)

        self.r_rate = r_rate  # Like mutation parameter in GA but for one variable
        self.ps_rate = ps_rate  # Like crossover parameter in GA
        self.p_field = p_field
        self.n_field = n_field
        # golden ratio
        self.r_force = uniform(0, 1, self.epoch)  # random force in each generation

    @property
    def is_update_improved(self):
        return True

    def optional_initialization(self):
        self.pop_new = deepcopy(self.prev_position)

    def generate_new_position(self):
        self.sort_population_by_fitness()
        r = self.r_force[self.current_epoch]
        for i in range(self.pop_size):
            r_idx1 = randint(0, int(self.pop_size * self.p_field))
            r_idx2 = randint(int(self.pop_size * (1 - self.n_field)), self.pop_size)
            r_idx3 = randint(
                int((self.pop_size * self.p_field) + 1),
                int(self.pop_size * (1 - self.n_field)),
            )
            if uniform() < self.ps_rate:
                self.pop_new[i] = (
                    self.prev_position[r_idx3]
                    + self.phi
                    * r
                    * (self.prev_position[r_idx1] - self.prev_position[r_idx3])
                    + r * (self.prev_position[r_idx3] - self.prev_position[r_idx2])
                )
            else:
                self.pop_new[i] = self.prev_position[r_idx1]

        for i in range(self.pop_size):
            # replacement of one electromagnet of generated particle with a random number
            # (only for some generated particles) to bring diversity to the population
            if uniform() < self.r_rate:
                RI = randint(0, self.problem_size)
                self.pop_new[i][RI] = uniform(
                    self.domain_range[0][RI], self.domain_range[1][RI]
                )

        new_position_list = []
        # checking whether the generated number is inside boundary or not
        for i in range(self.pop_size):
            temp = self.pop_new[i]
            new_position_list.append(temp)
        return new_position_list
