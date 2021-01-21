from copy import deepcopy

import numpy as np

from batchopt.optimizers.base_optimizer import BaseOptimizer


class SatinBowerbirdOptimizer(BaseOptimizer):
    """
    Satin-Bowerbird Optimizer
    """

    def __init__(
        self,
        domain_range,
        log=True,
        epoch=750,
        pop_size=100,
        alpha=0.94,
        pm=0.05,
        z=0.02,
    ):
        super().__init__(domain_range, log, epoch, pop_size)
        self.alpha = alpha  # the greatest step size
        self.p_m = pm  # mutation probability
        self.z = (
            z  # percent of the difference between the upper and lower limit (Eq. 7)
        )
        self.sigma = self.z * (
            self.domain_range[1] - self.domain_range[0]
        )  # proportion of space width

    @property
    def is_update_improved(self):
        return False

    def _roulette_wheel_selection(self, fitness_list=None):
        r = np.random.uniform()
        c = np.cumsum(fitness_list)
        f = np.where(r < c)[0][0]
        return f

    def generate_new_position(self):
        prev_position = self.prev_position
        prev_fitness = self.prev_fitness
        # Calculate the probability of bowers using Eqs. (1) and (2)
        fx_list = prev_fitness
        fit_list = deepcopy(fx_list)
        for i in range(self.pop_size):
            if fx_list[i] < 0:
                fit_list[i] = 1.0 + np.abs(fx_list[i])
            else:
                fit_list[i] = 1.0 / (1.0 + np.abs(fx_list[i]))
        fit_sum = np.sum(fit_list)
        # Calculating the probability of each bower
        prob_list = fit_list / fit_sum
        new_position_list = []
        for i in range(self.pop_size):
            temp = deepcopy(prev_position[i])
            for j in range(self.problem_size):
                # Select a bower using roulette wheel
                idx = self._roulette_wheel_selection(prob_list)
                # Calculating Step Size
                lamda = self.alpha / (1 + prob_list[idx])
                temp[j] = prev_position[i][j] + lamda * (
                    (prev_position[idx][j] + self.g_best_pos[j]) / 2
                    - prev_position[i][j]
                )
                # Mutation
                if np.random.uniform() < self.p_m:
                    temp[j] = (
                        prev_position[i][j] + np.random.normal(0, 1) * self.sigma[j]
                    )
            new_position_list.append(temp)
        return new_position_list
