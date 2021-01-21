from copy import deepcopy

import numpy as np

from batchopt.optimizers.base_optimizer import BaseOptimizer


class EmperorPenguinOptimizer(BaseOptimizer):
    """
    Emperor penguin optimizer
    """

    M = 2

    @property
    def is_update_improved(self):
        return True

    def generate_new_position(self):
        prev_position = self.prev_position
        # First: Changed Eq. T_s
        T_s = 1.0 - 1.0 / (self.epoch - self.current_epoch)
        self.temp = deepcopy(prev_position)
        P_grid = np.abs(self.g_best_pos - prev_position)
        A = (
            self.M
            * (T_s + P_grid)
            * np.random.uniform(size=(self.pop_size, self.problem_size))
            - T_s
        )
        C = np.random.uniform(size=self.problem_size)
        f = np.random.uniform(2, 3, size=self.problem_size)
        ll = np.random.uniform(1.5, 2, size=self.problem_size)
        S_A = np.abs(f * np.exp(-self.current_epoch / ll) - np.exp(-self.current_epoch))
        D_ep = np.abs(S_A * self.g_best_pos - C * prev_position)
        temp = self.g_best_pos - A * D_ep
        return temp
