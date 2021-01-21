import numpy as np

from batchopt.optimizers.base_optimizer import BaseOptimizer


class WhaleOptimizationAlgorithm(BaseOptimizer):
    """
    Whale Optimization Algorithm
    """

    @property
    def is_update_improved(self):
        return False

    def generate_new_position(self):
        prev_position = self.prev_position

        a = 2 - 2 * self.current_epoch / (
            self.epoch - 1
        )  # linearly decreased from 2 to 0
        r = np.random.rand(self.pop_size)
        A = 2 * a * r - a
        C = 2 * r
        lotate = np.random.uniform(-1, 1, self.pop_size)
        p = 0.5
        b = 1
        new_position_list = []
        for i in range(self.pop_size):
            if np.random.uniform() < p:
                if np.abs(A[i]) < 1:
                    D = np.abs(C[i] * self.g_best_pos - prev_position[i])
                    new_position = self.g_best_pos - A[i] * D
                else:
                    x_rand_pos = np.random.uniform(
                        self.domain_range[0],
                        self.domain_range[1],
                        (1, self.problem_size),
                    )
                    D = np.abs(C[i] * x_rand_pos - prev_position[i])
                    new_position = x_rand_pos - A[i] * D
            else:
                D1 = np.abs(self.g_best_pos - prev_position[i])
                new_position = (
                    D1 * np.exp(b * lotate[i]) * np.cos(2 * np.pi * lotate[i])
                    + self.g_best_pos
                )
            new_position_list.append(new_position)
        return new_position_list
