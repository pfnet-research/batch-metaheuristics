from copy import deepcopy

import numpy as np
from numpy import abs
from numpy.random import normal, randint, uniform

from batchopt.optimizers.base_optimizer import BaseOptimizer


class ArtificialEcosystemBasedOptimization(BaseOptimizer):
    """
    Artificial ecosystem-based optimization
    """

    @property
    def is_update_improved(self):
        return True

    def generate_new_position(self):
        prev_position = self.prev_position
        prev_fitness = self.prev_fitness
        sort_index = np.argsort(prev_fitness)
        prev_fitness = deepcopy(prev_fitness[sort_index])
        prev_position = deepcopy(prev_position[sort_index])
        new_position_list = []
        # Production
        # Eq. 2, 3, 1
        if self.current_epoch % 2 == 0:
            a = (1.0 - self.current_epoch / self.epoch) * uniform()
            x_rand = uniform(
                self.domain_range[0], self.domain_range[1], self.problem_size
            )
            x1 = (1 - a) * prev_position[-1] + a * x_rand
            new_position_list.append(x1)
            # Consumption
            for i in range(1, self.pop_size):
                rand = uniform()
                # Eq. 4, 5, 6
                v1 = normal(0, 1)
                v2 = normal(0, 1)
                c = 0.5 * v1 / abs(v2)  # Consumption factor

                j = randint(0, i)
                # Herbivore
                if rand < 1.0 / 3:
                    x_t1 = prev_position[i] + c * (
                        prev_position[i] - prev_position[0]
                    )  # Eq. 6
                # Omnivore
                elif 1.0 / 3 <= rand and rand <= 2.0 / 3:
                    x_t1 = prev_position[i] + c * (
                        prev_position[i] - prev_position[j]
                    )  # Eq. 7
                # Carnivore
                else:
                    r2 = uniform()
                    x_t1 = prev_position[i] + c * (
                        r2 * (prev_position[i] - prev_position[0])
                        + (1 - r2) * (prev_position[i] - prev_position[j])
                    )
                new_position_list.append(x_t1)
        else:
            # Decomposition
            # Eq. 10, 11, 12, 9
            for i in range(self.pop_size):
                u = normal(0, 1)
                r3 = uniform()
                d = 3 * u
                e = r3 * randint(1, 3) - 1
                h = 2 * r3 - 1
                x_t1 = self.g_best_pos + d * (
                    e * self.g_best_pos - h * prev_position[i]
                )
                new_position_list.append(x_t1)
        return new_position_list
