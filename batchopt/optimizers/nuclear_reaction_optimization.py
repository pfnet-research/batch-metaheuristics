from copy import deepcopy
from math import gamma

import numpy as np
from numpy import log as loge
from scipy.stats import rankdata

from batchopt.optimizers.base_optimizer import BaseOptimizer, RandomRelocator


class NuclearReactionOptimization(BaseOptimizer):
    """
    An Approach Inspired from Nuclear Reaction Processes for Numerical Optimization (NRO)
    """

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

    def nfi_phase(self, prev_position, prev_fitness):
        new_position_list = []
        xichma_v = 1
        xichma_u = (
            (gamma(1 + 1.5) * np.sin(np.pi * 1.5 / 2))
            / (gamma((1 + 1.5) / 2) * 1.5 * 2 ** ((1.5 - 1) / 2))
        ) ** (1.0 / 1.5)
        self.levy_b = (np.random.normal(0, xichma_u ** 2)) / (
            np.sqrt(np.abs(np.random.normal(0, xichma_v ** 2))) ** (1.0 / 1.5)
        )

        # NFi phase
        Pb = np.random.uniform()
        Pfi = np.random.uniform()
        self.alpha = 0.01
        for i in range(self.pop_size):
            # Calculate neutron vector Nei by Eq. (2)
            # Random 1 more index to select neutron
            temp1 = list(set(range(self.pop_size)) - {i})
            i1 = np.random.choice(temp1, replace=False)
            Nei = (prev_position[i] + prev_position[i1]) / 2
            # Update population of fission products according to Eq.(3), (6) or (9);
            if np.random.uniform() <= Pfi:
                # Update based on Eq. 3
                if np.random.uniform() <= Pb:
                    xichma1 = (
                        loge(self.current_epoch + 1) * 1.0 / (self.current_epoch + 1)
                    ) * np.abs(np.subtract(prev_position[i], self.g_best_pos))
                    gauss = np.array(
                        [
                            np.random.normal(self.g_best_pos[j], xichma1[j])
                            for j in range(self.problem_size)
                        ]
                    )
                    Xi = (
                        gauss
                        + np.random.uniform() * self.g_best_pos
                        - np.round(np.random.rand() + 1) * Nei
                    )
                # Update based on Eq. 6
                else:
                    i2 = np.random.choice(temp1, replace=False)
                    xichma2 = (
                        loge(self.current_epoch + 1) * 1.0 / (self.current_epoch + 1)
                    ) * np.abs(np.subtract(prev_position[i2], self.g_best_pos))
                    gauss = np.array(
                        [
                            np.random.normal(prev_position[i][j], xichma2[j])
                            for j in range(self.problem_size)
                        ]
                    )
                    Xi = (
                        gauss
                        + np.random.uniform() * self.g_best_pos
                        - np.round(np.random.rand() + 2) * Nei
                    )
            # Update based on Eq. 9
            else:
                i3 = np.random.choice(temp1, replace=False)
                xichma2 = (
                    loge(self.current_epoch + 1) * 1.0 / (self.current_epoch + 1)
                ) * np.abs(np.subtract(prev_position[i3], self.g_best_pos))
                Xi = np.array(
                    [
                        np.random.normal(prev_position[i][j], xichma2[j])
                        for j in range(self.problem_size)
                    ]
                )
            new_position_list.append(Xi)
        return new_position_list

    def nfu_phase(self, prev_position, prev_fitness):
        new_position_list = []
        # NFu phase
        # Ionization stage
        # Calculate the Pa through Eq. (10);
        ranked_pop = rankdata([prev_fitness[i] for i in range(self.pop_size)])
        for i in range(self.pop_size):
            X_ion = deepcopy(prev_position[i])
            if (ranked_pop[i] * 1.0 / self.pop_size) < np.random.uniform():
                i1, i2 = np.random.choice(
                    list(set(range(self.pop_size)) - {i}), 2, replace=False
                )
                for j in range(self.problem_size):
                    # Levy flight strategy is described as Eq. 18
                    if prev_position[i2][j] == prev_position[i][j]:
                        X_ion[j] = prev_position[i][j] + self.alpha * self.levy_b * (
                            prev_position[i][j] - self.g_best_pos[j]
                        )
                    # If not, based on Eq. 11, 12
                    else:
                        if np.random.uniform() <= 0.5:
                            X_ion[j] = prev_position[i1][j] + np.random.uniform() * (
                                prev_position[i2][j] - prev_position[i][j]
                            )
                        else:
                            X_ion[j] = prev_position[i1][j] - np.random.uniform() * (
                                prev_position[i2][j] - prev_position[i][j]
                            )

            else:  # Levy flight strategy is described as Eq. 21
                max_index = np.argmax(prev_fitness)
                X_worst = deepcopy(prev_position[max_index])
                for j in range(self.problem_size):
                    # Based on Eq. 21
                    if X_worst[j] == self.g_best_pos[j]:
                        X_ion[j] = prev_position[i][j] + self.alpha * self.levy_b * (
                            self.domain_range[1] - self.domain_range[0]
                        )
                    # Based on Eq. 13
                    else:
                        X_ion[j] = prev_position[i][j] + np.round(
                            np.random.uniform()
                        ) * np.random.uniform() * (X_worst[j] - self.g_best_pos[j])
            new_position_list.append(X_ion)

        return new_position_list

    def calculate_pc(self, prev_position, prev_fitness):
        new_position_list = []
        freq = 0.05
        # all ions obtained from ionization are ranked based on (14)
        # - Calculate the Pc through Eq. (14)
        ranked_pop = rankdata([prev_fitness[i] for i in range(self.pop_size)])
        for i in range(self.pop_size):
            i1, i2 = np.random.choice(
                list(set(range(self.pop_size)) - {i}), 2, replace=False
            )

            # Generate fusion nucleus
            if (ranked_pop[i] * 1.0 / self.pop_size) < np.random.uniform():
                t1 = np.random.uniform() * (prev_position[i1] - self.g_best_pos)
                t2 = np.random.uniform() * (prev_position[i2] - self.g_best_pos)
                temp2 = prev_position[i1] - prev_position[i2]
                X_fu = (
                    prev_position[i] + t1 + t2 - np.exp(-np.linalg.norm(temp2)) * temp2
                )
            # Else
            else:
                # Based on Eq. 22
                check_equal = prev_position[i1] == prev_position[i2]
                if check_equal.all():
                    X_fu = prev_position[i] + self.alpha * self.levy_b * (
                        prev_position[i] - self.g_best_pos
                    )
                # Based on Eq. 16, 17
                else:
                    if np.random.uniform() > 0.5:
                        X_fu = prev_position[i] - 0.5 * (
                            np.sin(2 * np.pi * freq * self.current_epoch + np.pi)
                            * (self.epoch - self.current_epoch)
                            / self.epoch
                            + 1
                        ) * (prev_position[i1] - prev_position[i2])
                    else:
                        X_fu = prev_position[i] - 0.5 * (
                            np.sin(2 * np.pi * freq * self.current_epoch + np.pi)
                            * self.current_epoch
                            / self.epoch
                            + 1
                        ) * (prev_position[i1] - prev_position[i2])
            new_position_list.append(X_fu)

        return new_position_list

    def generate_new_position(self):
        prev_position = self.prev_position
        prev_fitness = self.prev_fitness
        new_position_list = []
        # freq = 0.05
        if self.current_epoch % 3 == 0:
            new_position_list = self.nfi_phase(prev_position, prev_fitness)
        elif self.current_epoch % 3 == 1:
            new_position_list = self.nfu_phase(prev_position, prev_fitness)
        else:
            new_position_list = self.calculate_pc(prev_position, prev_fitness)
        return new_position_list
