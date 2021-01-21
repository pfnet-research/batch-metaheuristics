from copy import deepcopy

import numpy as np
from numpy import abs
from numpy.random import uniform

from batchopt.optimizers.base_optimizer import BaseOptimizer


class GreyWolfOptimizer(BaseOptimizer):
    """
    Standard version of Grey Wolf Optimizer (GWO)
    """

    def __init__(
        self, domain_range, log=True, epoch=750, pop_size=100,
    ):
        super().__init__(domain_range, log, epoch, pop_size)

    @property
    def is_update_improved(self):
        return True

    def generate_new_position(self):
        prev_position = self.prev_position
        a = 2 - 2 * self.current_epoch / (
            self.epoch - 1
        )  # linearly decreased from 2 to 0
        new_position_list = []
        for i in range(self.pop_size):
            A1, A2, A3 = (
                a * (2 * uniform() - 1),
                a * (2 * uniform() - 1),
                a * (2 * uniform() - 1),
            )
            C1, C2, C3 = 2 * uniform(), 2 * uniform(), 2 * uniform()

            X1 = self.top3_pos[0] - A1 * abs(C1 * self.top3_pos[0] - prev_position[i])
            X2 = self.top3_pos[1] - A2 * abs(C2 * self.top3_pos[1] - prev_position[i])
            X3 = self.top3_pos[2] - A3 * abs(C3 * self.top3_pos[2] - prev_position[i])
            temp = (X1 + X2 + X3) / 3.0
            new_position_list.append(deepcopy(temp))
        return new_position_list

    def tell(self, new_position_list, new_fitness):
        if self.current_epoch == -1:
            self.prev_position = new_position_list
            self.prev_fitness = new_fitness
            min_index = np.argmin(new_fitness)
            self.g_best_pos = deepcopy(new_position_list[min_index])
            self.g_best_fit = deepcopy(new_fitness[min_index])
            self.local_best_position = deepcopy(new_position_list)
            self.local_best_fitness = deepcopy(new_fitness)
            self.sort_population_by_fitness()
            self.top3_pos = deepcopy(self.prev_position[:3])
            self.top3_fit = deepcopy(self.prev_fitness[:3])
        else:
            new_position_list, new_fitness = self.update_only_improved_position(
                new_position_list, new_fitness
            )
        sorted_index = np.argsort(new_fitness)
        new_fitness = new_fitness[sorted_index]
        new_position_list = new_position_list[sorted_index]
        cur_top3_pos = deepcopy(new_position_list[:3])
        cur_top3_fit = deepcopy(new_fitness[:3])
        for i in range(3):
            if cur_top3_fit[i] < self.top3_fit[i]:
                self.top3_fit[i] = deepcopy(cur_top3_fit[i])
                self.top3_pos[i] = deepcopy(cur_top3_pos[i])
        self.g_best_pos = self.top3_pos[0]
        self.g_best_fit = self.top3_fit[0]
        self.update_local_best(new_position_list, new_fitness)
        self.prev_position = deepcopy(new_position_list)
        self.prev_fitness = deepcopy(new_fitness)
        self.loss_train.append(self.top3_fit[0])
        self.current_epoch += 1
        self.output_log()
