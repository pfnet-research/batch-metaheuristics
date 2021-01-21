from copy import deepcopy

import numpy as np
from numpy import abs, cos, exp, pi
from numpy.random import uniform

from batchopt.optimizers.base_optimizer import BaseOptimizer, Result


class MothFlameOptimization(BaseOptimizer):
    """
    Moth-flame optimization (MFO)
    """

    def __init__(
        self, domain_range, log=True, epoch=750, pop_size=100,
    ):
        super().__init__(domain_range, log, epoch, pop_size)
        self.epoch = epoch
        self.pop_size = pop_size

    @property
    def is_update_improved(self):
        raise NotImplementedError

    def generate_new_position(self):
        raise NotImplementedError

    def optimize(self, objective_func):
        position = np.random.uniform(
            self.domain_range[0],
            self.domain_range[1],
            (self.pop_size, self.problem_size),
        )
        fitness = objective_func(position)
        sorted_index = np.argsort(fitness)
        flame_fitness = deepcopy(fitness)[sorted_index]
        flame_position = deepcopy(position)[sorted_index]
        g_best_pos = deepcopy(position[0])
        g_best_fit = deepcopy(fitness[0])

        for epoch in range(self.epoch):
            # Number of flames Eq.(3.14) in the paper (linearly decreased)
            num_flame = round(
                self.pop_size - (epoch + 1) * ((self.pop_size - 1) / self.epoch)
            )

            # a linearly decreases from -1 to -2 to calculate t in Eq. (3.12)
            a = -1 + (epoch + 1) * ((-1) / self.epoch)
            new_position_list = []
            for i in range(self.pop_size):

                temp = deepcopy(position[i])
                for j in range(self.problem_size):
                    #   D in Eq.(3.13)
                    distance_to_flame = abs(flame_position[i][j] - position[i][j])
                    t = (a - 1) * uniform() + 1
                    b = 1
                    if (
                        i <= num_flame
                    ):  # Update the position of the moth with respect to
                        # its corresponding flame
                        # Eq.(3.12)
                        temp[j] = (
                            distance_to_flame * exp(b * t) * cos(t * 2 * pi)
                            + flame_position[i][j]
                        )
                    else:  # Update the position of the moth with respect to one flame
                        # Eq.(3.12).
                        # Here is a changed, I used the best solution of flames
                        # not the solution num_flame th (as original code)
                        temp[j] = (
                            distance_to_flame * exp(b * t) * cos(t * 2 * pi)
                            + g_best_pos[j]
                        )
                new_position_list.append(temp)
            new_position_list = np.vstack(new_position_list)
            new_fitness = objective_func(new_position_list)
            for i in range(self.pop_size):
                if fitness[i] < new_fitness[i]:
                    new_fitness[i] = fitness[i]
                    new_position_list[i] = position[i]
            min_index = np.argmin(new_fitness)
            cur_best_fit = new_fitness[min_index]
            if cur_best_fit < g_best_fit:
                g_best_pos = new_position_list[min_index]
                g_best_fit = deepcopy(cur_best_fit)
            position = deepcopy(new_position_list)
            fitness = deepcopy(new_fitness)
            flame_position = np.concatenate([flame_position, new_position_list], axis=0)
            flame_fitness = np.concatenate([flame_fitness, new_fitness], axis=0)
            sorted_index = np.argsort(flame_fitness)
            flame_fitness = flame_fitness[sorted_index]
            flame_position = flame_position[sorted_index]
            flame_fitness = flame_fitness[: self.pop_size]
            flame_position = flame_position[: self.pop_size]
            self.loss_train.append(g_best_fit)
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best_fit))

        return Result(g_best_pos, g_best_fit, self.loss_train)
