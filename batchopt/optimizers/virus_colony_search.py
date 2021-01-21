import numpy as np
from numpy import abs, array, log1p, prod, sum
from numpy.random import choice, normal, uniform

from batchopt.optimizers.base_optimizer import BaseOptimizer, RandomRelocator


class VirusColonySearch(BaseOptimizer):
    """
        Virus Colony Search (VCS)
    """

    def __init__(
        self,
        domain_range,
        log=True,
        epoch=750,
        pop_size=100,
        relocator=RandomRelocator,
        lamda=0.5,
        xichma=0.3,
    ):
        super().__init__(domain_range, log, epoch, pop_size, relocator)
        self.xichma = xichma  # Weight factor
        self.lamda = lamda  # Number of the best will keep
        if lamda < 1:
            self.n_best = int(lamda * self.pop_size)
        else:
            self.n_best = int(lamda)

    @property
    def is_update_improved(self):
        return True

    def optional_initialization(self):
        self.sort_population_by_fitness()
        self.x_mean = np.mean(self.prev_position, axis=0)

    def diffusion(self, prev_position):
        new_position_list = []
        for i in range(self.pop_size):
            xichma = (log1p(self.current_epoch + 1) / self.epoch) * (
                prev_position[i] - self.g_best_pos
            )
            gauss = array(
                [
                    normal(self.g_best_pos[idx], abs(xichma[idx]))
                    for idx in range(self.problem_size)
                ]
            )
            pos_new = gauss + uniform() * self.g_best_pos - uniform() * prev_position[i]
            new_position_list.append(pos_new)
        return new_position_list

    def host_cells_infection(self):
        new_position_list = []
        # Host cells infection
        xichma = self.xichma * (1 - (self.current_epoch + 1) / self.epoch)
        for _ in range(self.pop_size):
            pos_new = self.x_mean + xichma * normal(
                0, 1, self.problem_size
            )  # Basic / simple version, not the original version in the paper
            new_position_list.append(pos_new)
        return new_position_list

    def immune_response(self, prev_position, prev_fitness):
        new_position_list = []
        # Calculate the weighted mean of the λ best individuals by
        sorted_index = np.argsort(prev_fitness)
        prev_fitness = prev_fitness[sorted_index]
        position = prev_position[sorted_index]
        pos_list = position[: self.n_best]

        factor_down = self.n_best * log1p(self.n_best + 1) - log1p(
            prod(range(1, self.n_best + 1))
        )
        weight = log1p(self.n_best + 1) / factor_down
        weight = weight / self.n_best
        self.x_mean = weight * sum(pos_list, axis=0)

        # Immune response
        for i in range(self.pop_size):
            pr = (self.problem_size - i + 1) / self.problem_size
            pos_new = position[i]
            for j in range(self.problem_size):
                if uniform() > pr:
                    id1, id2 = choice(
                        list(set(range(self.pop_size)) - {i}), 2, replace=False,
                    )
                    pos_new[j] = (
                        position[id1][j]
                        - (position[id2][j] - position[i][j]) * uniform()
                    )
            new_position_list.append(pos_new)
        return new_position_list

    def generate_new_position(self):
        # Viruses diffusion
        prev_position = self.prev_position
        prev_fitness = self.prev_fitness
        if self.current_epoch % 3 == 0:
            new_position_list = self.diffusion(prev_position)
        elif self.current_epoch % 3 == 1:
            new_position_list = self.host_cells_infection()
        elif self.current_epoch % 3 == 2:
            new_position_list = self.immune_response(prev_position, prev_fitness)
        return new_position_list
