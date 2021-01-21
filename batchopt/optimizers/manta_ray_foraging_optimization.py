#!/usr/bin/env python

from copy import deepcopy

import numpy as np

from batchopt.optimizers.base_optimizer import BaseOptimizer


class MantaRayForagingOptimization(BaseOptimizer):
    """
    Manta Ray Foraging Optimization (MRFO)
    """

    def __init__(
        self, domain_range, log=True, epoch=750, pop_size=100, S=2,
    ):
        super().__init__(domain_range, log, epoch, pop_size)
        self.S = S  # somersault factor that decides the somersault range of manta rays

    @property
    def is_update_improved(self):
        return self.current_epoch % 2 == 1

    def next_move(
        self,
        position=None,
        fitness=None,
        g_best_pos=None,
        g_best_fit=None,
        epoch=None,
        i=None,
    ):
        position = deepcopy(position)
        fitness = deepcopy(fitness)
        g_best_fit = deepcopy(g_best_fit)
        g_best_pos = deepcopy(g_best_pos)
        if np.random.uniform() < 0.5:  # Cyclone foraging (Eq. 5, 6, 7)
            r1 = np.random.uniform()
            beta = (
                2
                * np.exp(r1 * (self.epoch - epoch) / self.epoch)
                * np.sin(2 * np.pi * r1)
            )

            if (epoch + 1) / self.epoch < np.random.uniform():
                x_rand = np.random.uniform(
                    self.domain_range[0], self.domain_range[1], self.problem_size
                )  # vector
                if i == 0:
                    x_t1 = (
                        x_rand
                        + np.random.uniform() * (x_rand - position[i])
                        + beta * (x_rand - position[i])
                    )
                else:
                    x_t1 = (
                        x_rand
                        + np.random.uniform() * (position[i - 1] - position[i])
                        + beta * (x_rand - position[i])
                    )
            else:
                if i == 0:
                    x_t1 = (
                        g_best_pos
                        + np.random.uniform() * (g_best_pos - position[i])
                        + beta * (g_best_pos - position[i])
                    )
                else:
                    x_t1 = (
                        g_best_pos
                        + np.random.uniform() * (position[i - 1] - position[i])
                        + beta * (g_best_pos - position[i])
                    )

        else:  # Chain foraging (Eq. 1,2)
            r = np.random.uniform()
            alpha = 2 * r * np.sqrt(np.abs(np.log(r)))
            if i == 0:
                x_t1 = (
                    position[i]
                    + r * (g_best_pos - position[i])
                    + alpha * (g_best_pos - position[i])
                )
            else:
                x_t1 = (
                    position[i]
                    + r * (position[i - 1] - position[i])
                    + alpha * (g_best_pos - position[i])
                )
        return x_t1

    def generate_new_position(self):
        prev_position = self.prev_position
        prev_fitness = self.prev_fitness
        new_position_list = []
        if self.current_epoch % 2 != 0:
            for i in range(self.pop_size):
                x_t1 = self.next_move(
                    prev_position,
                    prev_fitness,
                    self.g_best_pos,
                    self.g_best_fit,
                    self.current_epoch,
                    i,
                )
                new_position_list.append(x_t1)
        else:
            # Somersault foraging   (Eq. 8)
            for i in range(self.pop_size):
                x_t1 = prev_position[i] + self.S * (
                    np.random.uniform() * self.g_best_pos
                    - np.random.uniform() * prev_position[i]
                )
                new_position_list.append(x_t1)
        return new_position_list
