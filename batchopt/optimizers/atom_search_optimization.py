#!/usr/bin/env python
import numpy as np
from numpy import linalg

from batchopt.optimizers.base_optimizer import BaseOptimizer, RandomRelocator


class AtomSearchOptimization(BaseOptimizer):
    """
    Original: Atom Search Optimization (WDO)
    """

    def __init__(
        self,
        domain_range,
        log=True,
        epoch=750,
        pop_size=100,
        alpha=50,
        beta=0.2,
        relocator=RandomRelocator,
    ):
        super().__init__(domain_range, log, epoch, pop_size, relocator)
        self.alpha = alpha  # Depth weight
        self.beta = beta  # Multiplier weight
        self.loss_train = []

        self.velocity_list = np.random.uniform(
            self.domain_range[0],
            self.domain_range[1],
            (self.pop_size, self.problem_size),
        )
        self._atom_acc_list = None

    @property
    def is_update_improved(self):
        return True

    @property
    def atom_acc_list(self):
        if self._atom_acc_list is None:
            mass_list = np.zeros(self.pop_size)

            # Calculate acceleration.
            self._atom_acc_list = self._acceleration(
                self.prev_position,
                self.prev_fitness,
                mass_list,
                self.g_best_pos,
                iteration=0,
            )
        return self._atom_acc_list

    def _update_mass(self, fitness, mass_list):
        best_fit = np.min(fitness)
        worst_fit = np.max(fitness)
        sum_fit = np.sum(fitness)
        mass_list = np.exp((fitness - best_fit) / (worst_fit - best_fit)) / sum_fit
        return mass_list

    def _find_LJ_potential(self, iteration, average_dist, radius):
        c = (1 - iteration / self.epoch) ** 3
        # g0 = 1.1, u = 2.4
        rsmin = 1.1 + 0.1 * np.sin((iteration + 1) / self.epoch * np.pi / 2)
        rsmax = 1.24
        radius = np.where(radius / average_dist < rsmin, rsmin, radius)
        radius = np.where(radius / average_dist > rsmax, rsmax, radius)
        potential = c * (12 * (-radius) ** (-13) - 6 * (-radius) ** (-7))
        return potential

    def _acceleration(self, position, fitness, mass_list, g_best_pos, iteration):
        eps = 2 ** (-52)
        mass_list = self._update_mass(fitness, mass_list)

        G = np.exp(-20.0 * (iteration + 1) / self.epoch)
        k_best = (
            int(
                self.pop_size
                - (self.pop_size - 2) * ((iteration + 1) / self.epoch) ** 0.5
            )
            + 1
        )
        k_best_index = np.argsort(-mass_list)[:k_best]
        k_best_pos = position[k_best_index]
        mk_average = np.mean(np.array(k_best_pos))

        acc_list = np.zeros((self.pop_size, self.problem_size))
        dist_average = linalg.norm(position - mk_average, axis=1)
        dist_average = np.reshape(dist_average, (self.pop_size, 1))
        norm_pos = np.expand_dims(position, axis=-1)
        norm_pos = np.broadcast_to(norm_pos, (self.pop_size, self.problem_size, k_best))
        norm_pos = np.transpose(norm_pos, (0, 2, 1))
        radius = linalg.norm(norm_pos - k_best_pos, axis=2)
        potential = self._find_LJ_potential(iteration, dist_average, radius)
        temp_potential = np.reshape(potential, (self.pop_size, k_best, 1))
        for i in range(self.pop_size):
            temp = (
                np.broadcast_to(temp_potential[i], k_best_pos.shape)
                * (k_best_pos - position[i])
                / np.broadcast_to((radius[i] + eps)[:, np.newaxis], k_best_pos.shape)
            )
            temp *= np.random.uniform(0, 1, temp.shape)
            temp = np.sum(temp, axis=0)
            temp = self.alpha * temp + self.beta * (g_best_pos - position[i])
            # calculate acceleration
            acc = G * temp / mass_list[i]
            acc_list[i] = acc
        return acc_list

    def generate_new_position(self):
        prev_position = self.prev_position
        # Update velocity based on random dimensions and position of global best
        velocity_rand = np.random.uniform(
            self.domain_range[0],
            self.domain_range[1],
            (self.pop_size, self.problem_size),
        )
        velocity = velocity_rand * self.velocity_list + self.atom_acc_list
        temp = prev_position + velocity
        return temp
