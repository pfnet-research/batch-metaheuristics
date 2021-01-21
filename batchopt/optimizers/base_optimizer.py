from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np


class OutlierRelocator(ABC):
    def __init__(self, domain_range):
        self.domain_range = domain_range

    @abstractmethod
    def relocate(self, new_position_list):
        pass


class ClipRelocator(OutlierRelocator):
    def relocate(self, new_position_list):
        new_position_list = np.clip(
            new_position_list, self.domain_range[0], self.domain_range[1]
        )
        return new_position_list


class RandomRelocator(OutlierRelocator):
    def relocate(self, new_position_list):
        pop_size, problem_size = new_position_list.shape
        rand_amend = np.random.uniform(0, 1, size=(pop_size, 1))
        slope = self.domain_range[1] - self.domain_range[0]
        intercept = self.domain_range[0]
        rand_amend = np.broadcast_to(rand_amend, (pop_size, problem_size))
        rand_amend = rand_amend * slope + intercept
        new_position_list = np.where(
            np.logical_and(
                self.domain_range[0] <= new_position_list,
                new_position_list <= self.domain_range[1],
            ),
            new_position_list,
            rand_amend,
        )
        return new_position_list


class BaseOptimizer(ABC):
    def __init__(
        self, domain_range, log=False, epoch=750, pop_size=100, relocator=ClipRelocator,
    ):
        self.domain_range = np.asarray(domain_range).T
        self.problem_size = len(domain_range)
        self.pop_size = pop_size
        self.log = log
        self.epoch = epoch
        self.loss_train = []
        self.current_epoch = -1
        self.relocator = relocator(self.domain_range)

    def optional_initialization(self):
        pass

    def sort_population_by_fitness(self):
        sorted_index = np.argsort(self.prev_fitness)
        self.prev_fitness = deepcopy(self.prev_fitness[sorted_index])
        self.prev_position = deepcopy(self.prev_position[sorted_index])

    def generate_initial_position(self):
        return np.random.uniform(
            self.domain_range[0],
            self.domain_range[1],
            (self.pop_size, self.problem_size),
        )

    @property
    @abstractmethod
    def is_update_improved(self):
        pass

    @abstractmethod
    def generate_new_position(self):
        pass

    def ask(self):
        if self.current_epoch == -1:
            new_position_list = self.generate_initial_position()
        else:
            new_position_list = self.generate_new_position()
        new_position_list = np.vstack(new_position_list)
        new_position_list = self.relocator.relocate(new_position_list)
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
            self.optional_initialization()
        else:
            if self.is_update_improved:
                new_position_list, new_fitness = self.update_only_improved_position(
                    new_position_list, new_fitness
                )
            self.g_best_pos, self.g_best_fit = self.update(
                new_position_list, new_fitness
            )
        self.update_local_best(new_position_list, new_fitness)
        self.prev_position = deepcopy(new_position_list)
        self.prev_fitness = deepcopy(new_fitness)
        self.loss_train.append(self.g_best_fit)
        self.current_epoch += 1
        self.output_log()

    def update(self, new_position_list, new_fitness):
        min_index = np.argmin(new_fitness)
        cur_best_fit = deepcopy(new_fitness[min_index])
        if cur_best_fit < self.g_best_fit:
            return deepcopy(new_position_list[min_index]), cur_best_fit
        else:
            return self.g_best_pos, self.g_best_fit

    def update_local_best(self, new_position_list, new_fitness):
        # TODO: remove duplicate code
        for i in range(self.pop_size):
            if self.local_best_fitness[i] > new_fitness[i]:
                self.local_best_fitness[i] = new_fitness[i]
                self.local_best_position[i] = new_position_list[i]

    def update_only_improved_position(self, new_position_list, new_fitness):
        new_position_list = deepcopy(new_position_list)
        new_fitness = deepcopy(new_fitness)
        for i in range(self.pop_size):
            if self.prev_fitness[i] < new_fitness[i]:
                new_position_list[i] = self.prev_position[i]
                new_fitness[i] = self.prev_fitness[i]
        return new_position_list, new_fitness

    def output_log(self):
        if self.log:
            print(
                "> Epoch: {}, Best fit: {}".format(
                    self.current_epoch + 1, self.g_best_fit
                )
            )

    def optimize(self, objective_func):
        for _ in range(self.epoch):
            new_position_list = self.ask()
            new_fitness = objective_func(new_position_list)
            self.tell(new_position_list, new_fitness)
        return Result(self.g_best_pos, self.g_best_fit, self.loss_train)


class Result:
    def __init__(self, g_best_pos, g_best_fit, loss_train):
        self.g_best_pos = g_best_pos
        self.g_best_fit = g_best_fit
        self.loss_train = loss_train
