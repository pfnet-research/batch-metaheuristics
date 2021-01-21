from copy import deepcopy

import numpy as np
from numpy import array, setxor1d
from numpy.random import choice, rand, randint

from batchopt.optimizers.base_optimizer import BaseOptimizer


class TeachingLearningOptimization(BaseOptimizer):
    @property
    def is_update_improved(self):
        return True

    def generate_new_position(self):
        new_position_list = []
        MEAN = np.mean(self.prev_position, axis=0)
        for i in range(self.pop_size):
            if self.current_epoch % 2 == 0:
                # Teaching Phrase
                TF = randint(1, 3)  # 1 or 2 (never 3)
                arr_random = rand(self.problem_size)
                DIFF_MEAN = arr_random * (self.g_best_pos - TF * MEAN)
                temp = self.prev_position[i] + DIFF_MEAN
            else:
                # Learning Phrase
                temp = deepcopy(self.prev_position[i])
                id_partner = choice(setxor1d(array(range(self.pop_size)), array([i])))
                arr_random = rand(self.problem_size)
                if self.prev_fitness[i] < self.prev_fitness[id_partner]:
                    temp += arr_random * (
                        self.prev_position[i] - self.prev_position[id_partner]
                    )
                else:
                    temp += arr_random * (
                        self.prev_position[id_partner] - self.prev_position[i]
                    )
            new_position_list.append(temp)
        return new_position_list
