import numpy as np

from batchopt.optimizers.base_optimizer import BaseOptimizer, RandomRelocator


class SeaLionOptimization(BaseOptimizer):
    """
    Sea Lion Optimization Algorithm
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
        return False

    def generate_new_position(self):
        prev_position = self.prev_position
        new_position_list = []
        for i in range(self.pop_size):
            c = 2 - 2 * self.current_epoch / self.epoch
            SP_leader = np.random.uniform(0, 1)
            if SP_leader >= 0.6:
                new_pos = (
                    np.cos(2 * np.pi * np.random.uniform(-1, 1))
                    * np.abs(self.g_best_pos - prev_position[i])
                    + self.g_best_pos
                )
            else:
                b = np.random.uniform(0, 1, self.problem_size)
                if c <= 1:
                    new_pos = self.g_best_pos - c * b * np.abs(
                        2 * self.g_best_pos - prev_position[i]
                    )
                else:
                    rand_index = np.random.randint(0, self.pop_size)
                    rand_SL = prev_position[rand_index]
                    new_pos = rand_SL - c * np.abs(b * rand_SL - prev_position[i])
            new_position_list.append(new_pos)
        return new_position_list
