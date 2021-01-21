from copy import deepcopy

from numpy import array, cumsum, max, reshape
from numpy.random import uniform
from sklearn.preprocessing import normalize

from batchopt.optimizers.base_optimizer import BaseOptimizer


class MultiVerseOptimizer(BaseOptimizer):
    """
    Multi-Verse Optimizer (MVO)
    """

    def __init__(
        self, domain_range, log=True, epoch=750, pop_size=100, wep_minmax=(1.0, 0.2),
    ):
        super().__init__(domain_range, log, epoch, pop_size)
        self.wep_minmax = (
            wep_minmax  # Wormhole Existence Probability (min and max in Eq.(3.3) paper
        )

    @property
    def is_update_improved(self):
        return False

    # sorted_Inflation_rates
    # TODO: This function always (or almost) return 0.
    def _roulette_wheel_selection(self, weights=None):
        accumulation = cumsum(weights)
        p = uniform() * accumulation[-1]
        chosen_idx = -1
        for idx in range(len(accumulation)):
            if accumulation[idx] > p:
                chosen_idx = idx
                break
        return chosen_idx

    def generate_new_position(self):
        # Eq. (3.3) in the paper
        wep = self.wep_minmax[0] + (self.current_epoch + 1) * (
            (self.wep_minmax[1] - self.wep_minmax[0]) / self.epoch
        )
        # Travelling Distance Rate (Formula): Eq. (3.4) in the paper
        tdr = 1 - (self.current_epoch + 1) ** (1.0 / 6) / self.epoch ** (1.0 / 6)

        list_fitness_raw = self.prev_fitness
        maxx = max(list_fitness_raw)
        if maxx > (2 ** 64 - 1):
            list_fitness_normalized = uniform(0, 0.1, self.pop_size)
        else:
            # Normalize inflation rates (NI in Eq. (3.1) in the paper)
            list_fitness_normalized = reshape(
                normalize(array([list_fitness_raw])), self.pop_size
            )  # Matrix
        # Update the position of universes
        new_position_list = []
        elite = deepcopy(self.prev_position[0])
        new_position_list.append(elite)
        for i in range(1, self.pop_size):  # Starting from 1 since 0 is the elite
            black_hole_pos = deepcopy(self.prev_position[i])
            for j in range(self.problem_size):
                r1 = uniform()
                if r1 < list_fitness_normalized[i]:
                    white_hole_id = self._roulette_wheel_selection(
                        (-1 * list_fitness_raw)
                    )
                    if white_hole_id is None or white_hole_id == -1:
                        white_hole_id = 0
                    # Eq. (3.1) in the paper
                    black_hole_pos[j] = self.prev_position[white_hole_id][j]

                # Eq. (3.2) in the paper if the boundaries are all the same
                r2 = uniform()
                if r2 < wep:
                    r3 = uniform()
                    if r3 < 0.5:
                        black_hole_pos[j] = self.g_best_pos[j] + tdr * uniform(
                            self.domain_range[0][j], self.domain_range[1][j]
                        )
                    else:
                        black_hole_pos[j] = self.g_best_pos[j] - tdr * uniform(
                            self.domain_range[0][j], self.domain_range[1][j]
                        )
            new_position_list.append(black_hole_pos)
        return new_position_list
