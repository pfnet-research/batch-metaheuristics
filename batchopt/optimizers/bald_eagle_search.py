import numpy as np

from batchopt.optimizers.base_optimizer import BaseOptimizer


class BaldEagleSearch(BaseOptimizer):
    """
    Original version of: Bald Eagle Search (BES)
        (Novel meta-heuristic bald eagle search optimisation algorithm)
    Link:
        DOI: https://doi.org/10.1007/s10462-019-09732-5
    """

    def __init__(
        self,
        domain_range,
        log=True,
        epoch=750,
        pop_size=100,
        a=10,
        R=1.5,
        alpha=2,
        c1=2,
        c2=2,
    ):
        super().__init__(domain_range, log, epoch, pop_size)
        # default: 10, determining the corner between point search in the central point,
        # in [5, 10]
        self.a = a
        # default: 1.5, determining the number of search cycles, in [0.5, 2]
        self.R = R
        # default: 2, parameter for controlling the changes in position, in [1.5, 2]
        self.alpha = alpha
        # default: 2, in [1, 2]
        self.c1 = c1
        # c1 and c2 increase the movement intensity of bald eagles towards the best and
        # centre points
        self.c2 = c2

    @property
    def is_update_improved(self):
        return True

    def _create_x_and_y(self):
        # Eq. 2
        phi = self.a * np.pi * np.random.uniform()
        r = phi + self.R * np.random.uniform()
        xr, yr = r * np.sin(phi), r * np.cos(phi)

        # Eq. 3
        r1 = phi1 = self.a * np.pi * np.random.uniform()
        xr1, yr1 = r1 * np.sinh(phi1), r1 * np.cosh(phi1)
        return np.array([xr, yr, xr1, yr1])

    def generate_new_position(self):
        prev_position = self.prev_position

        # Three parts: selecting the search space, searching within the selected search
        # space and swooping.
        solution_list = np.array(prev_position)
        solution_mean = np.mean(solution_list, axis=0)
        new_position_list = []

        # 1. Select space
        if self.current_epoch % 3 == 0:
            xy_list = np.array([self._create_x_and_y() for _ in range(self.pop_size)]).T
            self.x_list = xy_list[0] / np.max(xy_list[0])
            self.y_list = xy_list[1] / np.max(xy_list[1])
            self.x1_list = xy_list[2] / np.max(xy_list[2])
            self.y1_list = xy_list[3] / np.max(xy_list[3])
            for i in range(self.pop_size):
                temp = self.g_best_pos + self.alpha * np.random.uniform() * (
                    solution_mean - prev_position[i]
                )
                new_position_list.append(temp)

        # 2. Search in space
        if self.current_epoch % 3 == 1:
            for i in range(self.pop_size):
                solution_i1 = prev_position[np.random.choice(range(self.pop_size))]
                temp = (
                    prev_position[i]
                    + self.y_list[i] * (prev_position[i] - solution_i1)
                    + self.x_list[i] * (prev_position[i] - solution_mean)
                )
                new_position_list.append(temp)

        # 3. Swoop
        # Originally, global best should be updated every time.
        # But here, use previous global best for batch process and reduce the accuracy.
        if self.current_epoch % 3 == 2:
            for i in range(self.pop_size):
                temp = (
                    np.random.uniform() * self.g_best_pos
                    + self.x1_list[i] * (prev_position[i] - self.c1 * solution_mean)
                    + self.y1_list[i] * (prev_position[i] - self.c2 * self.g_best_pos)
                )
                new_position_list.append(temp)
        return new_position_list
