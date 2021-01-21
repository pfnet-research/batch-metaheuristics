from numpy import array, cumsum, max, min, ones, reshape
from numpy.random import rand

from batchopt.optimizers.base_optimizer import BaseOptimizer


class AntLionOptimizer(BaseOptimizer):
    """
    Ant Lion Optimizer (ALO)
    """

    @property
    def is_update_improved(self):
        return True

    def _random_walk_around_antlion(self, solution, current_epoch):
        # Make the bounded vector
        lb = self.domain_range[0] * ones(self.problem_size)
        ub = self.domain_range[1] * ones(self.problem_size)

        I_ratio = 1  # I_ratio is the ratio in Equations (2.10) and (2.11)
        if current_epoch > self.epoch / 10:
            I_ratio = 1 + 100 * (current_epoch / self.epoch)
        if current_epoch > self.epoch / 2:
            I_ratio = 1 + 1000 * (current_epoch / self.epoch)
        if current_epoch > self.epoch * (3 / 4):
            I_ratio = 1 + 10000 * (current_epoch / self.epoch)
        if current_epoch > self.epoch * 0.9:
            I_ratio = 1 + 100000 * (current_epoch / self.epoch)
        if current_epoch > self.epoch * 0.95:
            I_ratio = 1 + 1000000 * (current_epoch / self.epoch)

        # Dicrease boundaries to converge towards antlion
        lb = lb / I_ratio  # Equation (2.10) in the paper
        ub = ub / I_ratio  # Equation (2.10) in the paper

        # Move the interval of [lb ub] around the antlion [lb+anlion ub+antlion]. Eq 2.8, 2.9
        lb = lb + solution if rand() < 0.5 else -lb + solution
        ub = ub + solution if rand() < 0.5 else -ub + solution

        # This function creates n random walks and normalize according to lb and ub vectors,
        # Using matrix and vector for better performance
        X = array(
            [
                cumsum(2 * (rand(self.epoch, 1) > 0.5) - 1)
                for _ in range(self.problem_size)
            ]
        )
        a = min(X, axis=1)
        b = max(X, axis=1)
        temp1 = reshape((ub - lb) / (b - a), (self.problem_size, 1))
        temp0 = X - reshape(a, (self.problem_size, 1))
        X_norm = temp0 * temp1 + reshape(lb, (self.problem_size, 1))
        return X_norm

    def _roulette_wheel_selection(self, weights):
        # The problem with this function is: it will not working with negative fitness values.
        accumulation = cumsum(weights)
        p = rand() * accumulation[-1]
        chosen_index = -1
        for idx in range(self.pop_size):
            if accumulation[idx] > p:
                chosen_index = idx
                break
        return chosen_index

    def generate_new_position(self):
        self.sort_population_by_fitness()
        new_position_list = []
        for _ in range(self.pop_size):
            # Select ant lions based on their fitness
            # (the better anlion the higher chance of catching ant)
            rolette_index = self._roulette_wheel_selection(1.0 / self.prev_fitness)
            if rolette_index == -1:
                rolette_index = 1

            # RA is the random walk around the selected antlion by rolette wheel
            RA = self._random_walk_around_antlion(
                self.prev_position[rolette_index], self.current_epoch
            )

            # RE is the random walk around the elite (best antlion so far)
            RE = self._random_walk_around_antlion(self.g_best_pos, self.current_epoch)

            temp = (
                RA[:, self.current_epoch] + RE[:, self.current_epoch]
            ) / 2  # Equation(2.13) in the paper
            new_position_list.append(temp)
        return new_position_list
