import numpy as np

from batchopt.optimizers.base_optimizer import BaseOptimizer


# TODO: better name from paper
def get_new_position1(positions, i, problem_size):
    temp = positions[i] + np.random.uniform(0, 1, problem_size) * positions[i]
    return temp


def get_new_position2(positions, i, fl, pop_size):
    FL = np.random.uniform() * 0.4 + fl
    idx = np.random.randint(0.5 * pop_size + 1, pop_size)
    temp = positions[i] + (positions[idx] - positions[i]) * FL
    return temp


class BirdSwarmAlgorithm(BaseOptimizer):
    """
    The original version of: Bird Swarm Algorithm (BSA)
        (A new bio-inspired optimisation algorithm: Bird Swarm Algorithm)
    Link:
        http://doi.org/10.1080/0952813X.2015.1042530
        https://www.mathworks.com/matlabcentral/fileexchange/51256-bird-swarm-algorithm-bsa
    """

    EPSILON = 10e-10

    def __init__(
        self,
        domain_range,
        log=True,
        epoch=750,
        pop_size=100,
        ff=10,
        p=0.8,
        c_couples=(1.5, 1.5),
        a_couples=(1.0, 1.0),
        fl=0.5,
    ):
        super().__init__(
            domain_range=domain_range, log=log, epoch=epoch, pop_size=pop_size,
        )

        self.ff = ff  # flight frequency - default = 10
        self.p = p  # the probability of foraging for food - default = 0.8
        # [c1, c2]: Cognitive accelerated coefficient,
        # Social accelerated coefficient same as PSO
        self.c_minmax = c_couples
        # [a1, a2]: The indirect and direct effect on the birds' vigilance behaviours.
        self.a_minmax = a_couples
        self.fl = fl  # The followed coefficient- default = 0.5

    @property
    def is_update_improved(self):
        return False

    def generate_new_position(self):
        pos_list = self.prev_position
        local_best_pos = self.local_best_position
        local_best_fit = self.local_best_fitness
        # Original implementation use self.ID_LBF but it seems bug.
        # fit_list = array([item[self.ID_FIT] for item in pop])
        fit_list = self.prev_fitness
        pos_mean = np.mean(pos_list, axis=0)
        fit_sum = np.sum(fit_list)
        global_best_pos = self.g_best_pos

        x_new_list = []
        if self.current_epoch % self.ff != 0:
            for i in range(self.pop_size):
                prob = (
                    np.random.uniform() * 0.2 + self.p
                )  # The probability of foraging for food
                if np.random.uniform() < prob:  # Birds forage for food. Eq. 1
                    x_new = (
                        pos_list[i]
                        + self.c_minmax[0]
                        * np.random.uniform()
                        * (local_best_pos[i] - pos_list[i])
                        + self.c_minmax[1]
                        * np.random.uniform()
                        * (global_best_pos - pos_list[i])
                    )
                else:  # Birds keep vigilance. Eq. 2
                    A1 = self.a_minmax[0] * np.exp(
                        -self.pop_size * local_best_fit[i] / (self.EPSILON + fit_sum)
                    )
                    k = np.random.choice(list(set(range(self.pop_size)) - {i}))
                    t1 = (fit_list[i] - fit_list[k]) / (
                        abs(fit_list[i] - fit_list[k]) + self.EPSILON
                    )
                    A2 = self.a_minmax[1] * np.exp(
                        t1 * self.pop_size * fit_list[k] / (fit_sum + self.EPSILON)
                    )
                    x_new = (
                        pos_list[i]
                        + A1 * np.random.uniform(0, 1) * (pos_mean - pos_list[i])
                        + A2
                        * np.random.uniform(-1, 1)
                        * (global_best_pos - pos_list[i])
                    )
                x_new_list.append(x_new)
        else:
            # Divide the bird swarm into two parts: producers and scroungers.
            min_idx = np.argmin(fit_list)
            max_idx = np.argmax(fit_list)
            choose = 0
            if min_idx < 0.5 * self.pop_size and max_idx < 0.5 * self.pop_size:
                choose = 1
            if min_idx > 0.5 * self.pop_size and max_idx < 0.5 * self.pop_size:
                choose = 2
            if min_idx < 0.5 * self.pop_size and max_idx > 0.5 * self.pop_size:
                choose = 3
            if min_idx > 0.5 * self.pop_size and max_idx > 0.5 * self.pop_size:
                choose = 4

            x_new_list = []

            if choose < 3:  # Producing (Equation 5)
                for i in range(int(self.pop_size / 2)):
                    if i == min_idx:
                        temp = get_new_position1(
                            pos_list, i, problem_size=self.problem_size
                        )
                    else:
                        temp = get_new_position2(
                            pos_list, i, fl=self.fl, pop_size=self.pop_size
                        )
                    x_new_list.append(temp)
                    # pop = self._update_position__(pop, i, temp)

                for i in range(int(self.pop_size / 2), self.pop_size):
                    temp = get_new_position1(
                        pos_list, i, problem_size=self.problem_size
                    )
                    x_new_list.append(temp)

            else:  # Scrounging (Equation 6)
                for i in range(int(self.pop_size / 2)):
                    temp = get_new_position1(
                        pos_list, i, problem_size=self.problem_size
                    )
                    x_new_list.append(temp)

                for i in range(int(self.pop_size / 2), self.pop_size):
                    if i == min_idx:
                        temp = get_new_position1(
                            pos_list, i, problem_size=self.problem_size
                        )
                    else:
                        temp = get_new_position2(
                            pos_list, i, fl=self.fl, pop_size=self.pop_size
                        )
                    x_new_list.append(temp)
        return x_new_list
