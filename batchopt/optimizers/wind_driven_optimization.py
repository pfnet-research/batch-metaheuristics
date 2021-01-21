from numpy import clip, ones
from numpy.random import randint, uniform

from batchopt.optimizers.base_optimizer import BaseOptimizer


class WindDrivenOptimization(BaseOptimizer):
    """
    Basic: Wind Driven Optimization (WDO)
        The Wind Driven Optimization Technique and its Application in Electromagnetics
    """

    def __init__(
        self,
        domain_range,
        log=True,
        epoch=750,
        pop_size=100,
        RT=3,
        g=0.2,
        alp=0.4,
        c=0.4,
        max_v=0.3,
    ):
        super().__init__(domain_range, log, epoch, pop_size)
        self.RT = RT  # RT coefficient
        self.g = g  # gravitational constant
        self.alp = alp  # constants in the update equation
        self.c = c  # coriolis effect
        self.max_v = max_v  # maximum allowed speed
        self.velocity_list = self.max_v * uniform(
            self.domain_range[0],
            self.domain_range[1],
            (self.pop_size, self.problem_size),
        )

    @property
    def is_update_improved(self):
        return False

    def generate_new_position(self):
        new_position_list = []
        # Update velocity based on random dimensions and position of global best
        for i in range(self.pop_size):
            rand_dim = randint(0, self.problem_size)
            temp = self.velocity_list[i][rand_dim] * ones(self.problem_size)
            vel = (
                (1 - self.alp) * self.velocity_list[i]
                - self.g * self.prev_position[i]
                + (1 - 1.0 / (i + 1))
                * self.RT
                * (self.g_best_pos - self.prev_position[i])
                + self.c * temp / (i + 1)
            )
            vel = clip(vel, -self.max_v, self.max_v)

            # Update air parcel positions, check the bound and calculate pressure (fitness)
            pos = self.prev_position[i] + vel
            self.velocity_list[i] = vel
            new_position_list.append(pos)
        return new_position_list
