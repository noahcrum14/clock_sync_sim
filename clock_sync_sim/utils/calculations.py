import numpy as np
import scipy
from clock_sync_sim.config.settings import SIGMA


def get_coin_prob(m, n, mismatch, td):
    return 1 - 1 / (2 ** (m + n - 1)) * sum([
        scipy.special.binom(m, j)
        * scipy.special.binom(n, j)
        * (mismatch ** (2 * j))
        * np.exp(-0.5 * j * SIGMA ** 2 * (td ** 2))
        for j in range(min(m, n) + 1)
    ])