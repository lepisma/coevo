# -*- coding: utf-8 -*-
"""
Operators on population
"""

import numpy as np


def mutate_uniform(individual, lb, ub):
    """
    Mutate one variable from the individual
    Uses uniform random within the range
    """

    indi = np.copy(individual)
    mut_idx = np.random.choice(len(indi))
    indi[mut_idx] = np.random.uniform(low=lb[mut_idx],
                                          high=ub[mut_idx])

    return indi

def cross_blend(individual1, individual2, alpha, lb, ub):
    """
    Alpha blending
    """

    indi1 = np.copy(individual1)
    indi2 = np.copy(individual2)
    for i, (var_i1, var_i2) in enumerate(zip(indi1, indi2)):
        gamma = (1 + 2.0 * alpha) * np.random.rand() - alpha
        indi1[i] = (1 - gamma) * var_i1 + gamma * var_i2
        indi2[i] = gamma * var_i1 + (1 - gamma) * var_i2

    # Clip
    indi1 = np.clip(indi1, lb, ub)
    indi2 = np.clip(indi2, lb, ub)

    return indi1, indi2
