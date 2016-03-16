# -*- coding: utf-8 -*-
"""
Population selection and ranking functions
"""

import numpy as np


def roulette(fitness_values, return_size, elite=0):
    """
    Perform a roulette wheel selection
    Return return_size item indices
    """

    sorted_indices = np.argsort(fitness_values)
    c_sorted = np.sort(fitness_values).cumsum()
    c_sorted /= np.max(c_sorted)

    sampled = [sorted_indices[np.sum(np.random.rand() > c_sorted)] for _ in \
               xrange(return_size)]
    elites = sorted_indices[::-1][:elite].tolist()

    return sampled, elites


def tournament(fitness_values, return_size, tournament_size, elite=0):
    """
    Perform a deterministic tournament selection
    """

    sampled = []
    for i in xrange(return_size):
        items = np.random.randint(0, len(fitness_values), tournament_size)
        sampled.append(items[np.argmax(fitness_values[items])])

    sorted_indices = np.argsort(fitness_values)
    elites = sorted_indices[::-1][:elite].tolist()

    return sampled, elites
