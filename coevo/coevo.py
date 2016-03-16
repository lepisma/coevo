# -*- coding: utf-8 -*-
"""
Main coevolution module
"""

import numpy as np
import selection
import operators


class CoEvoMO(object):
    """
    Coevolutionary Multi Objective GA
    """

    def __init__(self, n_pop, n_var, fitness_functions, lb=0, ub=1):
        """
        Initialize problem

        Parameters
        ----------
        n_pop : int
            Population size
        n_var : int
            Number of variables in problem
        fitness_functions : list of functions
            Objective functions (to maximize)
        lb : number or list
            Lower bound for variables
        ub : number or list
            Upper bound for variables
        """

        self.n_pop = n_pop
        self.n_var = n_var
        self.n_obj = len(fitness_functions)
        self.lb = lb
        self.ub = ub

        if not ((type(self.lb) == list) or (type(self.lb) == np.ndarray)):
            self.lb = np.ones(self.n_var) * self.lb
        if not ((type(self.ub) == list) or (type(self.ub) == np.ndarray)):
            self.ub = np.ones(self.n_var) * self.ub

        # Generate population
        self.population = np.ones((self.n_pop, self.n_var)) * self.lb + \
                          np.random.rand(self.n_pop, self.n_var) * \
                          (self.ub - self.lb)

        self.fitness_functions = fitness_functions
        self.fitness = np.full((self.n_pop, self.n_obj), np.NaN)

        # Generate initial cost
        self.get_cost(self.fitness_functions, xrange(self.n_pop))

    def get_cost(self, fitness_functions, update_indices):
        """
        Calculate fitness
        """

        # Calculate fitness for not given indices
        for idx in update_indices:
            self.fitness[idx] = np.array([f(self.population[idx]) for \
                                          f in fitness_functions])

    def evolve(self,
               cross_rate,
               mut_rate,
               cross_fraction=1.0,
               elite=0,
               fitness_functions=None):
        """
        Evolve the population, one step at a time
        """

        # Crossover parents per objective
        n_cross_obj = int(self.n_pop * cross_fraction / self.n_obj)

        update_indices = []

        # Sampled indices for each all objectives
        indices = []
        elites = []

        for i in xrange(self.n_obj):
            ps, els = selection.roulette(self.fitness[:, i],
                                         n_cross_obj,
                                         elite=elite)
            indices += ps
            elites += els

        np.random.shuffle(indices)
        # Unique elites
        elites = list(set(elites))

        # Make number of parents even
        if len(indices) % 2 != 0:
            indices.pop()

        # Next generation
        new_population = np.copy(self.population)

        cross_count = 0
        # Do crossover
        for idx1, idx2 in zip(indices[::2], indices[1::2]):
            if np.random.rand() < cross_rate:
                kid1, kid2 = operators.cross_blend(self.population[idx1],
                                                   self.population[idx2], 0.5,
                                                   self.lb, self.ub)
                new_population[idx1] = kid1
                new_population[idx2] = kid2
                update_indices += [idx1, idx2]

                cross_count += 1

        mut_count = 0
        # Mutate
        for idx in xrange(self.n_pop):
            if np.random.rand() < mut_rate:
                new_population[idx] = operators.mutate_uniform(
                    self.population[idx], self.lb, self.ub)

                update_indices += [idx]

                mut_count += 1

                # Restore elites
        new_population[elites] = self.population[elites]
        update_indices = set(update_indices)
        [update_indices.remove(el) for el in elites if el in update_indices]
        update_indices = list(update_indices)

        # Copy over generation
        self.population = new_population

        # Get fitness
        if fitness_functions is not None:
            # Allow new functions too
            self.get_cost(fitness_functions, update_indices)
        else:
            self.get_cost(self.fitness_functions, update_indices)
