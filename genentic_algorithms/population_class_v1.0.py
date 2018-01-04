# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 13:09:11 2018

@author: K.Ataman
"""

"""
TODO:
    crossover -> diversity_preservation = True case
    add more mutation methods (especially for non bool chromosomes)
"""
import numpy as np

class population:
    def __init__(self, population_size, crossover_rate, mutation_rate,
                 chromosome_matrix):
        """
        inputs:
            chromosome_matrix: a numpy matrix with shape [population_size, chromosome_size]
            that contains the x'th member's y'th chromosome element in chromosome_matrix[x, y]
        """
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.chromosome_matrix = chromosome_matrix
        self.chromosome_length = int(chromosome_matrix.shape[1])
        self.fitness_array = np.zeros(population_size)
        
    def crossover(self, crossover_point = 'random', 
                  selection_method = 'roulette_wheel', 
                  select_with_replacement = True,
                  diversity_preservation = False):
        """
        crosses over a single pair
        
        Crossover_point dictates where the crossover between parent 1 and 2
        will occur
        
        If select_with_replacement is false, then parent 1 and parent 2
        cannot be the same member
        
        diversity_preservation = True
        gives higher score to members that are different to preserve diversity
        """
            
        if diversity_preservation == 'True':
             pass
            
        elif diversity_preservation != 'False':
            raise ValueError ('diversity_preservation is a boolean, set it appropriately')
            
        if selection_method == 'roulette_wheel':
            """P(selection) = fitness/sum(fitness)"""
            probabilities = self.fitness_array / np.sum(self.fitness_array)
            [parent_1, parent_2] = np.random.choice(self.population_size, 2, 
                             select_with_replacement, p = probabilities)
        else:
            raise ValueError ('put a proper selection_method value')
        
        #no cross over
        if parent_1 == parent_2 | self.crossover_rate < np.random.random():
            child_1_chromosome = self.chromosome_matrix[parent_1]
            child_2_chromosome = self.chromosome_matrix[parent_2]
        #cross-over
        else:
            if crossover_point == 'middle':
                crossover_point = self.chromosome_length // 2
            elif crossover_point == 'random':
                crossover_point = np.random.randint(0, self.chromosome_length)
            else:
                raise ValueError ('put a proper crossover_point value')
                
            child_1_chromosome = np.concatenate(
                    (self.chromosome_matrix[parent_1][0:crossover_point], 
                     self.chromosome_matrix[parent_2][crossover_point: self.chromosome_length]))
            child_2_chromosome = np.concatenate(
                    (self.chromosome_matrix[parent_2][0:crossover_point], 
                     self.chromosome_matrix[parent_1][crossover_point: self.chromosome_length]))
                
        return child_1_chromosome, child_2_chromosome
            
    def mutation(chromosome_length, mutation_rate, chromosome, 
                 mutation_type = 'binary'):
        mutate_or_not = np.random.choice([0, 1], chromosome_length, p = [1 - mutation_rate, mutation_rate])
        if mutation_type == 'binary':
            """
            Converts 0 to 1 and 1 to 0 if the element of chromosome needs to be mutated
            """
            for i in range(chromosome_length):
                if mutate_or_not[i] == 1:
                    if chromosome[i] == 0:
                        chromosome[i] = 1 
                    else:
                        chromosome[i] = 0
        else:
            raise ValueError ('put a proper mutation_type value')
            
        return chromosome
            
    def epoch(self):
        """
        Assumes fitness array has been updated properly
        """
        chromosome_matrix = np.zeros([self.population_size, self.chromosome_length])
        for i in range(self.population_size // 2):
            #cross-over
            [child_1, child_2] = self.crossover()
            #mutate
            child_1 = self.mutation(self.chromosome_length, self.mutation_rate, child_1)
            child_2 = self.mutation(self.chromosome_length, self.mutation_rate, child_2)
            #put in chromosome matrix
            chromosome_matrix[i] = child_1
            chromosome_matrix[i + 1] = child_2
        #update self.chromosome_matrix
        self.chromosome_matrix = chromosome_matrix