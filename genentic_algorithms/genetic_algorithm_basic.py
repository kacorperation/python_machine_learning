# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 15:35:40 2017

@author: K.Ataman
"""
import numpy as np
"""
TODO:
    the algorithm converges to a local minima
    crossing method so that it can cross randomly (0.5 chance to cross maybe?)
    niching: punish similar individuals to prevent lack of diversity
        similarity = sum(same genomes)
"""

"""
Idea

Epoch: 
    1. Asses fitness of each chromosome
    2. Select two members, using roulette wheel proportional to fitness
    3. Cross over bits based on crossover rate
    4. Mutate selected members
    5. Repeat 2-4 until certain number of members have been created
    6. The set of changed members becomes the new population
"""

class maze_object:
    def __init__(self, maze, height, width):
        self.height = height
        self.width = width
        self.maze_array = np.reshape(maze, [height, width])
        self.start = [np.where(self.maze_array==5)[0][0], np.where(self.maze_array==5)[1][0]]
        self.finish = [np.where(self.maze_array==8)[0][0], np.where(self.maze_array==8)[1][0]]
        self.position = self.start
        self.steps_taken = 100
    
    def move(self, position, row, column):
        #if we go out of bounds, stay where you are
        if row >= self.height or row < 0 or column >= self.width or column < 0:
            return position
        elif self.maze_array[row, column] == 0 or self.maze_array[row, column] == 8:
            return [row, column]
        else:
            return position
            
    def maze_trial(self, member, num_actions, action_width):
        """does a maze trial for the member, and returns their fitness"""
        position = self.start
        chromosome_index = 0
        num_steps = 0
        finished_flag = 0
        while finished_flag == 0:
            step = ''
            #get the instruction for the current step
            for i in range (action_width):
                step += str(member.chromosome[chromosome_index + i])
            #move or hold in place accordingly
            if step == '00':
                position = self.move(position, position[0] + 1, position[1])
            elif step == '01':
                position = self.move(position, position[0], position[1] + 1)
            elif step == '10':
                position = self.move(position, position[0] - 1, position[1])
            elif step == '11':
                position = self.move(position, position[0], position[1] - 1)
            
            #increment action and num_steps
            chromosome_index += action_width
            num_steps += 1
            if position == self.finish or num_steps >= num_actions:
                finished_flag = 1
                self.steps_taken = num_steps
        self.position = position
        #fitness is a function of distance to exit and steps taken
        distance_to_finish = np.sqrt(
                (self.finish[0] - position[0])**2 + \
                (self.finish[1] - position[1])**2)
        #member.fitness = 1/ (distance_to_finish + num_steps + 1)
        member.fitness = 1/ (2 * distance_to_finish + num_steps)
        
#0 is passable, 1 is wall, 5 is begin, 8 is exit
maze_array =   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1,
                8, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1,
                1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1,
                1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1,
                1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1,
                1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1,
                1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 5,
                1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

#basic member of population with a chromosome
#it is preferred that chromosome size is an even number for even crossover    
class member_object:
    """Properties of a member of population"""
    def __init__(self, num_actions, action_width):
        temp = np.random.randint(0, 2, [num_actions*action_width])
        self.chromosome = ""
        for i in temp:
            self.chromosome += str(i)
        self.fitness = 0
        
class population_object:
    """Properties of the entire population"""
    def __init__(self, population_size, crossover_rate, mutation_rate, num_actions, action_width):
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.num_actions = num_actions
        #at which point does crossover begin?
        self.chromosome_halfway = num_actions * action_width //2
        self.action_width = action_width
        #total fitness of members for a given generation
        self.total_fitness = 0
        self.chromosome_length = num_actions*action_width
        #fitnesses of each member
        self.fitness_list = np.zeros([self.population_size])
        #create population
        self.member_list = []
        for i in range(population_size):
            self.member_list.append(member_object(num_actions, action_width))
    
    def calculate_fitness(self):
        """calculates total fitness and fitness_list, where fitness_list 
        gives the max range for selection for each member
        For example, fitness_list = [0.5, 1] means first member will be
        selected if a random roll is between 0 and 0.5, and the second
        member is selected if the roll is between 0.5 and 1"""
        self.total_fitness = 0
        self.fitness_list = np.zeros([self.population_size])
        for i in range(self.population_size):
            self.fitness_list[i] = self.member_list[i].fitness
            self.total_fitness += self.member_list[i].fitness
        previous_probability = 0
        #normalize
        self.fitness_list = self.fitness_list / self.total_fitness
        #make it so fitness_list contains the upper limits for it's members' ranges
        for fitness_index in range(self.population_size):
            self.fitness_list[fitness_index] += previous_probability
            previous_probability = self.fitness_list[fitness_index]
        
    #does roulette wheel selection to select two members
    def roulette_wheel_selection(self):
        """P(selection) = fitness/sum(fitness)
        Each element gets assigned a range between 0-1.
        random number gets generated, and the range which it lands determines
        which element is chosen
        Returns the indexes of the selected elements
        """
        selection_list = [-2, -1]
        #selection process
        for member_x in self.member_list:
            for i in range(2):
                fitness_index = 0
                number = np.random.random()
                #select a member
                while (number >= self.fitness_list[fitness_index]):
                    fitness_index += 1
                    #take care of the case of a member being selected twice
                    #by resetting the selection of the second guy
                #add member to selection list
                selection_list[i] = fitness_index
        #return the results
        return selection_list
        
    def crossover(self, member_1, member_2):
        """crosses over chromosomes of member 1 and member 2 from a random spot"""
        if np.random.random() < self.crossover_rate and member_1 != member_2 :
            #select a random point to cross over
            cutoff_point = np.random.randint(0, self.chromosome_length)
            temp_1 = member_1.chromosome[0 : cutoff_point]
            temp_2 = member_1.chromosome[cutoff_point : self.chromosome_length]
            temp_3 = member_2.chromosome[0 : cutoff_point]
            temp_4 = member_2.chromosome[cutoff_point : self.chromosome_length]
            baby_1 = member_object(self.num_actions, self.action_width)
            baby_2 = member_object(self.num_actions, self.action_width)
            baby_1.chromosome = temp_1 + temp_4
            baby_2.chromosome = temp_3 + temp_2
            return [baby_1, baby_2]
        #no cross over. Wah-wah
        else:
            baby_1 = member_1
            baby_2 = member_2
            return [baby_1, baby_2]
    
    def mutation(self, member_1):
        """does the mutation for a member"""
        #go through each bit
        temp_chromosome = ''
        for i in range (self.chromosome_length):
            #mutate bit
            if np.random.random() < self.mutation_rate:
                if int(member_1.chromosome[i])==0:
                    temp_chromosome += '1'
                else:
                    temp_chromosome += '0'
            else:
                temp_chromosome += member_1.chromosome[i]
        member_1.chromosome = ''
        for i in temp_chromosome:
            member_1.chromosome += str(i)
    
    def epoch(self):
        """Evolves to next epoch. This assumes trials have already been
        completed"""
        next_generation = []
        self.calculate_fitness()
        for member_x in range (self.population_size):
            #select members
            selected = self.roulette_wheel_selection()
            #cross-over
            [baby_1, baby_2] = self.crossover(
                    self.member_list[selected[0]], self.member_list[selected[1]])
            #mutate the selected
            self.mutation(baby_1)
            next_generation.append(baby_1)
            self.mutation(baby_2)
            next_generation.append(baby_2)
        #next generation becomes the current generation
        self.member_list = next_generation

"""population parameters"""
#has to be even. Default is 140
population_size = 140
#default is 0.7
crossover_rate = 0.7
mutation_rate = 0.001
#num_actions is the number of instructions stored, and action_width is 
#how many bits each instruction is
num_actions = 70
action_width = 2

"""epochs"""
num_epochs = 300

maze = maze_object(maze_array, 10, 15) 
        
population = population_object(
        population_size, crossover_rate, mutation_rate, num_actions, action_width)

for j in range (num_epochs):
    for i in range(population.population_size):
        #do trials for each member
        maze.maze_trial(population.member_list[i], population.num_actions, population.action_width)
    #advance to next epoch
    population.epoch()
    print (str(j) + ' out of ' + str(num_epochs))

#this is for final checks
max_fitness = 0
adonis = 0
for i in range(population.population_size):
    #do trials for each member
    maze.maze_trial(population.member_list[i], population.num_actions, population.action_width)
    if population.member_list[i].fitness > max_fitness:
        max_fitness = population.member_list[i].fitness
        adonis = i
        position = maze.position
        steps = maze.steps_taken