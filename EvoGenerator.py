%reset

import GA
import CNN
import cv2
%pylab inline
import os
import numpy as np
import matplotlib.pyplot
import matplotlib.image as mpimg
from matplotlib import ticker
import itertools
import functools
import operator
import random
import heapq
import math
import _pickle as cPickle
from matplotlib.pyplot import imsave as isave
from PIL import Image

ROWS = 224
COLS = 224
CHANNELS = 3

# Population size
sol_per_pop = 300
# Generation numbers
generation = 20000
# Elitism percentage
elit_percent = 10.0
# Gene mutation percentage
mutation_percent = 3.0
# Probability of performing mutation on an indv
p_m = 0.5
# Probability of performing crossover on an indv
p_c = 0.9
# Maximum possible value for a gene to be changed randomly (Default = 256 for RGB)
gene_val = 256

#======================>>>>>>>>>>>>>>>>>> Main Function
def main(target, n_individuals, num_generation, elitism, cross_prob, mut_prob, gene_mutation, gene_range):

  # Creating an initial population randomly.
  new_population = np.array(GA.initial_population(img_shape=target, 
                                           n_individuals=GA.sol_per_pop))
  
  # Defining 2 arrays for saving best and average solutions in each generation to use for plotting
  best_sol = []
  average_sol = []

  # Populating generation with arrays of desired image dimensions
  #im_pop = GA.image_pop(new_population) #test

  for iteration in range(num_generation):
      # Measing the fitness of each chromosome in the population.
      qualities = GA.cent_q(new_population) #test
      fit_sorted = np.argsort(qualities)

      print('Quality : ', np.max(qualities), ', Iteration : ', iteration)

      new_population = GA.sorting(new_population, fit_sorted)

      #im_pop = GA.sorting(im_pop, fit_sorted) #test

      qualities = GA.sorted(qualities)

      elit = GA.elits(qualities, elitism, new_population)

      num = 0

      #=========>>>>>>>>>>>>>Loop for crossover and mutation
      for idx in range(int((len(new_population) - elit)/2)):
        """
        Applying mutation for offspring.
        Mutation is important to avoid local maxima. Avoiding mutation makes 
        the GA falls into local maxima.
        Also mutation is important as it adds some little changes to the offspring. 
        If the previous parents have some common degaradation, mutation can fix it.
        Increasing mutation percentage will degarde next generations.
        """
        # Selecting 2 chromosomes in each iteration respectively to be replaced or changed through crossover, mutation or both
        pair = new_population[num : num + 2]

        # Randomly selecting 2 chromosomes to be used as parents for crossover or mutated or both
        selected = GA.selection(qualities)

        # Preventing global variable change
        sel1 = copy(new_population[selected[0]])
        sel2 = copy(new_population[selected[1]])

        rand = random.random()

        if cross_prob > rand:
          offspring = two_crossover((sel1, sel2))
          if mut_prob > rand:
            offspring = np.array((mutation(offspring[0], gene_mutation, gene_range),
                                   mutation(offspring[1], gene_mutation, gene_range)))
        elif mut_prob > rand:
          offspring = np.array((mutation(sel1, gene_mutation, gene_range),
                                 mutation(sel2, gene_mutation, gene_range)))
        else:
          offspring = pair

        # Replacing new 2 selected old chromosomes with 2 new generated chromosomes
        new_population[num : num + 2] = offspring
        num = num + 2

      #im_pop = GA.image_pop(new_population) #test

      # Appending best and average solutions in each generation to use for plotting
      best_sol.append(np.amax(qualities))
      average_sol.append(np.mean(qualities))

      #test ( to be used in google colab )
      best_solution_chrom = new_population[numpy.where(qualities == np.max(qualities))[0][0]]
      data = GA.chromosome2img(best_solution_chrom, (ROWS , COLS, CHANNELS))
      GA.isave("file.jpg", data)
      data = GA.cv2.imread("file.jpg", cv2.IMREAD_COLOR)
      data = GA.cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
      img = GA.Image.fromarray(data, 'RGB')
      img_path = '/content/drive/My Drive/best.jpg'
      GA.img.save(img_path)

      if iteration % int(num_generation/10) == 0:
        print('\n\nGeneration: ',iteration, '\nBest fitness: ', np.amax(qualities), '%\n\n')
  
  im_pop =  GA.image_pop(new_population)
  best_sol.append(np.amax(qualities))
  average_sol.append(np.mean(qualities))
  best_solution_chrom = GA.im_pop[numpy.where(qualities == numpy.max(qualities))[0][0]]
  print('\n\nFinal best fitness: ', np.amax(qualities), '%\n')

  GA.cPickle.dump( best_solution_chrom, open( "best.pkl", "wb" ) )
  GA.cPickle.dump( im_pop, open( "im_pop.pkl", "wb" ) )
  GA.cPickle.dump( new_population, open( "new_pop.pkl", "wb" ) )
  
  # Displaying result image
  GA.plt.imshow(best_solution_chrom)
  GA.plt.show()


  GA.plt.xlabel('Iterations')
  GA.plt.ylabel('Fitness ( Percent )')
  GA.plt.title('Solutions')
  GA.plt.plot(best_sol, 'blue', label='Best Solution')
  GA.plt.plot(average_sol, 'green', label='Average Solution')
  GA.plt.xticks(range(0,num_generation+1)[0::int(num_generation/10)])
  GA.plt.legend()
  GA.plt.show()


target_image = (ROWS, COLS, CHANNELS)

main(target = target_image, n_individuals = sol_per_pop, num_generation = generation,
     elitism = elit_percent, cross_prob = p_c, mut_prob = p_m,
     gene_mutation = mutation_percent, gene_range = gene_val)
