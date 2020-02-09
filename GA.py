%reset
#====================================>>>>> Genetic Algorithm
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

def initial_population(img_shape, n_individuals):
    """
    Creating an initial population randomly.
    """
    # Empty population of chromosomes accoridng to the population size specified.
    init_population = np.random.randint(0, high=256, size=(n_individuals,ROWS * COLS * CHANNELS), dtype='uint8')
    return init_population  

def img2chromosome(img_arr):
    """
    First step in GA is to represent/encode the input as a sequence of characters.
    The encoding used is value encoding by giving each gene in the 
    chromosome its actual value in the image.
    Image is converted into a chromosome by reshaping it as a single row vector.
    """
    chromosome = np.reshape(a=img_arr, 
                               newshape=(functools.reduce(operator.mul, 
                                                          img_arr.shape)))
    return chromosome

def chromosome2img(chromosome, img_shape):
    """
    First step in GA is to represent the input in a sequence of characters.
    The encoding used is value encoding by giving each gene in the chromosome 
    its actual value.
    """
    img_arr = np.reshape(a=chromosome, newshape=img_shape)
    return img_arr

# Populating generation with arrays of desired image dimensions
def image_pop(pop):
  # Generating population array of desired shape
  #data = np.ndarray((len(pop), ROWS, COLS, CHANNELS), dtype=np.uint8)
  data = np.ndarray((len(pop), ROWS, COLS, CHANNELS)) #test

  # Populating generation with correct image dimensions
  for i in range(len(pop)):
    image = chromosome2img(pop[i], (ROWS , COLS, CHANNELS))
    data[i] = image
  return data

def fitness_fun(indiv_chrom):
    """
    Calculating the fitness of a single solution.
    The fitness is basicly calculated using trained neural network model prediction.
    """
    """
    Expanding dimension of each chromosome in order to match
    the dimension required by neural network's model
    """
    #x = cv2.resize(indiv_chrom, dsize=(ROWS, COLS), interpolation=cv2.INTER_CUBIC)
    #x = np.expand_dims(indiv_chrom, axis=0) #test
    #x = CNN.preprocess_input(x) #test

    ## test

    data = chromosome2img(indiv_chrom, (ROWS , COLS, CHANNELS))
    isave("file.jpg", data)
    data = cv2.imread("file.jpg", cv2.IMREAD_COLOR)
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(data, 'RGB')
    #img_path = 'img.jpg'
    #img.save(img_path)
    #img = image.load_img(img_path, target_size=(ROWS, COLS))
    x = image.img_to_array(img, "channels_last", "uint8")
    x = np.expand_dims(x, axis=0)
    x = CNN.preprocess_input(x)

    preds = CNN.new_model.predict(x)

    quality = preds[0,target_img_idx]
    
    return quality

def cal_pop_fitness(target_chrom, pop):
    """
    This method calculates the fitness of all solutions in the population.
    """
    qualities = np.zeros(len(pop))
    for indv_num in range(len(pop)):
        # Calling fitness_fun(...) to get the fitness of the current solution.
        qualities[indv_num] = fitness_fun(target_chrom, pop[indv_num][:])
    return qualities



def selection(fitness_values):
  """
  Selecting 2 individuals as parents for crossover
  using Roulette-wheel selection via stochastic acceptance
  """
  # an array of cumulative sum of sorted fitness values
  cum_sum = np.cumsum(sorted(fitness_values))
  # Normalizing values between range 0 to 1
  cum_sum_normal = np.interp(cum_sum, (cum_sum.min(), cum_sum.max()), (0, 1))
  # Finding sorted finess values indexes in order to use in loop to prevent a repeated selection
  sorted_fitness_idx = np.argsort(fitness_values)
  # A random number is used as probability of selection
  rand_num = random.random()
  # Defining an empty array to fill with 2 selected values
  selected = []
  # Loop in range 2 in order to select 2 indexes
  for i in range(2):
    find = False
    # A counter for index iteration
    num = 0
    while find == False:
      # if random number is bigger than second last value, selects the last index
      if rand_num > (cum_sum_normal[-2]):
        selected.append(sorted_fitness_idx[-1])
        # Deleting selected value from search space in oreder to prevent repeated selection
        sorted_fitness_idx = np.delete(sorted_fitness_idx, -1)
        # Generating a new random number for next selection
        rand_num = random.random()
        find = True
      elif rand_num > cum_sum_normal[num] and rand_num < (cum_sum_normal[num + 1]):
        selected.append(sorted_fitness_idx[num])
        # Deleting selected value from search space in oreder to prevent repeated selection
        sorted_fitness_idx = np.delete(sorted_fitness_idx, num)
        # Generating a new random number for next selection
        rand_num = random.random()
        num += 1
        find = True
      else:
        num += 1
  return selected


def single_crossover(parents):
  """
  Mating selected parents with a probability to happen
  a random number is generated which will be used as a point which seperates
  parents chromosomes in order to generate two offsprings
  """
  single_point = random.randint(0, len(parents[0]))
  offsprings = np.array((np.array(np.append(parents[0][0:single_point], parents[1][single_point:])),
                         np.array(np.append(parents[1][0:single_point], parents[0][single_point:]))))

  return offsprings

def two_crossover(parents):
  '''
  Mating selected parents with a probability to happen
  a random number is generated which will be used as a point which seperates
  parents chromosomes in order to generate two offsprings
  '''
  two_points = sorted(random.sample(range(0,len(parents[0])), 2))
  offsprings = np.array((np.array(np.append(np.append(parents[0][0:two_points[0]],
                                                      parents[1][two_points[0]:two_points[1]]), parents[0][two_points[1]:])),
                         np.array(np.append(np.append(parents[1][0:two_points[0]], 
                                                      parents[0][two_points[0]:two_points[1]]), parents[1][two_points[1]:]))))

  return offsprings

def uni_crossover(parents):
  indexes = random.sample(range(0,len(parents[0])), int(len(parents[0])/2))
  sol1 = np.copy(parents[0])
  sol2 = np.copy(parents[1])
  sol1[indexes] = parents[1][indexes]
  sol2[indexes] = parents[0][indexes]
  
  offsprings = np.array((sol1,sol2))

  return offsprings

def mutation(chromosome, gene_mutation, gene_range):
  """
  Applying mutation by selecting a predefined percent of genes randomly.
  Values of the randomly selected genes are changed randomly.
  """
  # A predefined percent of genes are selected randomly.
  rand_idx = random.sample(range(len(chromosome)), int(gene_mutation/100*len(chromosome)))
  # Changing the values of the selected genes randomly.
  new_values = np.random.randint(0,gene_range,size=len(rand_idx))
  # Updating chromosome after mutation.
  chrome = chromosome
  chrome[rand_idx] = new_values
  return np.array(chrome)
  
# Selecting elit fitness values regarding elit_percent given
def elits(fitnesses, elitism, pop):
  elit = np.argsort(fitnesses[-(math.ceil((elitism)/100*len(pop))):])
  return len(elit)

# Sorting population regarding order of fitness values
def sorting(pop, fit_idx):
  sor = []
  for i in range(len(pop)):
    sor.append(pop[fit_idx[i]])
  return np.array(sor)


"""
Calculating and normalizing fitness values using difference in distance of each
 gene of each chromosome to its corresponding gene value in target chromosome
"""
def normal(target, pop):
  normal_fitness = []
  max_dist = sum(np.maximum(target, 255-target))
  for i in range(len(pop)):
    dist = sum(np.absolute(target - pop[i]))
    q = ((max_dist - dist)/max_dist)*100
    normal_fitness.append(q)
  return normal_fitness

# Qualities are given in percent
def cent_q(pop):
  normal_fitness = []
  for i in range(len(pop)):
    q = float(fitness_fun(pop[i]) * 100)
    normal_fitness.append(q)
  return normal_fitness
