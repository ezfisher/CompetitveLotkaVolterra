import numpy as np
import matplotlib.pyplot as plt

# Simulating the Lotka-Volterra predator prey model using 4th order runge kutta method

def CompetitiveLotkaVolterra(population, params):

    '''
    growth_rate: growth rate of each species (list)
    carrying_cap: carrying capacity of each species (list)
    alphas: parameter quantifying the affect FROM species 2 ON 1 (alphas[0]) and FROM spacies 1 ON 2 (alphas[1])
    '''
    growth_rate = params['growth_rate']
    carrying_cap = params['carrying_cap']
    alphas = params['alphas']

    # defining differential equation
    # population[0] is population of species 1, population[1] is the population of species 2

    derivative = [ growth_rate[0] * population[0] * ( 1 - (population[0] + alphas[0]*population[1])/carrying_cap[0] ),  
                growth_rate[1] * population[1] * ( 1 - ( population[1] + alphas[1]*population[0] )/carrying_cap[1] ) ]
    derivative = np.array(derivative)

    return derivative


# Runge kutta 4th order method
def RK4(f, population0, time0, time_max, time_step):
    # f is the generic name of the function that is being solved for

    # time vector
    time = np.arange(time0, time_max, time_step)


    num_times = time.size
    num_populations = population0.size

    # Initialize population array that holds information on the evolution of the system
    population = np.zeros((num_populations, num_times))

    population[:,0] = population0

    for i in range(num_times - 1):
        # define k values for RK4
        k1 = time_step * f(time[i], population[:,i])
        k2 = time_step * f(time[i] + time_step/2, population[:,i] + k1/2)
        k3 = time_step * f(time[i] + time_step/2, population[:,i] + k2/2)
        k4 = time_step * f(time[i] + time_step, population[:,i] + k3)

        # Calculate the change in population
        dpopulation = 1/6* (k1 + 2*k2 + 2*k3 + k4)

        population[:, i+1] = population[:, i] + dpopulation

    return population, time

# After some experimentation and consulting the Lotka-Volterra Wikipedia page, the following parameter values seem appropriate
params = {'growth_rate': [1.53, 1.27], 'carrying_cap': [1,1], 'alphas': [1.3, 1.3]}

# define function f
f = lambda time, population : CompetitiveLotkaVolterra(population, params)

# Initialize population. Starting with 50 prey animals and 20 predators arbitrarily
population0 = np.array([0.47, 0.35])
time0 = 0
time_max = 100
time_step = 0.01

population, time = RK4(f, population0, time0, time_max, time_step)

# Plots
plt.subplot(1, 2, 1)
plt.plot(time, population[0,:], 'r', label='Species 1')
plt.plot(time, population[1,:], 'b', label='Species 2')
plt.xlabel('Time')
plt.ylabel('Population')
plt.grid()
plt.legend()

# Phase plot
plt.subplot(1, 2, 2)
plt.plot(population[0,:], population[1,:])
plt.xlabel('Species 1')
plt.ylabel('Species 2')
plt.grid()

plt.show()