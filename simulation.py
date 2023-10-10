#%%  Importing
import numpy as np
from world_class import World
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from ode_solver import Euler,RungeKutta45
import random
import matplotlib.lines as mlines
from matplotlib.animation import FuncAnimation


#%% Initial values and parameters
T_final = 30 # 
dt = 0.2 # Numerical step for ODE solver
nr_of_cities =15*15
N = nr_of_cities
grid_length = 15# Cities /nations are placed in a square grid
population_density = [1000,5000] # populations are randomly generated in the range "population_density"
Tij = 0.03 # Travel probability between adjacent cities


## Initial population and infection data
population_data = [random.randint(population_density[0], population_density[1]) for i in range(nr_of_cities)]
population_data = 3000*np.ones(nr_of_cities)
Initial_infected_data = np.zeros(N)
Initial_infected_data[15*15//2] = 0.7*population_data[15*15//2]
initial_conditions = np.concatenate([population_data-Initial_infected_data,Initial_infected_data,np.zeros(N)])

# Make balanced travel matrix
T_matrix = World.make_balanced_travel_matrix(grid_length,3,Tij,population_data)
#T_random = World.make_balanced_randomized_travel_matrix(nr_of_cities,population_data)

#T_matrix[5][nr_of_cities-1] = 0.2
#T_matrix[nr_of_cities-1][5] = 0.1 *(population_data[5]/population_data[nr_of_cities-1])
# Dissease parameters
alpha  =np.ones(N)*0.1
beta = np.ones(N)*2.4
gamma = np.ones(N)*0.03
# Initiate the world model
my_world = World(population_data) # initiates a world with a list of population counts
my_world.add_dissease(alpha,beta,gamma,"disease",0) # adds the disease parameters



# getting solutions for the SIRT
sol_SIRT,t = my_world.forcast_SIRT(initial_conditions,T_final,dt,T_matrix)
# unpack the solutions
Solutions_norestr = [sol_SIRT[:, i] for i in range(N*3)]
infected_data_norestr = Solutions_norestr[N:2*N]

ani = World.world_annimation(population_data,infected_data_norestr,grid_length,1000,dt)
#ani.save('norestrictions2.gif', writer='pillow',dpi = 300, fps=1)

#################################################################
#%%




#%%
def f_restrictions(t):
    return np.exp(-3*t)
def sigmoid(x):
    L = 0.8  # Maximum value
    k =2  # Steepness of the curve
    x0 = 5  # Midpoint
    return L / (1 + np.exp(-k * (-(x+0.15)*10 + x0))) +0.2

my_world.add_restrictions(sigmoid,sigmoid)

# getting solutions for the SIRT
sol_SIRT,t = my_world.forcast_SIRT(initial_conditions,T_final,dt,T_matrix)
# unpack the solutions
Solutions_rest = [sol_SIRT[:, i] for i in range(N*3)]
infected_data_rest = Solutions_rest[N:2*N]

ani = World.world_annimation(population_data,infected_data_rest,grid_length,1000,dt)
#ani.save('restrictions2.gif', writer='pillow',dpi=400, fps=1)
#%%

#%%
Color = np.array([[43, 52, 103],[242, 102, 101],[254, 201, 143],[143, 227, 129]])/255
the_lucky_city = 15*7+5
plt.plot(t,infected_data_norestr[the_lucky_city]/population_data[the_lucky_city],color = (235/255, 69/255, 95/255),label = "Population 1, no restrictions")
plt.plot(t,infected_data_rest[the_lucky_city]/population_data[the_lucky_city],color = (43/255, 52/255, 103/255),label ="Population 1, restricions")

Color = np.array([[43, 52, 103],[242, 102, 101],[254, 201, 143],[143, 227, 129]])/255
the_lucky_city = 15*14
plt.plot(t,infected_data_norestr[the_lucky_city]/population_data[the_lucky_city],linestyle='--',color = (235/255, 69/255, 95/255),label = "Populatin 2,no restrictions")
plt.plot(t,infected_data_rest[the_lucky_city]/population_data[the_lucky_city],linestyle='--',color = (43/255, 52/255, 103/255),label ="Population 2, restricions")
plt.xlabel("Time [days]")
plt.ylabel("Number of people / total population")
plt.ylim(0,0.8)
plt.legend()
plt.savefig("restiction effect",dpi = 400,transparent = True)
plt.show()

