#%%  Importing
import numpy as np
from world_class import World
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from ode_solver import Euler,RungeKutta45

import matplotlib.lines as mlines
#%% Initial values and parameters

alpha , beta , gamma = [0.2,0.2,0.2,0.2],[0.7,0.5,0.4,0.3],[0,0,0,0]
Nations = [10000,10000,10000,10000]
initial_conditions_SIR = [9700,300,0]
k_SIS, k_SIR = 0,2 # selecting a single nation to model for the SIS and SIR models
T_final,dt = 100, 0.2

# Initiate the world model
my_world = World(Nations) # initiates a world with a list of population counts
my_world.add_dissease(alpha,beta,gamma,"Chicken Pox",0) # adds the disease parameters



##### Chickenpox
#  beta = between 1 and 2.4
# duration = 5-10 
# alpha = between 0.1 and 0.2

 
Red = np.array([[209, 15, 18],[242, 102, 101],[252, 153, 148]])/255
Green = np.array([[119, 221, 102],[143, 227, 129],[167, 233, 156]])/255
Yellow = np.array([[255, 160, 113],[254, 201, 143],[253, 241, 173]])/255
Color = np.array([[43, 52, 103],[242, 102, 101],[254, 201, 143],[143, 227, 129]])/255
leg = [0,0,0,0]
for i in range(4):
    sol_SIR,t = my_world.forcast_SIR(initial_conditions_SIR,T_final,dt,k=i)
    leg[i],=plt.plot(t,sol_SIR[:,1]/Nations[i],label = f'$\\beta$= {beta[i]}',linestyle='-',color=(Color[i][0],Color[i][1],Color[i][2]) )
    plt.plot(t,sol_SIR[:,0]/Nations[i],linestyle=':',color=(Color[i][0],Color[i][1],Color[i][2]))
    plt.plot(t,sol_SIR[:,2]/Nations[i],linestyle='--',color=(Color[i][0],Color[i][1],Color[i][2]))

# Create custom legend handles for line styles
legend2 = mlines.Line2D([], [], color='black', linestyle='-', label=r'$I(t)$ Infected')
legend3 = mlines.Line2D([], [], color='black', linestyle='--', label=r'$R(t)$ Recovered')
legend1 = mlines.Line2D([], [], color='black', linestyle=':', label=r'$S(t)$ Susceptible')
# Create the first legend for line styles and place it above the figure
line_style_legend = plt.legend(handles=[legend1, legend2,legend3], loc='upper left')
# Create the second legend for beta values and place it below the figure
plt.legend(loc='upper right')
plt.gca().add_artist(line_style_legend)
plt.xlabel("Time [days]")
plt.ylabel("Number of people/total population")
plt.title(f'Runge Kutta 4 solutions to the SIR-model for $\\alpha$ = 0.2')
plt.ylim(0,1.4)
plt.show()

