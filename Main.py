#%%
import numpy as np
from world_class import World
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from ode_solver import Euler,RungeKutta45

## TODO: Include error analysis
## TODO: Configure plots. Think about what kind of figures we want to present. if there is time do a heat map 2D simulation
## TODO: Try to find exact solutions if possible. and compare with our solutions
## TODO: Work on further extensions of the SIR model:
#           suggestions: - Supressed traveling. compare what happens if traveling is supressed based on the number of infected people in a population
#                        - SIR with different alpha, beta, gamma for each nation 
#                        -  . . . 
# TODO: choose a disease to model

#%%                                            SIS-model
#---------------------------------------------------------------------------------------------------------
# Disease parameters and initial conditions
alpha , beta , gamma = [0.07,0.07,0.07],[0.2,0.2,0.2],[0.01,0.01,0.01]
Nations = [100,500,1000]
initial_conditions_SIS = [90,10]
initial_conditions_SIR = [900,100,0]
initial_conditions_SIRT = [90,350,900,10,150,100,0,0,0]
k_SIS, k_SIR = 0,2 # selecting a single nation to model for the SIS and SIR models
T_final,dt = 100, 1
# Initiate the world model
my_world = World(Nations) # initiates a world with a list of population counts
my_world.add_dissease(alpha,beta,gamma,"disease") # adds the disease parameters
# ----------------------------------------------------------------------------

#%%
# SIS forcast and solution
sol_SIS,t = my_world.forcast_SIS(initial_conditions_SIS,T_final,dt,k=k_SIS)

# SIS EXACT solution
def exact_SIS(t,beta,alpha,N,I0):
    r = beta/N
    B = beta -alpha
    C = (B-I0*r)/(B*I0)
    return B*(r+B*C*np.exp(-B*t))**(-1)

# Plot
plt.figure()
plt.plot(t,sol_SIS[:,1],label = "Infected RK solution")
plt.plot(t,exact_SIS(t,beta[0],alpha[0],Nations[0],initial_conditions_SIS[1]),label = "Exact")
plt.title("Infected")
plt.xlabel("days")
plt.legend()
plt.show()

#%%                                             SIR-model
#-------------------------------------------------------------------------------------------------------------

sol_SIR,t = my_world.forcast_SIR(initial_conditions_SIR,T_final,dt,k=k_SIR)
plt.plot(t,sol_SIR[:,1],label = "I(t)",color = "r")
plt.plot(t,sol_SIR[:,0],label = "S(t)",color = "y")
plt.plot(t,sol_SIR[:,2],label = "R(t)",color = "g")
plt.title("SIR model")
plt.legend()
plt.show()


#%%                             Vaccination or no vaccination: SIS vs SIR forcasts
#-------------------------------------------------------------------------------------------------------------
alpha , beta , gamma = [0.07,0.07,0.07],[0.6,0.6,0.6],[0.01,0.01,0.01] 
v_sol,t = my_world.forcast_SIR([900,100,0],T_final,dt,k=2)
av_sol,t = my_world.forcast_SIS([900,100],T_final,dt,k=2)

plt.plot(t,v_sol[:,1],label="Vaccination:ON")
plt.plot(t,av_sol[:,1],label = "Vaccination: OFF")
plt.legend()
plt.title("Number of infected members")
plt.title("Vaccination effect")
plt.xlabel("days")
plt.ylabel("number of infected people")
plt.show()
#-------------------------------------------------------------------------------------------------------------



# %% SIRT : SIR model with traveling
# -----------------------------------------------------------------------------------------------------------

# For the population of each nation to be constant, the travel matrix elements must satisfy:
# T_ij N_i = T_ji N_j , number of people traveling from i to j is equal to those traveling from j to i

#Constructing the travel matrix
N0, N1,N2 = Nations[0],Nations[1],Nations[2]
T01 ,T02,T12 = 0.1,0.3,0.07
T10,T20,T21 = T01*N0/N1,T02*N0/N2,T12*N1/N2
T_matrix = np.array([[0,T01,T02],[T10,0,T12],[T20,T21,0]])

# getting solutions for the SIRT
sol_SIRT,t = my_world.forcast_SIRT(initial_conditions_SIRT,T_final,dt,T_matrix)
# unpack the solutions
S1, S2, S3, I1, I2, I3, R1, R2, R3 = [sol_SIRT[:, i] for i in range(9)]

# Plotting
nations_names = ["Nation 1", "Nation 2","Nation 3"]
colors = ["r", "y", "g"]
data = [I1, S1, R1, I2, S2, R2,I3,S3,R3]
fig, axs = plt.subplots(1, len(nations_names), figsize=(20, 6))

for i, nation in enumerate(nations_names):
    ax = axs[i]
    start = i * 3
    end = (i + 1) * 3
    for j in range(start, end):
        ax.plot(t, data[j], label=["Infected", "Susceptible", "Recovered"][j % 3], color=colors[j % 3])
    ax.set_title(nation)
    ax.legend()
    ax.set_aspect('auto')

plt.tight_layout()
plt.show()


# %%  Compaire traveling vs no traveling:

# %% 
