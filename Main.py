#%%
import numpy as np
from world import World
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
# constants and initial conditions
alpha,beta,initial_conditions,N = 0.1,0.15,[90,10],100
T_final,dt = 200, 1

# SIS forcast and solution
Mynation = World(N)
solRK,t = Mynation.forcast_SIS(alpha,beta,initial_conditions,T_final,dt)
solE,t = Mynation.forcast_SIS(alpha,beta,initial_conditions,T_final,dt,odesolver="E")
# SIS EXACT solution
def exact_SIS(t,beta,alpha,N,I0):
    r = beta/N
    B = beta -alpha
    C = (B-I0*r)/(B*I0)
    return B*(r+B*C*np.exp(-B*t))**(-1)

# Plot
plt.figure()
plt.plot(t,solRK[:,1],label = "Euler")
plt.plot(t,solE[:,1],label="Runge Kutta")
plt.plot(t,exact_SIS(t,beta,alpha,N,initial_conditions[1]),label = "Exact")
plt.title("Infected")
plt.xlabel("days")
plt.legend()
plt.show()

#%%                                             SIR-model
#-------------------------------------------------------------------------------------------------------------
B,g,a,N,SIRinitial_conditions = 0.5,0.1,0.05,10000, [9900,100,0]
T,dt = 100,1
Mynation = World(N)

sol,t = Mynation.forcast_SIR(a,B,g,SIRinitial_conditions,T,dt)
plt.plot(t,sol[:,1],label = "RK4 I")
plt.plot(t,sol[:,0],label = "S(t)")
plt.title("SIR model")
plt.legend()
plt.show()

#%% you can also use the SIRT model for one nation
B,g,a,N,SIRinitial_conditions = 0.5,0.1,0.05,10000, [9900,100,0]
T,dt = 100,1
Mynation = World(N)
T_matrix = [0]
sol,t = Mynation.forcast_SIRT(a,B,g,T_matrix,SIRinitial_conditions,T,dt)
plt.plot(t,sol[:,0])
plt.show()

#%%                             Vaccination or no vaccination: SIS vs SIR forcasts
#-------------------------------------------------------------------------------------------------------------
b,g,a,N,SIRinitial_conditions = 0.6,0.01,0.07,10000, [9900,100,0]
T,dt = 200,1
vaxWorld = World(N)
antivaxWorld = World(N)

v_sol,t = vaxWorld.forcast_SIR(a,b,g,[9900,100,0],T,dt)
av_sol,t = antivaxWorld.forcast_SIS(a,b,[9900,100],T,dt)
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

# Disease parameters
b,g,a,N,SIRinitial_conditions = 0.15,0.07,0.07,[100,1000], [85,500,10,500,5,0]
T,dt = 60,0.2 # time interval. max time and step size
Mynation = World(N)

# For the population of each nation to be constant, the travel matrix elements must satisfy:
# T_ij N_i = T_ji N_j , number of people traveling from i to j is equal to those traveling from j to i
T_matrix = np.array([[0,0.1],[0.01,0]]) 
sol,t = Mynation.forcast_SIRT(a,b,g,T_matrix,SIRinitial_conditions,T,dt)

# unpack the solutions
S1,S2 = sol[:,0],sol[:,1]
I1,I2 = sol[:,2],sol[:,3]
R1,R2 = sol[:,4],sol[:,5]

fig, axs = plt.subplots(1, 2, figsize=(20, 6))

axs[0].plot(t,I1,label= "Infected",color = "r")
axs[0].plot(t,S1,label = "Susceptible",color = "y")
axs[0].plot(t,R1,label = "Recovered",color= "g")
axs[0].set_aspect('auto')
axs[0].set_title("Nation 1")
axs[0].legend()

axs[1].plot(t,I2,label= "Ifected",color = "r")
axs[1].plot(t,S2,label = "Susceptible",color= "y")
axs[1].plot(t,R2,label = "Recovered",color = "g")
axs[1].set_title("Nation 2")
axs[1].legend()
axs[1].set_aspect('auto')



# %%
