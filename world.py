import numpy as np
from ode_solver import Euler,RungeKutta45
class World:
    """ This is a world class meant for the use of modeling and forcasting disease spreading 
    - Each world has a number of nations each containing N members.
    - Each nation can have a number of susceptible members Sn and a number of infected members In
    - Optionally the nation can have a number of vaccinated and immune members Rn.
    - Members of one nation can travel from one place to another with traveling matrix T where T_ij means the probability a member from nation i travels to nation j
    - The rate of infection of susceptible is (b) or (beta)
    - The rate of recovery is (a) or (alpha)
    - The rate of vaccination of susceptibles turning S into R is (g) or (gamma)
    - 
    """
    def __init__(self,population: int):
        if isinstance(population,(int,float)):
            self.nr_of_nations = 1
            population_count = population
            total_count = population
        else:
            self.nr_of_nations = (np.array(population)).size
            population_count = np.array(population)
            total_count = sum(population_count)

        self.population_count = population_count # a list with the number of people in each nation ex: [100,200,40, . . . ]
        self.total_population_count = total_count # total number of people in the world


    def forcast_SIS(self,alpha,beta,initial_conditions,T_final,dt,odesolver = "RK"):
        if self.nr_of_nations != 1: raise ValueError("Number of nations must be 1 for this SIS model") # assumes one nation world 
        N = self.total_population_count
        t = np.arange(0,T_final,dt)
        def f(t,y):
            # The right hand function of the ODE dy/dt = f(t,y)
            # 
            S,I = y
            return np.array([  -beta*S*I/N + alpha*I,+beta*S*I/N - alpha*I])
        if odesolver == "RK": mysolver = RungeKutta45(f)
        elif odesolver == "E": mysolver  = Euler(f)
        else: raise ValueError("odesolver unrecognized odesolver. Try E for Euler and RK for Runge kutta 45")

        mysolver.set_initial_conditions(initial_conditions)
        sol,t = mysolver.solve(t)
        return sol , t

    def forcast_SIR(self,alpha,beta,gamma,initial_conditions,T_final,dt,odesolver = "RK"):
        if self.nr_of_nations != 1: raise ValueError("Number of nations must be 1 for this SIS model") # assumes one nation world 
        # alpha: recovery rate , beta: infection rate  , gamma: vaccinatin rate
        N = self.total_population_count 
        def f(t,y):
            # Defines the right hand function of the ODE dy/dt = f(t,y)
            S,I,R = y
            return np.array([-beta*S*I/N -gamma*S, beta*S*I/N -alpha*I, gamma*S+alpha*I])
        # Assign solver from ode_solver
        if odesolver == "RK": solver = RungeKutta45(f)
        elif odesolver == "E":solver = Euler(f)
        else: raise ValueError("ode solver unrecognized")

        solver.set_initial_conditions(initial_conditions)
        t = np.arange(0,T_final,dt)
        sol,t = solver.solve(t)
        return sol ,t
    
    
    def forcast_SIRT(self,alpha,beta,gamma,Travel_matrix,initial_conditions,T_final,dt):
        # SIR model with traveling 
        nr_of_nations = self.nr_of_nations
        N = self.population_count
        T = Travel_matrix

        def f1(t,y):
            S,I,R = y
            return np.array([-beta*S*I/N -gamma*S, beta*S*I/N -alpha*I, gamma*S+alpha*I])
        def f(t,SIR_vect):
            # Expect a vector of the form [S1, S2, S3, . . , I1,I2,I3, . . . , R1,R2,R3, . . . ]
            f_vect = np.zeros(3*nr_of_nations)
            S,I,R = np.split(SIR_vect,len(SIR_vect)//nr_of_nations)[0],np.split(SIR_vect,len(SIR_vect)//nr_of_nations)[1],np.split(SIR_vect,len(SIR_vect)//nr_of_nations)[2]
            for i in range(nr_of_nations):
                f_vect[i] = -beta*S[i]*I[i]/N[i] -gamma*S[i] + np.dot(T[:,i],S)-sum(T[i])*S[i]
                f_vect[nr_of_nations+i] = beta*S[i]*I[i]/N[i] - alpha*I[i] + np.dot(T[:,i],I) - sum(T[i])*I[i]
                f_vect[2*nr_of_nations+i] = gamma*S[i] + alpha*I[i] +np.dot(T[:,i],R) - sum(T[i])*R[i]
            return f_vect
        if nr_of_nations ==1: odesolver = RungeKutta45(f1)
        else: odesolver = RungeKutta45(f)
        t = np.arange(0,T_final,dt)
        odesolver.set_initial_conditions(initial_conditions)
        sol,t = odesolver.solve(t)
        return sol,t
    
