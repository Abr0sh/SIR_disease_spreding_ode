import numpy as np
from ode_solver import Euler,RungeKutta45
import matplotlib.animation as animation
import matplotlib.pyplot as plt
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
        self.restrictions = False
        self.disease = False

    def add_restrictions(self,TR,CR):
        self.restrictions = True
        self.travel_restriction_function = TR
        self.contact_restricion_function = CR
    def remove_restrictions(self):
        self.restrictions = False

    def add_dissease(self,alpha,beta,gamma,name,fatality = 0,):
        self.alpha = np.array(alpha)
        self.beta = np.array(beta)
        self.gamma = np.array(gamma)
        self.disease_name = name
        self.fatality = fatality
        # TODO : implement fatality to all the SIR models
        # TODO : add check disease before using SIR
        self.disease = True


    def forcast_SIS(self,initial_conditions,T_final,dt,odesolver = "RK",k = 0):
        # add check for k and alpha beta
        alpha,beta = self.alpha[k],self.beta[k]
        N = self.population_count[k]
        t = np.arange(0,T_final,dt)
        fat = self.fatality
        def f(t,y):
            # The right hand function of the ODE dy/dt = f(t,y)
            # 
            S,I = y
            return np.array([  -beta*S*I/N + alpha*I,+beta*S*I/N - alpha*I-fat*I])
        if odesolver == "RK": mysolver = RungeKutta45(f)
        elif odesolver == "E": mysolver  = Euler(f)
        else: raise ValueError("odesolver unrecognized odesolver. Try E for Euler and RK for Runge kutta 45")

        mysolver.set_initial_conditions(initial_conditions)
        sol,t = mysolver.solve(t)
        return sol , t

    def forcast_SIR(self,initial_conditions,T_final,dt,odesolver = "RK",k = 0):
        alpha,beta,gamma = self.alpha[k],self.beta[k],self.gamma[k]
        N = self.population_count[k]
        def f(t,y):
            # Defines the right hand function of the ODE dy/dt = f(t,y)
            S,I,R = y
            fat = self.fatality
            return np.array([-beta*S*I/N -gamma*S, beta*S*I/N -alpha*I-fat*I, gamma*S+alpha*I])
        # Assign solver from ode_solver
        if odesolver == "RK": solver = RungeKutta45(f)
        elif odesolver == "E":solver = Euler(f)
        else: raise ValueError("ode solver unrecognized")

        solver.set_initial_conditions(initial_conditions)
        t = np.arange(0,T_final,dt)
        sol,t = solver.solve(t)
        return sol ,t
    
    
    def forcast_SIRT(self,initial_conditions,T_final,dt,Travel_matrix,odesolver = "RK"):
        # SIR model with traveling 
        alpha,beta,gamma = self.alpha,self.beta,self.gamma
        nr_of_nations = self.nr_of_nations
        N = self.population_count
        T = Travel_matrix

        if self.restrictions == True:
            TR = self.travel_restriction_function
            CR = self.contact_restricion_function
        else: 
            TR = lambda x: x-x +1
            CR = lambda x: x-x+1


        def f1(t,y):
            S,I,R = y
            # TODO apply restrictions to beta
            return np.array([-beta[0]*S*I/N -gamma[0]*S, beta[0]*S*I/N -alpha[0]*I, gamma[0]*S+alpha[0]*I])
        def f(t,SIR_vect):
            # Expect a vector of the form [S1, S2, S3, . . , I1,I2,I3, . . . , R1,R2,R3, . . . ]
            f_vect = np.zeros(3*nr_of_nations)
            S,I,R = np.split(SIR_vect,len(SIR_vect)//nr_of_nations)[0],np.split(SIR_vect,len(SIR_vect)//nr_of_nations)[1],np.split(SIR_vect,len(SIR_vect)//nr_of_nations)[2]
            r = np.array(I)/N
            cr = CR(r) # contact restriction 
            tr = TR(r) # travel restrictions
            T_restr = np.array([[T[i][j]*tr[i]*tr[j] for j in range(len(T))] for i in range(len(T))])
            for i in range(nr_of_nations):
                f_vect[i] = -cr[i]*beta[i]*S[i]*I[i]/N[i] -gamma[i]*S[i] + np.dot(T_restr[:,i],S)-sum(T_restr[i])*S[i]
                f_vect[nr_of_nations+i] = cr[i]*beta[i]*S[i]*I[i]/N[i] - alpha[i]*I[i] + np.dot(T_restr[:,i],I) - sum(T_restr[i])*I[i]
                f_vect[2*nr_of_nations+i] = gamma[i]*S[i] + alpha[i]*I[i] + np.dot(T_restr[:,i],R) - sum(T_restr[i])*R[i]
            return f_vect
        if nr_of_nations ==1: odesolver = RungeKutta45(f1)
        else: odesolver = RungeKutta45(f)
        t = np.arange(0,T_final,dt)
        odesolver.set_initial_conditions(initial_conditions)
        sol,t = odesolver.solve(t)
        return sol,t
    def world_annimation(population_data,infected_data,grid_length,interv,dt):

        # Create a figure and axis
        fig, ax = plt.subplots()
        num_cities = len(population_data)

        # Function to update tile colors and display percentage of infected people
        def update(frame):
            ax.clear()
            num_cols = grid_length  # Number of columns in the grid 
            num_rows = grid_length  # Number of rows in the grid 
            tile_size = 0.3  # Size of each tile
            color_map = plt.cm.RdYlGn  # Color map with red and green
            
            for i in range(num_cities):
                row = i // num_cols
                col = i % num_cols
                x = col * tile_size
                y =  ((num_rows - 1 - row) * tile_size)  # Adjusted y-coordinate
                
                infected_percentage = infected_data[i][frame] / population_data[i]
                color = color_map(1-infected_percentage)  # Invert color based on infection rate
                ax.add_patch(plt.Rectangle((x, y), tile_size, tile_size, color=color))
                
                '''ax.text(
                    x + tile_size / 2,  # X-coordinate for the text (centered)
                    y + tile_size / 2,  # Y-coordinate for the text (centered)
                    f'{int(infected_percentage * 100)}',  # Text to display (percentage)
                    ha='center',  # Horizontal alignment
                    va='center',  # Vertical alignment
                    color='black' if infected_percentage > 0.5 else 'white'  # Text color for visibility
                )'''
                
                ax.set_xlim(0, num_cols * tile_size)
                ax.set_ylim(0, num_rows * tile_size)
                ax.set_aspect('equal')
                ax.axis('off')
                time = round(dt*(frame+1),1)
                ax.set_title(f'Time Step {time} days')
            
        # Create a color bar
        cax = fig.add_axes([0.85, 0.1, 0.03, 0.8])  # [x, y, width, height]
        norm = plt.Normalize(0, 1)  # Normalize values to 0-1
        color_map = plt.cm.RdYlGn
        sm = plt.cm.ScalarMappable(cmap=color_map.reversed(), norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label('Infection Rate')

        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=len(infected_data[0]), repeat=True, interval=interv)

        # Display the animation
        plt.show()
        return ani

    def make_balanced_travel_matrix(n,m,a,population_data):
    # Make balanced travel matrix
    # n Size of the larger matrix
    # m Size of the smaller matrix
        def place_matrix_smaller_inside_larger(smaller_matrix, larger_matrix, i, j):
            # square matricies assumed
            pad = (len(smaller_matrix)-1)//2
            m = len(larger_matrix)

            # Create a new matrix with zeros and the desired padding
            matrix_with_padding = np.zeros((m + 2 * pad, m + 2 * pad), dtype=float)
            # Put the larger matrix inside the padded matrix
            matrix_with_padding[pad:-pad, pad:-pad] = larger_matrix

            matrix_with_padding[i+1-pad:i+2+pad,j+1-pad:j+2+pad] = smaller_matrix

            # Trim the paddings 
            return matrix_with_padding[pad:-pad, pad:-pad]
        T = []
        N = len(population_data)
        for i in range(n):
            for j in range(n):
                larger_matrix = np.zeros((n, n))
                smaller_matrix = np.array([[0,0,0], [0, 0, a], [a, a, a]])
                
                M = place_matrix_smaller_inside_larger(smaller_matrix, larger_matrix, i, j)
                T.append(M.ravel())
        T = np.array(T)
        for i in range(N):
            for j in range(i):
                T[i][j] = T[j][i]*(population_data[j]/population_data[i])
        return T
    def make_balanced_randomized_travel_matrix(n,population_data):
        T = np.zeros((n,n))
        for i in range(n):
            for j in range(i+1,n):
                T[i][j] = np.random.uniform(0.0003,0.1)
                T[j][i] = (population_data[i]/population_data[j])*T[i][j]
        return T

