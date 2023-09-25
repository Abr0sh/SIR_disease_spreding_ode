import numpy as np
from tqdm import tqdm
class ODE_Solver:
    """ODESOLVER Mother class
    
    ODE:
    dy/dt = f(t,y),
    y(t=0)= y0 initial condition
    
    """

    def __init__(self,f):
        self.f = f

    def set_initial_conditions(self,Y0):
        if isinstance(Y0, (int, float)):
            # 1D ODE
            self.dimension = 1
            Y0 = float(Y0)
        else:
            # n dimensional ODE
            Y0 = np.array(Y0)
            self.dimension = Y0.size
        self.Y0 = Y0 
    
    def solve(self,time_points):
        self.t = np.array(time_points)
        n = self.t.size
        
        self.y = np.zeros((n, self.dimension))
        self.y[0, :] = self.Y0

        for i in tqdm(range(n-1),ascii = True):
            self.i = i
            self.y[i+1] = self.advance()

        return self.y, self.t
    def advance(self):
        """Advance the solution by one time step"""
        raise NotImplementedError
    

class Euler(ODE_Solver):
    def advance(self):
        y,f,i,t = self.y, self.f, self.i, self.t
        dt = t[i+1]- t[i]
        return y[i, :] + dt*f(t[i],y[i, :])
    def name(self): return "Euler method"

class RungeKutta45(ODE_Solver):
    def advance(self):
        y,f,i,t = self.y, self.f, self.i ,self.t
        dt = t[i+1]-t[i]
        dt2 = dt/2
        K1 = dt*f(t[i],y[i, :])
        K2 = dt*f(t[i]+dt2 ,y[i, :]+ 0.5*K1)
        K3 = dt*f(t[i]+dt2 ,y[i, :]+ 0.5*K2)
        K4 = dt * f(t[i]+ dt, y[i, :]+K3)

        return y[i, :] + (1/6)*(K1 + 2*K2 +2*K3 + K4)
    def name(self): return "Runge Kutta 4 method"