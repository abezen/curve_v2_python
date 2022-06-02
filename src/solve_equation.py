import numpy as np
import scipy
from scipy.optimize import fsolve
from scipy.optimize import root
from scipy.optimize import show_options
from scipy.optimize import OptimizeResult
from matplotlib import pyplot as plt
from sympy import plot_implicit, symbols, Eq, And 
y1, y2 = symbols('y1 y2')


x0 =   150000000000.0  # offer pool
x1 =   40000000000.0  # ask pool
offer = 5500000000.0 # offer
x0_o = x0 + offer
gamma = 0.0001
A = 100.0
p = x0 * x1
s = x0 + x1
x_start = 2.0 * np.sqrt(p)

def compute_curve_v2_d(x):
    return 4 * A * p * gamma**2 * x**3 * (s - x) / (gamma * x**2 + x**2 - 4 * p)**2 + p - x**2/4

def compute_ask_pool(x):
    d = scipy.optimize.fsolve(compute_curve_v2_d, x_start)
    return d

def compute_curve_v1_d(x):
    return 4.0 * A * s + x - 4.0 * x * A - x**3/(4.0 * p) 


def main():
    d = scipy.optimize.fsolve(compute_curve_v2_d, x_start)
    print("d = ", d[0])
    
    d_v1 = scipy.optimize.fsolve(compute_curve_v1_d, x_start)
    print("v1  d =", d_v1[0]);
    
   
    # d = scipy.optimize.fsolve(compute_ask_pool, x_start)
    t1 = 4 * x0_o * A * gamma**2 * d**3
    t2 = d**2 * gamma + d**2
    
    def compute_curve_v1_ask_pool(x):
        4.0 * A * (x0_o + x) + d_v1 - 4.0 * A * d_v1 - d_v1**3/(4.0 * x0_o * x)
    
    def compute_curve_v2_ask_pool(x):
        return t1 * x * (x0_o + x - d) / (t2 - 4 * x0_o * x)**2 + x0_o * x - d**2/4
    
    def compute_curve_v2_ask_pool_1(x):
        return 4.0 * x0_o * x * A * gamma**2 / (d**2 * (gamma**2 + 1- 4*x0_o*x/d**2)**2) *d*(x0_o+x-d)+ \
            x0_o * x - d**2/4
    
    
  
    x_1_start = d**2 / (2 * x0)
    
    x1_v1_start = d - x0
    
   
    show_options(solver="root_scalar", method="newton")
   # show_options(solver="root_scalar", method="bisect")
    #sol_root_v2 = root(compute_curve_v2_ask_pool, x_1_start)
    #print("v2 root = ", sol_root_v2.x)
    #print("v2 ask pool = ", sol_root_v2.x[0])
    #print("v2 ask amount = ", x1 - sol_root_v2.x[0])
    
    sol_root_v1 = root(compute_curve_v1_ask_pool, x1_v1_start)
    print("v1 root = ", sol_root_v1.x)
    print("v1 ask pool = ", sol_root_v1.x[0])
    print("v1 ask amount = ", x1 - sol_root_v1.x[0])
    
    
    
    
main()