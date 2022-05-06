import numpy as np
import scipy
from scipy.optimize import fsolve
from scipy.optimize import root
from scipy.optimize import show_options
from scipy.optimize import OptimizeResult
from matplotlib import pyplot as plt


x0 = 80000000000.0  # offer pool
x1 = 80000000000.0  # ask pool
offer = 5500000000.0 # offer
x0_o = x0 + offer
gamma = 0.001
A = 100.0
p = x0 * x1
s = x0 + x1
x_start = 2.0 * np.sqrt(p)

def compute_curve_v2_d(x):
    return 4 * A * p * gamma**2 * x**3 * (s - x) / (gamma * x**2 + x**2 - 4 * p)**2 + p - x**2/4

def compute_ask_pool(x):
    d = scipy.optimize.fsolve(compute_curve_v2_d, x_start)
    return d


def main():
    d = scipy.optimize.fsolve(compute_curve_v2_d, x_start)
    print("d = ", d)
    
   
    # d = scipy.optimize.fsolve(compute_ask_pool, x_start)
    t1 = 4 * x0_o * A * gamma**2 * d**3
    t2 = d**2 * gamma + d**2
    
    def compute_curve_v2_ask_pool(x):
        return t1 * x * (x0_o + x - d) / (t2 - 4 * x0_o * x)**2 + x0_o * x - d**2/4
    
    def compute_curve_v2_ask_pool_1(x):
        return 4.0 * x0_o * x * A * gamma**2 / (d**2 * (gamma**2 + 1- 4*x0_o*x/d**2)**2) *d*(x0_o+x-d)+ \
            x0_o * x - d**2/4
    
   # x_1_start = d /  (2 * x0)
    x_1_start = d**2 / (2 * x0)
    
    #sol = scipy.optimize.fsolve(compute_curve_v2_ask_pool_1, x_1_start)
    #ask_pool = x1 - sol
    #print("sol = ", ask_pool)
    show_options(solver="root_scalar", method="newton")
    sol_root = root(compute_curve_v2_ask_pool_1, x_1_start)
    print("root = ", sol_root.x)
    print("ask pool = ", sol_root.x[0])
    print("ask amount = ", x1 - sol_root.x[0])
    #show_options(solver="root_scalar", method="bisect")
    #sol_root_bisect = root(compute_curve_v2_ask_pool, x_1_start)
    # print("root bisect = ", sol_root_bisect)
    #print(sol_root_bisect.x)
    
    
    
main()