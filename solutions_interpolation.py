import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator


#-------------------
#--- LOAD ARRAYS ---
#-------------------

solutions_dir = "./solutions/"

#--- axes ---

L_x, L_y = 1.0, 1.0
n_x, n_y = 100, 100

x = np.linspace(0, L_x, n_x + 1)
y = np.linspace(0, L_y, n_y + 1)
time_instances = np.linspace(140, 7000, 50, dtype=int)

#--- iterate ---

for field in ["p_1", "p_2"]:
    for t in time_instances:

        #--- interpolate ---

        field_vector = np.loadtxt(f"{solutions_dir}{field}/{field}_{t}.txt", dtype=float)

        field_matrix = np.reshape(field_vector, (n_x + 1, n_y + 1)).T

        field_func = RegularGridInterpolator((x, y), field_matrix)

        #--- check ---

        X, Y = np.meshgrid(x, y)
        points = (X, Y)

        field_eval = field_func(points)

        def plot_solution(vals):
            plt.figure(figsize=(6, 4))
            plt.contourf(X, Y, vals, cmap="viridis")
            plt.title(field)
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.colorbar()
            plt.savefig(f"{solutions_dir}{field}/{field}_{t}.png")
            plt.close()
        
        plot_solution(field_eval)