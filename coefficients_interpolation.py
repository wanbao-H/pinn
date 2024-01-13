import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.interpolate import RegularGridInterpolator


#-------------------
#--- LOAD ARRAYS ---
#-------------------

coefficients_dir = "./coefficients/"

k_1_vector = np.loadtxt(coefficients_dir + "k_1.txt", dtype=float)
k_2_vector = np.loadtxt(coefficients_dir + "k_2.txt", dtype=float)
r_vector = np.loadtxt(coefficients_dir + "r.txt", dtype=float)


#--------------------------
#--- INTERPOLATE ARRAYS ---
#--------------------------

#--- axes ---

L_x, L_y = 1.0, 1.0
n_x, n_y = 100, 100

x = np.linspace(0, L_x, n_x + 1)
y = np.linspace(0, L_y, n_y + 1)

#--- grid values ---

k_1_matrix = np.reshape(k_1_vector, (n_x + 1, n_y + 1)).T
k_2_matrix = np.reshape(k_2_vector, (n_x + 1, n_y + 1)).T
r_matrix = np.reshape(r_vector, (n_x + 1, n_y + 1)).T

#--- interpolated functions ---

k_1_func = RegularGridInterpolator((x, y), k_1_matrix, bounds_error=False)
k_2_func = RegularGridInterpolator((x, y), k_2_matrix, bounds_error=False)
r_func = RegularGridInterpolator((x, y), r_matrix, bounds_error=False)

if __name__ == "__main__":
    
    #-------------
    #--- CHECK ---
    #-------------

    #--- define the grid for evaluation ---

    X, Y = np.meshgrid(x, y)
    points = (X, Y)

    #--- evaluate the functions on the grid ---

    k_1_eval = k_1_func(points)
    k_2_eval = k_2_func(points)
    r_eval = r_func(points)

    #--- plot ---

    def plot_coefficient(vals, name):
        # lev_exp = np.arange(np.floor(np.log10(vals.min())), np.ceil(np.log10(vals.max())))
        # levs = np.power(10, lev_exp)
        plt.figure(figsize=(6, 4))
        # plt.contourf(X, Y, vals, levs, norm=colors.LogNorm(), cmap="viridis")
        plt.contourf(X, Y, vals, cmap="viridis")
        plt.title(name)
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.colorbar()
        plt.savefig(f"{coefficients_dir}{name}.png")

    plot_coefficient(k_1_eval, "k_1")
    plot_coefficient(k_2_eval, "k_2")
    plot_coefficient(r_eval, "r")