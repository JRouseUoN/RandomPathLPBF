# -*- coding: utf-8 -*-
"""
Spyder Editor
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix

#A program to predict the transient thermal response of a moving 2D heat source



# Define the Gaussian function
def gaussian(x, y, mu_x, mu_y, sigma_x, sigma_y):
    return np.exp(-((x - mu_x)**2 / (2 * sigma_x**2) + (y - mu_y)**2 / (2 * sigma_y**2)))



# Parameters
Lx, Ly = 100, 100  # Length of the plane in x and y directions
T = 1.0   # Total time
Nx, Ny = 500, 500  # Number of spatial points in x and y directions
Nt = 10  # Number of time points

density=7.8e-6 # kg/mm^3
thermal_conductivity=45e-3 # W/mmK
specific_heat= 420 # J/kgK

alpha = thermal_conductivity/(specific_heat*density)  # Thermal diffusivity m^2/s

dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
dt = T / (Nt - 1)

thickness=1 #plate thickness 

# Parameters for the Gaussian
mu_x = 50
mu_y = 50
sigma_x = 1
sigma_y = 1

# Moving heat source parameters
source_amplitude = 0.001
# source_width = 0.1
source_speed = 20

# Coordinate Matrices
x_mat=np.ones((Nx,Ny))
for i in range(0,Nx,1):
    x_mat[:,i]=x_mat[:,i]*(i*dx)


y_mat=np.ones((Nx,Ny))
for i in range(0,Ny,1):
    y_mat[i,:]=y_mat[i,:]*(i*dy)

# Initial temperature distribution
u = np.zeros((Nt, Nx, Ny))
heat_source = np.zeros((Nt, Nx, Ny))
# Set Boundary Condition and initial state
u[0,:,:]=np.ones((Nx, Ny))*20

   
# Starting heat source            
heat_source_temp = gaussian(x_mat, y_mat, mu_x, mu_y, sigma_x, sigma_y)    
heat_source_temp = heat_source_temp*source_amplitude
heat_source[0,:,:] = heat_source_temp
u_temp=heat_source_temp*(1/(dx*dy*thickness*density))*(1/specific_heat)
u[0, :, :]=u[0, :, :]+u_temp
mu_y=mu_y+(source_speed*dt)


# Coefficient matrix A
A = lil_matrix((Nx*Ny, Nx*Ny))
for i in range(1, Nx-1):
    for j in range(1, Ny-1):
        index = i * Ny + j
        A[index, index] = -2 * alpha * dt / dx**2 - 2 * alpha * dt / dy**2 - 1
        A[index, index - Ny] = alpha * dt / dx**2
        A[index, index + Ny] = alpha * dt / dx**2
        A[index, index - 1] = alpha * dt / dy**2
        A[index, index + 1] = alpha * dt / dy**2
        

# Time-stepping loop
for n in range(1, Nt,1):
    b = u[n-1,:,:].flatten()
    deltaT=A.dot(b)
    T_new_flattened=b+(deltaT*dt)
    u[n,:,:]=T_new_flattened.reshape((Nx, Ny))
    
    
    
    # Evaluate heat source
    # heat_source_temp = gaussian(x_mat, y_mat, mu_x, mu_y, sigma_x, sigma_y)    
    # heat_source_temp = heat_source_temp*source_amplitude
    # heat_source[n,:,:] = heat_source_temp
    # # Apply heat source to temperature field
    # u_temp=heat_source_temp*(1/(dx*dy*thickness*density))*(1/specific_heat)
    # u[n, :, :]=u[n, :, :]+u_temp
    # # Update heat source location
    # mu_y=mu_y+(source_speed*dt)
    # print(n)




# # Plot the results
plt.figure(figsize=(10, 6))
X, Y = np.meshgrid(np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny))
for n in range(0, Nt, 1):
    plt.contourf(X, Y, u[n].T, levels=50, cmap='hot')
    plt.colorbar()
    plt.title(f'Temperature distribution at t={n*dt:.2f}s')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


# # Plot the results
# plt.figure(figsize=(10, 6))
# X, Y = np.meshgrid(np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny))
# for n in range(0, Nt, 1):
#     plt.contourf(X, Y, heat_source[n].T, levels=50, cmap='hot')
#     plt.colorbar()
#     plt.title(f'Heat Source at t={n*dt:.2f}s')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.show()
