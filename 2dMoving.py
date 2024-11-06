# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt

#A program to predict the transient thermal response of a moving 2D heat source

# Parameters
Lx, Ly = 0.1, 0.1  # Length of the plane in x and y directions
T = 5.0   # Total time
Nx, Ny = 1000, 1000  # Number of spatial points in x and y directions
Nt = 100  # Number of time points

density=7800 # kg/m^3
thermal_conductivity=45 # W/mK
specific_heat= 420 # J/kgK

alpha = thermal_conductivity/(specific_heat*density)  # Thermal diffusivity m^2/s

dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
dt = T / (Nt - 1)

thickness=1 #plate thickness 

# Parameters for the Gaussian
mu_x = 0.01
mu_y = 0.05
sigma_x = 0.01
sigma_y = 0.01

# Coordinate Matrices
x_mat=np.ones((Nx,Ny))
for i in range(0,Nx,1):
    x_mat[:,i]=x_mat[:,i]*(i*dx)


y_mat=np.ones((Nx,Ny))
for i in range(0,Ny,1):
    y_mat[i,:]=y_mat[i,:]*(i*dy)


# Stability condition
if alpha * dt / dx**2 > 0.25 or alpha * dt / dy**2 > 0.25:
    raise ValueError("Stability condition violated. Reduce dt or increase dx/dy.")

# Initial temperature distribution
u = np.zeros((Nt, Nx, Ny))
# Set Boundary Condition and initial state
u[0,:,:]=np.ones((Nx, Ny))*20

# Moving heat source parameters
source_amplitude = 40.0
# source_width = 0.1
source_speed = 0.02
               

# Time-stepping loop
for n in range(1, Nt):
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            u[n, i, j] = (u[n-1, i, j] + alpha * dt / dx**2 * (u[n-1, i+1, j] - 2*u[n-1, i, j] + u[n-1, i-1, j]) +
                          alpha * dt / dy**2 * (u[n-1, i, j+1] - 2*u[n-1, i, j] + u[n-1, i, j-1]))
    
    # Evaluate heat source
    heat_source_temp = gaussian(x_mat, y_mat, mu_x, mu_y, sigma_x, sigma_y)    
    heat_source_temp = heat_source_temp*source_amplitude
    # plt.contourf(x_mat, y_mat, heat_source_temp, cmap='viridis')
    # plt.colorbar()
    # plt.show()
    
    # Apply heat source to temperature field
    u_temp=heat_source_temp
    
    dx*dy*thickness
    
    
    # Update heat source location
    mu_x=mu_x+(source_speed*dt)
    
    # if source_index_x < Nx:
        # u[n, source_index_x, j] += source_amplitude * np.exp(-((source_position - source_index_x*dx) / source_width)**2)
        #for j in range(Ny):
        #    u[n, source_index_x, j] += source_amplitude * np.exp(-((source_position - source_index_x*dx) / source_width)**2)

# Plot the results
plt.figure(figsize=(10, 6))
X, Y = np.meshgrid(np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny))
for n in range(0, Nt, Nt // 100):
    plt.contourf(X, Y, u[n].T, levels=50, cmap='hot')
    plt.colorbar()
    plt.title(f'Temperature distribution at t={n*dt:.2f}s')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


# Define the Gaussian function
def gaussian(x, y, mu_x, mu_y, sigma_x, sigma_y):
    return np.exp(-((x - mu_x)**2 / (2 * sigma_x**2) + (y - mu_y)**2 / (2 * sigma_y**2)))

