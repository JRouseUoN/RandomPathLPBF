# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt

#A program to predict the transient thermal response of a moving 2D heat source

# Parameters
Lx, Ly = 10.0, 10.0  # Length of the plane in x and y directions
T = 50.0   # Total time
Nx, Ny = 100, 100  # Number of spatial points in x and y directions
Nt = 1000  # Number of time points
alpha = 0.01  # Thermal diffusivity

dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
dt = T / (Nt - 1)

# Stability condition
if alpha * dt / dx**2 > 0.25 or alpha * dt / dy**2 > 0.25:
    raise ValueError("Stability condition violated. Reduce dt or increase dx/dy.")

# Initial temperature distribution
u = np.zeros((Nt, Nx, Ny))

# Moving heat source parameters
source_amplitude = 80.0
source_width = 0.1
source_speed = 2.0

source_centre=np.array((1,5))
               

# Time-stepping loop
for n in range(1, Nt):
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            u[n, i, j] = (u[n-1, i, j] + alpha * dt / dx**2 * (u[n-1, i+1, j] - 2*u[n-1, i, j] + u[n-1, i-1, j]) +
                          alpha * dt / dy**2 * (u[n-1, i, j+1] - 2*u[n-1, i, j] + u[n-1, i, j-1]))
    
    # Update the position of the moving heat source
    source_position = source_speed * n * dt
    source_index_x = int(source_position / dx)
    
    if source_index_x < Nx:
        u[n, source_index_x, j] += source_amplitude * np.exp(-((source_position - source_index_x*dx) / source_width)**2)
        source_centre[0]=int(source_centre[0]+(source_speed*dt))
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
