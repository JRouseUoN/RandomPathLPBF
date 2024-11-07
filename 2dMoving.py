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
T = 4.0   # Total time
Nx, Ny = 500, 500  # Number of spatial points in x and y directions
Nt = (np.rint(40*T)).astype(int)  # Number of time points

density=7.8e-6 # kg/mm^3
thermal_conductivity=45e-3 # W/mmK
specific_heat= 420 # J/kgK

alpha = thermal_conductivity/(specific_heat*density)  # Thermal diffusivity m^2/s

dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
dt = T / (Nt - 1)

thickness=1 #plate thickness 

# Parameters for the Gaussian
mu_x = 30
mu_y = 30
sigma_x = 1.5
sigma_y = 1.5

# Moving heat source parameters
source_amplitude = 0.1
# source_width = 0.1
source_speed = 30



# Coordinate Matrices
x_mat, y_mat = np.meshgrid(np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny))


# Initial temperature distribution
u = np.zeros((Nt, Nx, Ny))
heat_source = np.zeros((Nt, Nx, Ny))
# Set Boundary Condition and initial state
initial_temperature=20
u[0,:,:]=np.ones((Nx, Ny))*initial_temperature

   
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
        
T_track=0

# Time-stepping loop
for n in range(1, Nt,1):
    b = u[n-1,:,:].flatten()
    deltaT=A.dot(b)
    T_new_flattened=b+(deltaT*dt)
    u[n,:,:]=T_new_flattened.reshape((Nx, Ny))
    T_track=T_track+dt
    if T_track<2:
        # Evaluate heat source
        heat_source_temp = gaussian(x_mat, y_mat, mu_x, mu_y, sigma_x, sigma_y)    
        heat_source_temp = heat_source_temp*source_amplitude
        heat_source[n,:,:] = heat_source_temp
        # Apply heat source to temperature field
        u_temp=heat_source_temp*(1/(dx*dy*thickness*density))*(1/specific_heat)
        u[n, :, :]=u[n, :, :]+u_temp
        #Ensure outer edge is constant temperature
        u[n, 0, :]=initial_temperature
        u[n, Nx-1, :]=initial_temperature
        u[n, :, 0]=initial_temperature
        u[n, :, Ny-1]=initial_temperature
        # Update heat source location
        mu_x=mu_x+(source_speed*dt)
        mu_y=mu_y+(source_speed*dt)
    print(n)


vmin=0
vmax=800
colorbar_ticks = np.round(np.linspace(vmin, vmax, 6))
# # Plot the results
plt.figure(figsize=(10, 6))
for n in range(0, Nt, 10):
    plt.contourf(x_mat, y_mat, u[n,:,:], cmap='hot', levels=np.linspace(vmin, vmax, 50), vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(label='Temperature (deg C)', ticks=colorbar_ticks)
    cbar.ax.set_yticklabels([str(tick) for tick in colorbar_ticks])
    plt.title(f'Temperature distribution at t={n*dt:.2f}s')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()






