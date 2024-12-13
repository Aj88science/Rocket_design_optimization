import numpy as np
import matplotlib.pyplot as plt     
import pandas as pd
import random

# Set seeds for reproducibility
np.random.seed(41)
random.seed(41)

# Constants
deg_to_rad = np.pi / 180  # Conversion from degrees to radians
Rt = 0.05  # Throat radius (in meters)
epsilon = 6.0  # Expansion ratio (exit area / throat area)
theta_n = 20  # Nozzle exit angle (in degrees)
theta_e = 5   # Nozzle end angle (in degrees)
Re = np.sqrt(epsilon) * Rt  # Exit radius
LN = 0.8 * ((np.sqrt(epsilon) - 1) * Rt) / np.tan(theta_n * deg_to_rad)  # Nozzle length

# Genetic Algorithm Parameters
population_size = 50
num_generations = 100  # Number of generations for the genetic algorithm
mutation_rate = 0.1
num_control_points = 2  # Only the Q point is optimized

# Define the fitness function
def fitness(Qx, Qy):
    Nx = 0.382 * Rt * np.cos((theta_n - 90) * deg_to_rad)
    Ny = 0.382 * Rt * np.sin((theta_n - 90) * deg_to_rad) + 0.382 * Rt + Rt
    Ex, Ey = LN, Re
    
    x_bell, y_bell = bezier_curve(Nx, Ny, Qx, Qy, Ex, Ey)
    
    # Calculate smoothness and length as part of fitness
    smoothness = np.sum(np.diff(np.gradient(y_bell))**2)  # Minimize curvature changes
    length = np.sqrt((x_bell[-1] - x_bell[0])**2 + (y_bell[-1] - y_bell[0])**2)
    length_penalty = abs(length - LN)
    
    return -(smoothness + length_penalty)  # Higher values for smoother, optimal length

# Helper function to compute Bézier curve
def bezier_curve(Nx, Ny, Qx, Qy, Ex, Ey, t=np.linspace(0, 1, 100)):
    # Quadratic Bézier parameterization
    x = (1 - t)**2 * Nx + 2 * (1 - t) * t * Qx + t**2 * Ex
    y = (1 - t)**2 * Ny + 2 * (1 - t) * t * Qy + t**2 * Ey
    return x, y

# Initialize population with random control points (only Q point optimized)
population = np.random.rand(population_size, 2) * [LN, Re]

# Run the genetic algorithm
convergence = []  # Store best fitness values for plotting convergence
for generation in range(num_generations):
    fitness_scores = np.array([fitness(ind[0], ind[1]) for ind in population])
    convergence.append(np.max(fitness_scores))  # Track the best fitness value
    
    # Selection: choose individuals with top fitness
    selected_indices = np.argsort(fitness_scores)[-population_size//2:]
    selected_population = population[selected_indices]
    
    # Crossover
    offspring = []
    for i in range(population_size//2):
        parent1, parent2 = selected_population[np.random.randint(0, len(selected_population), 2)]
        crossover_point = np.random.randint(1, 2)
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        offspring.append(child)
    
    # Mutation
    offspring = np.array(offspring)
    mutation_indices = np.random.rand(*offspring.shape) < mutation_rate
    offspring[mutation_indices] += np.random.normal(0, 0.01, mutation_indices.sum())
    offspring = np.clip(offspring, 0, [LN, Re])  # Ensure values stay within bounds
    
    # Create new population by combining parents and offspring
    population = np.vstack((selected_population, offspring))
    
    # Print progress
    if generation % 10 == 0:
        print(f"Generation {generation} - Best Fitness: {np.max(fitness_scores)}")

# Select the best individual from the final generation
best_individual = population[np.argmax([fitness(ind[0], ind[1]) for ind in population])]
Qx, Qy = best_individual[0], best_individual[1]

# Compute final nozzle profile
Nx, Ny = 0.382 * Rt * np.cos((theta_n - 90) * deg_to_rad), 0.382 * Rt * np.sin((theta_n - 90) * deg_to_rad) + 0.382 * Rt + Rt
Ex, Ey = LN, Re
x_bell, y_bell = bezier_curve(Nx, Ny, Qx, Qy, Ex, Ey)

# Plotting the optimized nozzle profile with interpolated dashed lines
plt.figure(figsize=(8, 6))
plt.plot(x_bell, y_bell, label='Optimized Bell Section (Bézier Curve)')
plt.scatter([Nx, Qx, Ex], [Ny, Qy, Ey], color='red', label='Control Points (N, Q, E)')
for t in np.linspace(0, 1, 10):  # Add interpolated points
    x_dashed, y_dashed = bezier_curve(Nx, Ny, Qx, Qy, Ex, Ey, t=np.array([t]))
    plt.plot(x_dashed, y_dashed, 'k--', alpha=0.5)
plt.title('Optimized Parabolic Nozzle Contour')
plt.xlabel('X-axis (m)')
plt.ylabel('Y-axis (m)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# Create 3D coordinates by revolving the 2D curve around the x-axis
theta = np.linspace(0, 2 * np.pi, 100)  # For revolution
X_3D = np.tile(x_bell, (100, 1)).T
Y_3D = np.outer(y_bell, np.cos(theta))
Z_3D = np.outer(y_bell, np.sin(theta))

# Plot the 3D revolved nozzle contour
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_3D, Y_3D, Z_3D, color='lightgreen', edgecolor='k', alpha=0.6)
ax.set_title("3D Revolved Nozzle Contour")
ax.set_xlabel("X-axis (m)")
ax.set_ylabel("Y-axis (m)")
ax.set_zlabel("Z-axis (m)")
plt.show()

# Plot convergence
plt.figure(figsize=(8, 6))
plt.plot(range(num_generations), convergence, label='Fitness Convergence')
plt.title('Convergence of Genetic Algorithm')
plt.xlabel('Generation')
plt.ylabel('Best Fitness Value')
plt.grid(True)
plt.legend()
plt.show()
