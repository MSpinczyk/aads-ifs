import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Define bounds and resolution
x_min, x_max, y_min, y_max = (-1, 1, -1, 1)
resolution = 200

# Create the grid
x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, resolution),
                             np.linspace(y_min, y_max, resolution))

grid_points = np.column_stack((x_grid.flatten(), y_grid.flatten()))


def find_closest_point(point, grid_points):
    distances = cdist([point], grid_points, metric='euclidean')
    closest_index = np.argmin(distances)
    return grid_points[closest_index]

target_points = np.random.random([500, 2])
attractor_points = np.random.random([500, 2])
# Example for target points
target_closest_points = [find_closest_point(target_point, grid_points) for target_point in target_points]

# Example for attractor points
attractor_closest_points = [find_closest_point(attractor_point, grid_points) for attractor_point in attractor_points]

coincident_points = [point for point in attractor_closest_points if any(np.array_equal(point, target_point) for target_point in target_closest_points)]
outside_points = [point for point in attractor_closest_points if not any(np.array_equal(point, target_point) for target_point in target_closest_points)]


# Scatter plot of target points
plt.scatter(*zip(*target_points), color='blue', label='Target Points')

# Scatter plot of attractor points
plt.scatter(*zip(*attractor_points), color='red', label='Attractor Points')

# Scatter plot of coincident points
plt.scatter(*zip(*coincident_points), color='green', label='Coincident Points')

# Scatter plot of outside points
plt.scatter(*zip(*outside_points), color='orange', label='Outside Points')

# Plot the grid
plt.scatter(*zip(*grid_points), color='gray', s=1, alpha=0.2, marker='.')

plt.legend()
plt.show()
