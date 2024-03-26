import plotly.graph_objects as go
import numpy as np

# Function to generate points on the sphere boundary from vertices
def generate_points_from_vertices(vertices):
    points = []
    for vertex in vertices:
        x, y, z = vertex
        points.append((x, y, z))
    return points

# Vertices of the points on the boundary
vertices = [
    (5.000000, 0.000000, 0.816497),
    (5.500000, -0.288675, 0.000000),
    (5.000000, 0.577350, 0.000000)
]

# Generate points from vertices
boundary_points = generate_points_from_vertices(vertices)
print(boundary_points)

# Calculate the center of the sphere using the average of the points
center = np.mean(boundary_points, axis=0)

# Calculate the radius of the sphere
radius = np.max(np.linalg.norm(np.array(boundary_points) - center, axis=1))

# Create a sphere meshgrid
phi = np.linspace(0, np.pi, 30)
theta = np.linspace(0, 2 * np.pi, 30)
phi, theta = np.meshgrid(phi, theta)
x_sphere = center[0] + radius * np.sin(phi) * np.cos(theta)
y_sphere = center[1] + radius * np.sin(phi) * np.sin(theta)
z_sphere = center[2] + radius * np.cos(phi)

# Plot the sphere
fig = go.Figure(data=[
    go.Surface(
        x=x_sphere,
        y=y_sphere,
        z=z_sphere,
        opacity=0.5,
        colorscale='reds',
        showscale=False)])

# Plot the points on the boundary
x_points, y_points, z_points = zip(*boundary_points)
fig.add_trace(go.Scatter3d(x=x_points, y=y_points, z=z_points, mode='markers', marker=dict(size=5, color='blue')))

# Set layout
fig.update_layout(scene=dict(aspectmode='data'))

# Show plot
fig.show()
