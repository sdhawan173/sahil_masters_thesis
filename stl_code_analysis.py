import time
import numpy as np
import plotly.graph_objects as go


def print_time(time_value):
    if time_value < 60:
        print('--> Total Compute Time: {} seconds'.format(time_value))
    else:
        print('--> Total Compute Time: {} minutes, {} seconds'.format(int(time_value // 60), time_value % 60))


def func_timer(start, end=None):
    """
    computes and prints out the time taken for a function to run
    :param start: start time
    :param end: end time
    return: run time
    """
    if end is None:
        end = time.time()
    diff = end - start
    print_time(diff)
    return diff


def plotly_3d(x_points, y_points, z_points, i_val, j_val, k_val,
              show_mesh=True, show_scatter=True, show_connections=False, show_lines=True):
    layout = go.Layout(
        scene=dict(
            aspectmode='data',
            xaxis_title='X AXIS',
            yaxis_title='Y AXIS',
            zaxis_title='Z AXIS'
        )
    )
    fig = go.Figure(layout=layout)
    if show_mesh:
        fig.add_trace(
            go.Mesh3d(x=x_points,
                      y=y_points,
                      z=z_points,
                      i=i_val,
                      j=j_val,
                      k=k_val,
                      color='#7f7f7f',
                      opacity=0.375
                      )
        )
    if show_scatter:
        fig.add_trace(
            go.Scatter3d(
                x=x_points,
                y=y_points,
                z=z_points,
                mode='markers',
                marker=dict(
                    color=x_points,
                    colorscale='Rainbow',
                    size=3
                ),
                name='Points',
                text=[str(i) for i in range(len(x_points))]
            )
        )
    if show_connections:
        fig.add_trace(
            go.Scatter3d(
                x=x_points,
                y=y_points,
                z=z_points,
                mode='lines',
                line=dict(
                    color='#FF0000',
                    width=1,
                ),
                name='Connections',
                text=[str(i) for i in range(len(x_points))]
            )
        )
    if show_lines:
        # https://community.plotly.com/t/show-edges-of-the-mesh-in-a-mesh3d-plot/33614/3
        triangles = np.vstack((i_val, j_val, k_val)).T
        vertices = np.vstack((x_points, y_points, z_points)).T
        tri_points = vertices[triangles]
        x_line = []
        y_line = []
        z_line = []
        for T in tri_points:
            x_line.extend([T[k % 3][0] for k in range(4)] + [None])
            y_line.extend([T[k % 3][1] for k in range(4)] + [None])
            z_line.extend([T[k % 3][2] for k in range(4)] + [None])
        fig.add_trace(
            go.Scatter3d(
                x=x_line,
                y=y_line,
                z=z_line,
                mode='lines',
                line=dict(
                    color='#000000',
                    width=5
                ),
                name='Edges'
            )
        )
    return fig
