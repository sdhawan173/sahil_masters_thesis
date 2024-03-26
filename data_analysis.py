import time
from datetime import datetime
import numpy as np
import math
import plotly.graph_objects as go
import stl_mesh_math as smm
import file_operations as fo
import persdia_creator as pc


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


def list_vertices(object_tuple, unique=True):
    """
    Finds vertices/points in the 3d object
    :param object_tuple: list of all data extracted from .ast file
    :param unique: only lists unique vertices, default=True
    :return: list of points from .ast file
    """
    vertex_list = []
    for simplex in object_tuple:
        for vertex_num in range(3):
            vertex = [simplex[2][vertex_num][0],
                      simplex[2][vertex_num][1],
                      simplex[2][vertex_num][2]]
            if unique and not vertex_list or not vertex_list.__contains__(vertex):
                vertex_list.append(vertex)
            elif not unique:
                vertex_list.append(vertex)
    return vertex_list


def plotly_3d(x_points, y_points, z_points, i_val, j_val, k_val,
              x_camera=1.5, y_camera=1.5, z_camera=3,
              show_mesh=True, show_scatter=True, show_connections=False, show_lines=True, show_circumspheres=True):
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

    triangles = np.vstack((i_val, j_val, k_val)).T
    vertices = np.vstack((x_points, y_points, z_points)).T
    tri_points = vertices[triangles]

    if show_lines:
        # https://community.plotly.com/t/show-edges-of-the-mesh-in-a-mesh3d-plot/33614/3
        x_line = []
        y_line = []
        z_line = []
        for triangle in tri_points:
            x_line.extend([triangle[k % 3][0] for k in range(4)] + [None])
            y_line.extend([triangle[k % 3][1] for k in range(4)] + [None])
            z_line.extend([triangle[k % 3][2] for k in range(4)] + [None])
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

    if show_circumspheres:
        random_indices = []
        while len(random_indices) < 1:
            index = np.random.randint(len(tri_points))
            if index not in random_indices:
                if len(random_indices) == 0 or abs(index - random_indices[-1]) > 1:
                    random_indices.append(index)

        # Create a new list by selecting the datasets at the random indices
        selected_triangles = [tri_points[i] for i in random_indices]

        for triangle_points in selected_triangles:
            print(triangle_points)
            points = []
            for vertex in triangle_points:
                x, y, z = vertex
                points.append((x, y, z))
            x_points = []
            y_points = []
            z_points = []
            for point in points:
                x_points.append(point[0])
                y_points.append(point[1])
                z_points.append(point[2])

            # Extracting coordinates
            x1, y1, z1 = points[0]
            x2, y2, z2 = points[1]
            x3, y3, z3 = points[2]

            # Calculate the vectors representing two sides of the triangle
            vec1 = [x2 - x1, y2 - y1, z2 - z1]
            vec2 = [x3 - x1, y3 - y1, z3 - z1]

            # Calculate the cross product of the two vectors to find the normal vector of the plane
            normal = [
                vec1[1] * vec2[2] - vec1[2] * vec2[1],
                vec1[2] * vec2[0] - vec1[0] * vec2[2],
                vec1[0] * vec2[1] - vec1[1] * vec2[0]
            ]

            # Calculate the midpoints of the sides of the triangle
            midpoint1 = [(x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2]
            midpoint2 = [(x1 + x3) / 2, (y1 + y3) / 2, (z1 + z3) / 2]

            # Calculate the direction vector of the normal line passing through one midpoint
            dir_vec = [normal[1], -normal[0], 0]

            # Calculate the parameter t
            t = ((midpoint2[0] - midpoint1[0]) * normal[0] + (midpoint1[1] - midpoint2[1]) * normal[1]) / (
                        dir_vec[0] * normal[1] - dir_vec[1] * normal[0])

            # Calculate the circumcenter
            circumcenter = [
                midpoint1[0] + t * dir_vec[0],
                midpoint1[1] + t * dir_vec[1],
                midpoint1[2] + t * dir_vec[2]
            ]

            # Calculate the radius (distance from circumcenter to any point)
            radius = math.sqrt((circumcenter[0] - x1) ** 2 + (circumcenter[1] - y1) ** 2 + (circumcenter[2] - z1) ** 2)

            # Create a sphere meshgrid
            space_size = 30
            phi = np.linspace(0, np.pi, space_size)
            theta = np.linspace(0, 2 * np.pi, space_size)
            phi, theta = np.meshgrid(phi, theta)
            x_sphere = circumcenter[0] + radius * np.sin(phi) * np.cos(theta)
            y_sphere = circumcenter[1] + radius * np.sin(phi) * np.sin(theta)
            z_sphere = circumcenter[2] + radius * np.cos(phi)

            # Plot the sphere
            fig.add_trace(
                go.Surface(
                    x=x_sphere,
                    y=y_sphere,
                    z=z_sphere,
                    opacity=0.5,
                    colorscale='reds',
                    showscale=False
                )
            )
            fig.add_trace(
                go.Scatter3d(x=x_points, y=y_points, z=z_points, mode='markers', marker=dict(size=7, color='black')))

    # Update camera view
    if x_camera and y_camera and z_camera is not None:
        fig.update_layout(scene=dict(
            camera=dict(
                eye=dict(x=x_camera, y=y_camera, z=z_camera)
            )
        ))
    return fig


def print_object3d(object_tuple):
    """
    Prints all data associated with object tuple
    :param object_tuple: list of all data extracted from .ast file
    """

    print('Number of faces:         {}'.format(len(object_tuple)))
    object_unique = list_vertices(object_tuple)
    print('Unique points:')
    for point in range(len(object_unique)):
        print('{}'.format(object_unique[point]))
    print('')

    for simplex in object_tuple:
        print(simplex[0] + ':')
        print('   Normal: \n'
              '      x: {}\n'
              '      y: {}\n'
              '      z: {}'.format(simplex[1][0], simplex[1][1], simplex[1][2]))
        print('   Vertices:\n'
              '      vertex 1:\n'
              '         x: {}\n'
              '         y: {}\n'
              '         z: {}\n'
              '      vertex 2:\n'
              '         x: {}\n'
              '         y: {}\n'
              '         z: {}\n'
              '      vertex 3:\n'
              '         x: {}\n'
              '         y: {}\n'
              '         z: {}'.format(simplex[2][0][0], simplex[2][0][1], simplex[2][0][2],
                                      simplex[2][1][0], simplex[2][1][1], simplex[2][1][2],
                                      simplex[2][2][0], simplex[2][2][1], simplex[2][2][2]))


def plotly_from_ast(object_tuple, file_name=None,
                    save_html=False, file_time=None,
                    save_dir=None, show_plot=True):
    """
    Plots data extracted from .ast file using plotly
    :param object_tuple: list of all data extracted from .ast file
    :param file_name: String name of 3D object original file
    :param save_html: Boolean variable to save the html plotly file, default=False
    :param file_time: String time of when the main file was run
    :param save_dir: String of directory to save html file to
    :param show_plot: boolean variable to create pop-up plot window, default=True
    """
    if file_name:
        print('\nPlotting .ast Mesh of \"' + file_name + '\"...')

    x_points = []
    y_points = []
    z_points = []
    i_val = []
    j_val = []
    k_val = []
    for objectface in object_tuple:
        x_points.append(objectface[2][0][0])
        y_points.append(objectface[2][0][1])
        z_points.append(objectface[2][0][2])
        i_val.append(len(x_points) - 1)

        x_points.append(objectface[2][1][0])
        y_points.append(objectface[2][1][1])
        z_points.append(objectface[2][1][2])
        j_val.append(len(y_points) - 1)

        x_points.append(objectface[2][2][0])
        y_points.append(objectface[2][2][1])
        z_points.append(objectface[2][2][2])
        k_val.append(len(z_points) - 1)
    fig = plotly_3d(x_points, y_points, z_points,
                    i_val, j_val, k_val)

    if file_time and save_dir is not None and save_html:
        save_file_name = '{}/{}, ({}) ast plotly'.format(save_dir, file_time, file_name.split('.')[0])
        fig.write_html('{}.html'.format(save_file_name))
        print('Plotly of .ast HTML file saved to {}'.format(save_dir))
    if show_plot:
        fig.show()


def run_main_code(file_index, file_ext, input_dir, save_dir, meshpy_switch, max_dim, save_orig_plotly, show_orig_plotly,
                  save_meshpy_plotly, show_meshpy_plotly, save_persdia, list_points_persdia,
                  save_points_persdia, multi_run, final_run=False):
    """
    :param file_index: Int index of file in file list to be run
    :param file_ext: String extension of file to be run, only '.ast' files supported currently
    :param input_dir: String of directory to read file from
    :param save_dir: String of directory to save files
    :param meshpy_switch: String of switch to be used for meshpy mesh creation
    :param save_orig_plotly: Boolean to save the original 3D Plotly plot of the '.ast' file
    :param show_orig_plotly: Boolean to show the original 3D Plotly plot of the '.ast' file
    :param save_meshpy_plotly: Boolean to save the meshpy 3D Plotly of the meshed '.ast' file
    :param show_meshpy_plotly: Boolean to show the meshpy 3D Plotly of the meshed '.ast' file
    :param save_persdia: Boolean to save the persistence diagram of the meshed '.ast' file
    :param max_dim: maximum dimension of points to be shown
    :param list_points_persdia: Boolean to list all points in the persistence diagram
    :param save_points_persdia: Boolean to save all points in the persistence diagram to a '.txt' file
    :param multi_run:  Boolean variable for keeping multiple plots consistent
    :param max_dim: Value of highest dimension to plot
    :param final_run: Boolean to replace timestamp string with 'Final Run'
    return: value of maximum y_tick from persistence diagram to be stored in y_max_list for second run
    """

    # Track time it takes to run function
    total_start = time.time()

    # Create prefix for save file name
    time_stamp = 'Final Run'
    if not final_run:
        time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")

    # Get file name and file path of chosen file from file_index
    file_name, file_path = fo.choose_file(input_dir=input_dir, file_type=file_ext,
                                          file_index=file_index, show_list=False)

    # Run ast file operations code
    if file_ext == '.ast' and (show_orig_plotly or save_orig_plotly):
        # Print original file contents of '.ast' file
        # fo.print_file(file_path)

        # Parse '.ast' file data into tuple
        ast_tuple = fo.read_object3d(file_path=file_path, file_name=file_name)
        # Print contents of '.ast' tuple
        # print_object3d(ast_tuple)

        # Create 3D Plotly of '.ast' tuple data
        plotly_from_ast(object_tuple=ast_tuple, file_name=file_name, save_html=save_orig_plotly,
                        file_time=time_stamp, save_dir=save_dir, show_plot=show_orig_plotly)

    # Create meshpy mesh of '.ast' file
    tet_mesh = smm.meshpy_from_file(file_name=file_name, file_path=file_path, meshpy_switch=meshpy_switch, verbose=True)

    # Create 3D Plotly of meshpy mesh
    if show_meshpy_plotly or save_meshpy_plotly:
        smm.plotly_from_meshpy(meshpy_mesh=tet_mesh, file_name=file_name, save_html=save_meshpy_plotly,
                               file_time=time_stamp, save_dir=save_dir, show_plot=show_meshpy_plotly)

    input('wait')

    # Create and modify Alpha complex and simplex tree of meshpy mesh with gudhi
    obj3d_complex, obj3d_smplx_tree = smm.create_gudhi_elements(meshpy_mesh=tet_mesh)
    obj3d_smplx_tree = smm.modify_alpha_complex(gudhi_complex=obj3d_complex, gudhi_simplex_tree=obj3d_smplx_tree,
                                                meshpy_mesh=tet_mesh, verbose=True)

    # Get list of persistence points from simplex tree
    persistence_points = obj3d_smplx_tree.persistence()

    # Create persistence diagram
    y_max_value = pc.plot_persdia_main(persistence_points=persistence_points,
                                       file_name=file_name,
                                       max_dim=max_dim,
                                       save_plot=save_persdia,
                                       file_time=time_stamp,
                                       save_dir=save_dir,
                                       list_points=list_points_persdia,
                                       save_points=save_points_persdia,
                                       multi_run=multi_run)
    print('\nTotal Run Time:')
    func_timer(total_start)
    return y_max_value
