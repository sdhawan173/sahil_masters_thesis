import time
from datetime import datetime
import numpy as np
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
              x_camera=None, y_camera=None, z_camera=None,
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
                  save_meshpy_plotly, show_meshpy_plotly, complex_type, save_persdia, list_points_persdia,
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
    :param complex_type: String of complex to create, only 'Alpha' supported currently
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

    # Create and modify Alpha complex and simplex tree of meshpy mesh with gudhi
    obj3d_complex, obj3d_smplx_tree = smm.create_gudhi_elements(meshpy_mesh=tet_mesh, complex_type=complex_type)
    obj3d_smplx_tree = smm.modify_alpha_complex(gudhi_complex=obj3d_complex, gudhi_simplex_tree=obj3d_smplx_tree,
                                                meshpy_mesh=tet_mesh, complex_type=complex_type, verbose=True)

    # Get list of persistence points from simplex tree
    persistence_points = obj3d_smplx_tree.persistence()

    # Create persistence diagram
    y_max_value = pc.plot_persdia_main(persistence_points=persistence_points,
                                       file_name=file_name,
                                       meshpy_switch=meshpy_switch,
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
