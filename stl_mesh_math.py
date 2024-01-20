import os
import time
import shutil
import numpy as np
from meshpy import tet
import gudhi
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
import stl_code_analysis as sca


def meshpy_switch_creator(file_name, file_path,
                          plc=True,
                          preserve=False,
                          coarsening=True,
                          max_vol_const=False,
                          quiet=False,
                          verbose=False):
    """
    Function to create meshpy switches based on documentation from tetgen.pdf
    :param file_name: file name with extension
    :param file_path: file path with extension
    :param plc: Tetrahedralizes a piecewise linear complex (PLC)
    :param preserve: Perserves the input surface mesh (does not modify it)
    :param coarsening: Mesh coarsening (to reduce the mesh elements)
    :param max_vol_const: Applies a maximum tetrahedron volume constraint
    :param quiet: No terminal output except errors
    :param verbose: Detailed information, more terminal output
    """
    meshpy_switch = ''
    if plc:
        meshpy_switch += 'p'
    if preserve:
        meshpy_switch += 'Y'
    if coarsening:
        meshpy_switch += 'R'
    if max_vol_const:
        meshpy_switch += 'a'
    if quiet:
        meshpy_switch += 'Q'
    if verbose:
        meshpy_switch += 'V'
    return meshpy_switch


def meshpy_from_file(file_name, file_path, switch,
                     verbose=True):
    """
    Uses meshpy to create a list of tetrahedrons in a
    mesh generated by meshpy
    :param file_name: file name with extension
    :param file_path: file path with extension
    :param switch: https://wias-berlin.de/software/tetgen/switches.html
    :param verbose: prints progress statements
    :return: tetrahedron mesh
    """
    print('\nCreating Tetrahedral Mesh with MeshPy ...')

    tet_options = tet.Options(switches=switch)
    meshinfo = tet.MeshInfo()

    dst_dir = None
    if file_name[-4:] == '.ast':
        src_dir = file_path
        temp_string = ' (temp STL conversion)'
        dst_dir = file_name[:-4] + temp_string + '.stl'
        shutil.copy(src_dir, dst_dir)
        meshinfo.load_stl(dst_dir)

    if file_name[-4:] == '.ply':
        # print(os.path.exists("ant.ply"))
        meshinfo.load_ply(file_path)

    object_mesh = tet.build(meshinfo, options=tet_options, verbose=verbose)

    if file_name[-4:] == '.ast':
        os.remove(dst_dir)

    if verbose:
        print('Number of MeshPy Mesh Points: {}'.format(len(list(object_mesh.points))))
        print('Number of MeshPy Mesh Edges:  {}'.format(len(list(object_mesh.edges))))
        print('Number of MeshPy Mesh Faces:  {}'.format(len(list(object_mesh.faces))))
        print('Number of MeshPy Mesh Tetra:  {}'.format(len(list(object_mesh.elements))))
    return object_mesh


def extract_meshpy_points(meshpy_mesh):
    """
    Extracts the x, y, and z points from each coordinate
    :param meshpy_mesh: Mesh generated by MeshPy
    :return: list of x, y, and z points
    """
    tetrahedron_points = list(meshpy_mesh.points)
    x_points = [point[0] for point in tetrahedron_points]
    y_points = [point[1] for point in tetrahedron_points]
    z_points = [point[2] for point in tetrahedron_points]
    return x_points, y_points, z_points


def plotly_from_meshpy(meshpy_mesh, file_name=None,
                       save_html=False, file_time=None,
                       save_dir=None, show_plot=True):
    """
    Plots the mesh created by MeshPy using Plotly
    :param meshpy_mesh: MeshPy mesh
    :param file_name: String name of 3D object original file
    :param save_html: Boolean variable to save the html plotly file, default=False
    :param file_time: String time of when the main file was run
    :param save_dir: String of directory to save html file to
    :param show_plot: boolean variable to create pop-up plot window, default=True
    """
    if file_name:
        print('\nPlotting MeshPy Mesh of \"' + file_name + '\"...')

    x_points, y_points, z_points = extract_meshpy_points(meshpy_mesh)
    tet_faces = list(meshpy_mesh.faces)
    i_val = [point_name[0] - 1 for point_name in tet_faces]
    j_val = [point_name[1] - 1 for point_name in tet_faces]
    k_val = [point_name[2] - 1 for point_name in tet_faces]

    fig = sca.plotly_3d(x_points, y_points, z_points,
                        i_val, j_val, k_val)

    if file_time and save_dir is not None and save_html:
        save_file_name = '{}/{}, ({}) meshpy plotly'.format(save_dir, file_time, file_name.split('.')[0])
        fig.write_html('{}.html'.format(save_file_name))
        print('Plotly MeshPy HTML file saved to {}.html'.format(save_dir))
    if show_plot:
        fig.show()


def create_gudhi_elements(meshpy_mesh, complex_type):
    """
    :param meshpy_mesh: MeshPy mesh
    :param complex_type: type of complex, 'VR' for vietoris-rips, 'Alpha' for alpha
    :return: gudhi_simplex_tree
    """
    gudhi_complex = None
    gudhi_simplex_tree = None
    print('\nCreating ' + complex_type + ' Complex and Simplex Tree ...')
    if complex_type == 'Alpha':
        # noinspection PyUnresolvedReferences
        gudhi_complex = gudhi.AlphaComplex(points=meshpy_mesh.points)
        gudhi_simplex_tree = gudhi_complex.create_simplex_tree()
    elif complex_type == 'VR':
        # noinspection PyUnresolvedReferences
        gudhi_complex = gudhi.RipsComplex(points=meshpy_mesh.points)
        gudhi_simplex_tree = gudhi_complex.create_simplex_tree(max_dimension=3)
    return gudhi_complex, gudhi_simplex_tree


def point_name_array(meshpy_mesh):
    """
    Create array of points names from meshpy tetrahedra to be concatenated for gudhi
    """
    point_array = []
    for tetrahedron in list(meshpy_mesh.elements):
        for point in tetrahedron:
            if not point_array.__contains__(point):
                point_array.append(point)
    point_array = sorted(point_array)
    # Adjust point_array values for 0-indexing
    point_array = [[point - 1] for point in point_array]
    return point_array


def face_name_array(meshpy_mesh):
    """
    Create array of face names from meshpy tetrahedra to be concatenated for gudhi
    """
    face_array = []
    for tetrahedron in list(meshpy_mesh.elements):
        tetrahedron = sorted(tetrahedron)
        for index in range(4):
            face_array.append([tetrahedron[0 - index] - 1,
                               tetrahedron[1 - index] - 1,
                               tetrahedron[2 - index] - 1])
    face_array = [sorted(face) for face in face_array]
    return face_array


def edge_name_array(face_array):
    """
    Create array of edge names from array of faces to be concatenated for gudhi
    """
    edge_array = []
    for face in face_array:
        for index in range(3):
            edge = sorted([face[0 - index], face[1 - index]])
            if not edge_array.__contains__(edge):
                edge_array.append(edge)
    edge_array = sorted([sorted(edge) for edge in edge_array])
    return edge_array


def extend_reformat_mesh(meshpy_mesh):
    """
    Concatenates item names of MeshPy mesh in order of
    points, edges, faces, and tetrahedrons
    :param meshpy_mesh: Mesh generated by MeshPy
    :return: A list matching the formatting of the original indices in
    the list generated by gudhi vietoris-rips complex.
    """
    print('Extending and reformatting MeshPy array to use with Gudhi ...')

    gudhi_ready_mesh_list = []
    point_array = point_name_array(meshpy_mesh)
    face_array = face_name_array(meshpy_mesh)
    edge_array = edge_name_array(face_array)

    gudhi_ready_mesh_list.extend(point_array)
    gudhi_ready_mesh_list.extend(edge_array)
    gudhi_ready_mesh_list.extend(face_array)
    gudhi_ready_mesh_list.extend([sorted([item - 1 for item in array]) for array in list(meshpy_mesh.elements)])
    return gudhi_ready_mesh_list


def modify_alpha_complex(gudhi_complex, gudhi_simplex_tree, meshpy_mesh,
                         complex_type, verbose=True):
    """
    fixes a gudhi complex
    :param gudhi_complex: gudhi complex element
    :param gudhi_simplex_tree: gudhi simplex tree element
    :param meshpy_mesh: MeshPy mesh
    :param complex_type: type of complex, 'VR' for vietoris-rips, 'Alpha' for alpha
    :param verbose: prints progress statements
    :return: complex, simplex tree from selected complex
     """
    mesh_points = np.array(meshpy_mesh.points)
    gudhi_complex_points = [gudhi_complex.get_point(i) for i in range(len(mesh_points))]
    mesh_to_complex = {
        i: np.argmin(
            [
                np.linalg.norm(mesh_points[i] - m) for m in gudhi_complex_points
            ]
        )
        for i in range(len(gudhi_complex_points))
    }
    new_total_mesh = []
    all_mesh_items = extend_reformat_mesh(meshpy_mesh)
    for x in all_mesh_items:
        new_total_mesh.append(sorted([mesh_to_complex[i] for i in x]))
    for mesh_item in new_total_mesh:
        gudhi_simplex_tree.insert(mesh_item)
    print('Reassigning filtration values based on existing mesh items ...')
    zero_counter = 0
    for filtered_item in tqdm(gudhi_simplex_tree.get_filtration()):
        if new_total_mesh.__contains__(filtered_item[0]):
            gudhi_simplex_tree.assign_filtration(filtered_item[0], 0)
            zero_counter += 1
    if verbose:
        print('{} Complex, Dim = {}'
              '\nvertices:  {}\nsimplices: {}'
              '\nreassigned {} mesh items'
              .format(complex_type, gudhi_simplex_tree.dimension(),
                      gudhi_simplex_tree.num_vertices(), gudhi_simplex_tree.num_simplices(),
                      str(zero_counter)))
        if zero_counter <= 25:
            print('Printing Reassigned ' + complex_type + ' Filtration Values:')
            for item in gudhi_simplex_tree.get_filtration():
                if all_mesh_items.__contains__(item[0]):
                    print(item)
    return gudhi_simplex_tree


def plot_persdia_gudhi(gudhi_simplex_tree, file_name, show_legend=True,
                       save_plot=True, file_time=None, save_dir=None,
                       list_points=False, show_plot=True):
    """
    plots the persistence diagram of the gudhi_simplex_tree generated by create_simplex.
    :param gudhi_simplex_tree: Simplex tree from selected complex
    :param file_name: String name of 3D object original file
    :param show_legend: Boolean variable to control legend on plot, default=True
    :param save_plot: Boolean variable to save the html plotly file, default=False
    :param file_time: String time of when the main file was run
    :param save_dir: String of directory to save html file to
    :param list_points: Boolean variable to list points of diagram, default=False
    :param show_plot: boolean variable to create pop-up plot window, default=True
    """
    print('\nPlotting Persistence Diagram ...')

    persistence = gudhi_simplex_tree.persistence()
    dimensions = list({item[0] for item in persistence})
    colormap = matplotlib.cm.Set1.colors
    fig, ax = plt.subplots()

    # noinspection PyUnresolvedReferences
    ax = gudhi.plot_persistence_diagram(persistence,
                                        legend=show_legend,
                                        axes=ax)
    ax.set_title('Persistence Diagram of \"{}\"'.format(file_name))
    ax.legend(handles=[
        mpatches.Patch(
            color=colormap[dim],
            label=r'$H_{}$'.format(str(dim))) for dim in dimensions
    ]
    )

    unique_indices = [persistence.index(point) for point in set(persistence)]
    frequency = []
    for index in range(len(persistence)):
        point_count = 0
        if unique_indices.__contains__(index):
            point_count = persistence.count(persistence[index])
        if point_count > 1:
            append_value = str(point_count)
        else:
            append_value = ''
        frequency.append(append_value)

    for index, text in enumerate(frequency):
        ax.annotate(text,
                    (persistence[index][1][0], persistence[index][1][1]),
                    xytext=(-5, 3),
                    textcoords='offset points',
                    fontsize=12)

    if list_points:
        for item in gudhi_simplex_tree.persistence():
            print(item)
    if file_time and save_dir is not None and save_plot:
        save_file_name = '{}/{}, ({}) persdia'.format(save_dir, file_time, file_name.split('.')[0])
        plt.savefig('{}.svg'.format(save_file_name))
        print('\nPersistence Diagram saved to {}.svg'.format(save_dir))
    if show_plot:
        plt.show()
