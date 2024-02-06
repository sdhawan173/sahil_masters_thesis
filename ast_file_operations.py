import networkx as nx
import data_analysis as sca


def print_file(file_path):
    """
    Prints the contents of a text file
    :param file_path: location of file ending in file name and extension
    :return:
    """
    file = open(file_path, 'r')
    print(file.read())
    file.close()


def rm_newline(string):
    """
    removes the newline characters in a string
    :param string: string with or with out newline characters
    :return: string without newline characters
    """
    if string.__contains__('\n'):
        return string.replace('\n', '')
    else:
        return string


def read_object3d(file_path, file_name):
    """
    Reads a .ast file from a string file path and creates a tuple from the 3d object
    :param file_path: file path string
    :param file_name: Name of 3D object original file
    :return: tuple with 3d object information
    """
    print('\nReading in data from \"' + file_name + '\" and storing to tuple ...')

    # open file
    opened_file = open(file_path, 'r')
    # identifiers to search for information in the document
    begin_facet_string = 'facet normal'
    endof_facet_string = 'endfacet'
    begin_vertex_string = 'outer loop'
    endof_vertex_string = 'endloop'
    # create empty tuples and triangle counter
    object_tuple = []
    triangle_vertices = []
    triangle_count = 0
    for line in opened_file:
        # print('Scanning Triangle {}...'.format(triangle_count))
        # Remove '\n' from end of string
        line = rm_newline(line)
        if line.__contains__(begin_facet_string):
            object_tuple.append([])
            # Extract x, y, and z normal coordinates from facet data
            normal_coords = line.split(' ')[4:]
            for item in range(len(normal_coords)):
                normal_coords[item] = float(normal_coords[item])
                # Add face name and normal coordinates to faces_tuple
            object_tuple[triangle_count].append('Face ' + str(triangle_count))
            object_tuple[triangle_count].append(normal_coords)
        if line.__contains__(begin_vertex_string):
            triangle_vertices = []
        if line.__contains__('vertex'):
            vertex_coords = line.split(' ')[7:]
            for item in range(len(vertex_coords)):
                vertex_coords[item] = float(vertex_coords[item])
            triangle_vertices.append(vertex_coords)
        if line.__contains__(endof_vertex_string):
            object_tuple[triangle_count].append(triangle_vertices)
        if line.__contains__(endof_facet_string):
            triangle_count += 1
    opened_file.close()
    return object_tuple


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


def extract_xyz(object_tuple_points):
    """
    returns lists of x, y, and z points from tuple of points
    :param object_tuple_points: tuple of points
    :return: x, y, ancd z points
    """
    x_values = [object_tuple_points[coord][1][0] for coord in range(len(object_tuple_points))]
    y_values = [object_tuple_points[coord][1][1] for coord in range(len(object_tuple_points))]
    z_values = [object_tuple_points[coord][1][2] for coord in range(len(object_tuple_points))]
    return x_values, y_values, z_values


def plotly_from_file(object_tuple, file_name=None,
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
    fig = sca.plotly_3d(x_points, y_points, z_points,
                        i_val, j_val, k_val)

    if file_time and save_dir is not None and save_html:
        save_file_name = '{}/{}, ({}) ast plotly'.format(save_dir, file_time, file_name.split('.')[0])
        fig.write_html('{}.html'.format(save_file_name))
        print('Plotly of .ast HTML file saved to {}.html'.format(save_dir))
    if show_plot:
        fig.show()


def create_nodes(object_tuple, object_graph, verbose=False):
    """
    Creates nodes for a networkx graph from object tuple
    :param object_tuple: list of all data extracted from .ast file
    :param object_graph: empty networkx graph
    :param verbose: prints progress statements
    :return: networkx graph
    """
    if verbose:
        print('\nCreating nodes ...')
    object_points = list_vertices(object_tuple)
    # print('Adding Nodes to Graph ...')
    for object_face in object_tuple:
        # print('\n\nIteration: {}'.format(count))
        # print('{}, {}'.format(face[0], face[2]))
        for point in object_points:
            # print('search term: {}'.format(point[1]))
            # print('search against: {}'.format(face[2]))
            for point_index in range(3):
                if [object_face[2][point_index]].__contains__(point[1]):
                    # print('MATCH FOUND')
                    # print('MATCH:\nsearch term: {}\nresult: {}\n'.format(point[1], face[2][point_index]))
                    object_graph.add_node(point[0])
    return object_graph


def find_triangles(object_tuple, object_points=None, verbose=False):
    """
    finds all triangles in object_tuple
    :param object_tuple: list of all data extracted from .ast file
    :param object_points: a list of unique xyz coordinates in the object
    :param verbose: prints progress statements
    :return: list of triangles in object
    """
    if verbose:
        print('\nFinding triangles ...')
    object_triangle_list = []
    if object_points is None:
        object_points = list_vertices(object_tuple)
    object_points_names = [object_points[name][0] for name in range(len(object_points))]
    object_points_coords = [object_points[coord][1] for coord in range(len(object_points))]
    for object_face in object_tuple:
        found_triangle = []
        for vertex in object_face[2]:
            name_index = object_points_coords.index(vertex)
            found_triangle.append(object_points_names[name_index])
        object_triangle_list.append(found_triangle)
        # print('{}: {}'.format(object_face[0], found_triangle))
    object_triangle_list = [sorted(object_triangle) for object_triangle in object_triangle_list]
    return object_triangle_list


def create_edges(object_tuple, object_graph, object_points=None, object_triangle_list=None, verbose=False):
    """
    Creates edges for a networkx graph from object tuple
    :param object_tuple: list of all data extracted from .ast file
    :param object_graph: list of all nodes
    :param object_points: a list of unique xyz coordinates in the object
    :param object_triangle_list: list of all triangles in object
    :param verbose: prints progress statements
    :return: networkx graph
    """
    if verbose:
        print('\nCreating edges ...')
    edge_list = []
    if object_triangle_list is None:
        object_triangle_list = find_triangles(object_tuple, object_points=object_points)
    for object_triangle in object_triangle_list:
        for index in range(3):
            edge_list.append([object_triangle[index], object_triangle[index - 1]])
    for object_edge in edge_list:
        for other_edge in edge_list:
            if other_edge == [object_edge[1], object_edge[0]]:
                edge_list.remove(other_edge)
    for object_edge in edge_list:
        object_graph.add_edge(object_edge[0], object_edge[1])
    return object_graph


def create_graph(object_tuple, verbose=False):
    """
    Creates a networkx graph from .ast file information
    :param object_tuple: list of all data extracted from .ast file
    :param verbose: prints progress statements
    :return: networkx graph of 3D object
    """
    object_graph = nx.Graph()
    graph = create_nodes(object_tuple, object_graph, verbose=verbose)
    graph_triangles = find_triangles(object_tuple, verbose=verbose)
    graph = create_edges(object_tuple, graph, object_triangle_list=graph_triangles, verbose=verbose)
    if verbose:
        print('\nNumber of nodes in Graph: {}'.format(len(list(graph.nodes))))
        print('Number of tri in Graph:   {}'.format(len(graph_triangles)))
        print('Number of edges in Graph: {}\n'.format(len(list(graph.edges))))
        print(nx.is_connected(graph))
        print(nx.number_connected_components(graph))
    return graph
