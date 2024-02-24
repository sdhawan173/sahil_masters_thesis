import os
import stl
from stl import mesh


def search_filetype(file_ext, dir_string=os.getcwd()):
    """
    searches a directory, with the current working directory as default, for a given filetype.
    :param dir_string: string of directory to search
    :param file_ext: string of filetype, input as a string in the format: '.type'
    :return: list of file names with extensions that match search term
    """
    file_list = []

    # Run through list and add files with .ast extension to ast_list
    for list_item in os.listdir(dir_string):
        if list_item.__contains__(file_ext):
            file_list.append(list_item)
    return sorted(file_list, key=str.casefold)


def print_file_list(file_list, input_dir):
    print('Listing files in: {}'.format(input_dir))
    for file_count in range(len(file_list)):
        if file_count < 10:
            print('0' + str(file_count) + ': ' + file_list[file_count])
        else:
            print(str(file_count) + ': ' + file_list[file_count])


def choose_file(input_dir, file_type, file_index=None, show_list=True):
    """
    gets file name and file path of selected file from list
    :param input_dir: String of directory to read in files from
    :param file_type: file type in format '.ext'
    :param file_index: index of file to analyze from list_files()
    :param show_list: Boolean variable to show list, default=True
    :return:
    """
    # List files with file_type in directory
    file_list = search_filetype(file_type, input_dir)
    if show_list:
        print_file_list(file_list, input_dir)
    if file_index is None:
        # Select file, print output to confirm choice
        file_index = input('Choose ' + file_type + ' file by entering the corresponding number:\n')
        if int(file_index) < 10 and len(file_index) >= 2:
            file_index = int(file_index[-1])
        if int(file_index) > 100 and len(file_index) >= 3:
            file_index = int(file_index)
        else:
            file_index = int(file_index)
    file = file_list[file_index]
    file_path = input_dir + '/' + file
    return file, file_path


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


def read_stl(file_path):
    """
    reads stl file and stores to array
    :param file_path: file path of ast or stl file
    """
    opened_file = None
    read_file = []
    if file_path[-4:] == '.stl':
        # from https://stackoverflow.com/questions/53608038/stl-file-to-a-readable-text-file
        file_path = mesh.Mesh.from_file(file_path)
        file_path.save('temp.stl', mode=stl.Mode.ASCII)
        opened_file = open('temp.stl', 'r')
        os.remove('temp.stl')
    elif file_path[-4:] == '.ast':
        opened_file = open(file_path, 'r')
    for line in opened_file:
        if line.__contains__('\n'):
            line = line.replace('\n', '')
            read_file.append(line)
    opened_file.close()
    return read_file
