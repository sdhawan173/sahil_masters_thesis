import os
import stl
from stl import mesh


def search_filetype(file_string, dir_string=os.getcwd()):
    """
    searches a directory, with the current working directory as default, for a given filetype.
    :param dir_string: string of directory to search
    :param file_string: string of filetype, input as a string in the format: '.type'
    :return: list of file names with extensions that match search term
    """
    file_list = []

    # Run through list and add files with .ast extension to ast_list
    for list_item in os.listdir(dir_string):
        if list_item.__contains__(file_string):
            file_list.append(list_item)
    return sorted(file_list, key=str.casefold)


def print_file_list(file_list, input_dir):
    print('Listing files in: {}'.format(input_dir))
    for file_count in range(len(file_list)):
        if file_count < 10:
            print('0' + str(file_count) + ': ' + file_list[file_count])
        else:
            print(str(file_count) + ': ' + file_list[file_count])


def list_files(input_dir, file_type):
    """
    Lists files in a directory with specific file type
    :param input_dir: String of directory to read in files from
    :param file_type: file type in format '.ext'
    :return: String list of files
    """
    file_list = search_filetype(file_type, dir_string=input_dir)
    return file_list


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
    file_list = list_files(input_dir, file_type)
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


def parse_stl_text(read_file):
    """
    converts facet data from ast file to array and stores stl data to two tuples:
    parsed_file contains facet data in the following format per facet:
        (face_count, normal, vertex1 coords, vertex2 coords, vertex3 coords)
    vertex_count contains data in the following format per vertex:
        (vertex_count, vertex coords)
    """
    facet_data = []
    vertex_coords = []
    temp_face = []
    vertex_count = 0
    face_count = 0
    normal_vector = None
    for line in read_file:
        if line.__contains__('facet normal'):
            normal_vector = [float(_) for _ in line.split(' ')[-3:]]
        if line.__contains__('vertex'):
            vertex = [float(_) for _ in line.split(' ')[-3:]]
            temp_face.append(vertex)
            if not vertex_coords or not [item[1] for item in vertex_coords].__contains__(vertex):
                vertex_coords.append(([vertex_count], vertex))
                vertex_count += 1
        if line.__contains__('endfacet'):
            facet_data.append((face_count,
                               normal_vector,
                               temp_face[0],
                               temp_face[1],
                               temp_face[2]))
            face_count += 1
            temp_face = []
    return facet_data, vertex_coords


def create_simplex_names(parsed_file, vertex_coords):
    """
    Assigns labels of point names to edges and faces in original file data
    """
    vertex_names = []
    face_names = []
    edge_names = []
    for face in parsed_file:
        temp_face = []
        for point_index in range(2, 5):
            point_name = [item[-3:][1] for item in vertex_coords].index(face[point_index])
            temp_face.append(point_name)
        face_names.append(temp_face)
    for face in face_names:
        for index in range(3):
            edge = sorted([face[0 - index], face[1 - index]])
            if not edge_names.__contains__(edge):
                edge_names.append(edge)
    for i in vertex_names:
        vertex_names.append(i[0])
        print(vertex_names[-1])
    for i in edge_names:
        print(i)
    for i in face_names:
        print(i)
    return vertex_names, edge_names, face_names
