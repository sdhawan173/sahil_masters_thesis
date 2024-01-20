import os
import stl
from stl import mesh
import stl_code_analysis as sca


def file_handler(file_path):
    """
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
    stores stl data to two tuples:
    parsed_file contains data in the following format per face:
        (face_count, normal, vertex1 coords, vertex2 coords, vertex3 coords)
    vertex_count contains data in the following format per vertex:
        (vertex_count, vertex coords)
    """
    for line in read_file:
        print(line)
    parsed_file = []
    vertex_list = []
    temp_face = []
    vertex_count = 0
    face_count = 0
    normal = None
    for line in read_file:
        if line.__contains__('facet normal'):
            normal = [float(_) for _ in line.split(' ')[-3:]]
        if line.__contains__('vertex'):
            vertex = [float(_) for _ in line.split(' ')[-3:]]
            temp_face.append(vertex)
            if not vertex_list or not [item[1] for item in vertex_list].__contains__(vertex):
                vertex_list.append(([vertex_count], vertex))
                vertex_count += 1
        if line.__contains__('endfacet'):
            parsed_file.append((face_count,
                                normal,
                                temp_face[0],
                                temp_face[1],
                                temp_face[2]))
            face_count += 1
            temp_face = []
    for i in parsed_file:
        print(i)
    for i in vertex_list:
        print(i)
    return parsed_file, vertex_list


def create_labels(parsed_file, vertex_list):
    """
    Assigns labels of point names to edges and faces in original file data
    """
    face_list = []
    edge_list = []
    print('hi')
    for face in parsed_file:
        temp_face = []
        for point_index in range(2, 5):
            point_name = [item[-3:][1] for item in vertex_list].index(face[point_index])
            temp_face.append(point_name)
        face_list.append(temp_face)
    for face in face_list:
        for index in range(3):
            edge = sorted([face[0 - index], face[1 - index]])
            if not edge_list.__contains__(edge):
                edge_list.append(edge)
    for i in vertex_list:
        print(i[0])
    for i in edge_list:
        print(i)
    for i in face_list:
        print(i)
    x_points = [i[1][0] for i in vertex_list]
    y_points = [i[1][1] for i in vertex_list]
    z_points = [i[1][2] for i in vertex_list]
    i_val = [point_name[0] for point_name in face_list]
    j_val = [point_name[1] for point_name in face_list]
    k_val = [point_name[2] for point_name in face_list]
    fig = sca.plotly_3d(x_points, y_points, z_points, i_val, j_val, k_val)
    fig.show()
    input('wait')
