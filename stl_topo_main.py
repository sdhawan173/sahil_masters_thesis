import os
import time
from datetime import datetime
import stl_mesh_math as smm
import stl_code_analysis as sca
import ast_file_operations as afo
import stl_processing as stlp


def run_main_code(file_index, file_ext,
                  save_orig_plotly, show_orig_plotly,
                  save_meshpy_plotly, show_meshpy_plotly,
                  save_gudhi_persdia, show_gudhi_persdia, list_points_persdia):
    switch_string = 'pq10a100c'  # 'pDqa25'
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    total_start = time.time()
    file, file_path = stlp.choose_file(input_dir=input_directory,
                                       file_type=file_ext,
                                       file_index=file_index,
                                       show_list=False)
    if file_ext == '.ast':
        # afo.print_file(chosen_file_path)
        ast_tuple = afo.read_object3d(file_path=file_path,
                                      file_name=file)
        # afo.print_object3d(ast_tuple)
        afo.plotly_from_file(object_tuple=ast_tuple,
                             file_name=file,
                             save_html=save_orig_plotly,
                             file_time=time_stamp,
                             save_dir=output_directory,
                             show_plot=show_orig_plotly)
        # unique_vertices = afo.list_vertices(ast_tuple, unique=True)
        # print('Unique Vertices in:  \"{}\": {}'.format(chosen_file, len(unique_vertices)))
    tet_mesh = smm.meshpy_from_file(file_name=file,
                                    file_path=file_path,
                                    switch=switch_string,
                                    verbose=True)
    smm.plotly_from_meshpy(meshpy_mesh=tet_mesh,
                           file_name=file,
                           save_html=save_meshpy_plotly,
                           file_time=time_stamp,
                           save_dir=output_directory,
                           show_plot=show_meshpy_plotly)
    obj3d_complex, obj3d_smplx_tree = smm.create_gudhi_elements(meshpy_mesh=tet_mesh,
                                                                complex_type=complex_type_name)
    # for i in obj3d_smplx_tree.get_filtration():
    #     print(i)
    # input('wait 2')
    if complex_type_name == 'Alpha':
        obj3d_smplx_tree = smm.modify_alpha_complex(gudhi_complex=obj3d_complex,
                                                    gudhi_simplex_tree=obj3d_smplx_tree,
                                                    meshpy_mesh=tet_mesh,
                                                    complex_type=complex_type_name,
                                                    verbose=True)
    # for i in obj3d_smplx_tree.get_filtration():
    #     print(i)
    # input('wait 3')
    smm.plot_persdia_gudhi(gudhi_simplex_tree=obj3d_smplx_tree,
                           file_name=file,
                           show_legend=True,
                           save_plot=save_gudhi_persdia,
                           file_time=time_stamp,
                           save_dir=output_directory,
                           list_points=list_points_persdia,
                           show_plot=show_gudhi_persdia)
    print('\nTotal Run Time:')
    sca.func_timer(total_start)


extension = '.ast'
complex_type_name = 'VR'
input_directory = '{}/stl_files'.format(os.getcwd())
output_directory = '{}/code_output'.format(os.getcwd())

file_list = stlp.list_files(input_dir=input_directory, file_type=extension)
stlp.print_file_list(file_list=file_list, input_dir=input_directory)
chosen_file, chosen_file_path = stlp.choose_file(input_dir=input_directory, file_type=extension, show_list=False)
read_file = stlp.read_stl(chosen_file_path)
parsed_facets, vertex_coords = stlp.parse_stl_text(read_file)
print([item[-3:][1] for item in vertex_coords])
vertex_names, edge_names, face_names = stlp.create_simplex_names(parsed_facets, vertex_coords)
input('wait')

index_list = []
if len(index_list) == 0:
    index = int(input('Choose ' + extension + ' file by entering the corresponding number:\n'))
    index_list = [index]
for index in index_list:
    run_main_code(index, extension,
                  save_orig_plotly=False,
                  show_orig_plotly=True,
                  save_meshpy_plotly=False,
                  show_meshpy_plotly=True,
                  save_gudhi_persdia=False,
                  show_gudhi_persdia=True,
                  list_points_persdia=True)
