import os
import time
from datetime import datetime
import stl_file_processing as sfp
import stl_mesh_math as smm
import stl_code_analysis as sca
import ast_file_operations as afo
import void_inclusive_mesh as vim


def run_main_code(file_index,
                  save_orig_plotly, show_orig_plotly,
                  save_meshpy_plotly, show_meshpy_plotly,
                  save_gudhi_persdia, show_gudhi_persdia, list_points_persdia):
    switch_string = 'p'  # 'pq10a100c'  # 'pDqa25'
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    total_start = time.time()
    chosen_file, chosen_file_path = sfp.initialize(input_dir=input_directory,
                                                   file_type=file_ext,
                                                   file_index=file_index,
                                                   show_list=False)
    if file_ext == '.ast':
        # afo.print_file(chosen_file_path)
        ast_tuple = afo.read_object3d(file_path=chosen_file_path,
                                      file_name=chosen_file)
        # afo.print_object3d(ast_tuple)
        afo.plotly_from_file(object_tuple=ast_tuple,
                             file_name=chosen_file,
                             save_html=save_orig_plotly,
                             file_time=time_stamp,
                             save_dir=output_directory,
                             show_plot=show_orig_plotly)
        # unique_vertices = afo.list_vertices(ast_tuple, unique=True)
        # print('Unique Vertices in:  \"{}\": {}'.format(chosen_file, len(unique_vertices)))
    tet_mesh = smm.meshpy_from_file(file_name=chosen_file,
                                    file_path=chosen_file_path,
                                    switch=switch_string,
                                    verbose=True)
    smm.plotly_from_meshpy(meshpy_mesh=tet_mesh,
                           file_name=chosen_file,
                           save_html=save_meshpy_plotly,
                           file_time=time_stamp,
                           save_dir=output_directory,
                           show_plot=show_meshpy_plotly)
    obj3d_complex, obj3d_smplx_tree = smm.create_gudhi_elements(meshpy_mesh=tet_mesh,
                                                                complex_type=complex_type_name)
    obj3d_smplx_tree = smm.modify_gudhi_elements(gudhi_complex=obj3d_complex,
                                                 gudhi_simplex_tree=obj3d_smplx_tree,
                                                 meshpy_mesh=tet_mesh,
                                                 complex_type=complex_type_name,
                                                 verbose=True)
    smm.plot_persdia_gudhi(gudhi_simplex_tree=obj3d_smplx_tree,
                           file_name=chosen_file,
                           show_legend=True,
                           save_plot=save_gudhi_persdia,
                           file_time=time_stamp,
                           save_dir=output_directory,
                           list_points=list_points_persdia,
                           show_plot=show_gudhi_persdia)
    print('\nTotal Run Time:')
    total_compute_time = sca.func_timer(total_start)


file_ext = '.ast'
complex_type_name = 'Alpha'
input_directory = '{}/stl_files'.format(os.getcwd())
output_directory = '{}/code_output'.format(os.getcwd())

file_list = sfp.list_files(input_dir=input_directory,
                           file_type=file_ext)
sfp.print_file_list(file_list=file_list,
                    input_dir=input_directory)


# test = vim.file_handler(chosen_file_path)
# parse, vector = vim.parse_stl_text(test)
# vim.create_labels(parse, vector)

index_list = []
for index in index_list:
    run_main_code(index,
                  save_orig_plotly=False,
                  show_orig_plotly=False,
                  save_meshpy_plotly=False,
                  show_meshpy_plotly=False,
                  save_gudhi_persdia=False,
                  show_gudhi_persdia=False,
                  list_points_persdia=False)
