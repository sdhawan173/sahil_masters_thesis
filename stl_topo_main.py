import os
import file_operations as fo
import data_analysis as da
import persdia_creator as pc

file_ext = '.ast'
meshpy_switch = 'pq'  # 'pDq'  # 'pq10a100c'  # 'pDqa25'
max_dim = 1
input_dir = '{}/stl_files'.format(os.getcwd())
save_dir = '{}/code_output'.format(os.getcwd())
file_list = fo.search_filetype(file_ext=file_ext, dir_string=input_dir)
fo.print_file_list(file_list=file_list, input_dir=input_dir)

# Lists of indices for final run 3D object examples
equil_tri = list(reversed(range(10, 19)))  # equilateral triangle triangle hole
rect_pris = list(reversed(range(22, 34)))  # rect prism ring
two_cubes = list(reversed(range(47, 53)))  # two cubes, three pockets each
# run_list = [equil_tri, rect_pris, two_cubes]
run_list = [[2]]

# chosen_list = []
# # If no list of indices is provided, choose one
# if len(chosen_list) == 0:
#     index = int(input('Choose ' + file_ext + ' file by entering the corresponding number:\n'))
#     run_list = [[index]]

for index_list in run_list:
    multi_run = False
    for index in index_list:
        if len(index_list) > 1:
            multi_run = True
        multi_run = True
        da.run_main_code(index, file_ext, input_dir, save_dir, meshpy_switch, max_dim,
                         save_orig_plotly=False,
                         show_orig_plotly=False,
                         save_meshpy_plotly=True,
                         show_meshpy_plotly=True,
                         save_persdia=False,
                         save_points_persdia=True,
                         list_points_persdia=False,
                         final_run=True,
                         multi_run=multi_run
                         )
    if multi_run:
        # Load all save states into one list
        all_save_states = []
        for index in index_list:
            file_name, file_path = fo.choose_file(input_dir=input_dir, file_type=file_ext,
                                                  file_index=index, show_list=False)
            save_state = pc.load_plot_state(save_dir, file_name)
            all_save_states.append(save_state)

        # Iterate over all save states to get max y tick values for consistent persistence diagrams
        all_y_max_orig = []
        all_y_max_tick = []
        all_y_inf_tick = []
        for iterate_index, file_index in enumerate(index_list):
            all_y_max_orig.append(all_save_states[iterate_index][1])
            all_y_max_tick.append(all_save_states[iterate_index][2])
            all_y_inf_tick.append(all_save_states[iterate_index][3])
        total_y_max_orig = max(all_y_max_orig)
        total_y_max_tick = max(all_y_max_tick)
        total_y_inf_tick = max(all_y_inf_tick)

        for iterate_index, file_index in enumerate(index_list):
            (persistence_points, y_max_orig, y_max_tick, y_inf_tick, inf_bool, save_dir, file_time,
             file_name, save_plot, save_points, list_points) = all_save_states[iterate_index]

            pc.plot_persdia_main(persistence_points, file_name, max_dim,
                                 save_dir=save_dir,
                                 file_time=file_time,
                                 save_plot=save_plot,
                                 save_points=False,
                                 list_points=False,
                                 multi_run=False,
                                 override_orig=total_y_max_orig,
                                 y_max_tick=total_y_max_tick,
                                 y_inf_tick=total_y_inf_tick,
                                 all_y_inf_tick=all_y_inf_tick)
