import os


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


def initialize(input_dir, file_type, file_index=None, show_list=True):
    """
    Runs initial starting blocks of code
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
        else:
            file_index = int(file_index)
    file = file_list[file_index]
    file_path = input_dir + '/' + file
    return file, file_path
