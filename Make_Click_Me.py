__author__ = 'Brian M Anderson'
# Created on 6/29/2020

from Workshop_Modules.TF_2.Merge_Notebooks import merge_notebooks_function, os


def main():
    note_books_paths = os.path.join('.','Workshop_Modules','TF_2')
    notebooks = [os.path.join(note_books_paths, 'Download_Data_Template.ipynb'),
                 os.path.join(note_books_paths, 'DeepBox.ipynb'),
                 os.path.join(note_books_paths, 'Data_Curation.ipynb'),
                 os.path.join(note_books_paths, 'Liver_Model.ipynb')]
    merge_notebooks_function(path_to_notebooks=note_books_paths, notebook_paths=notebooks,
                             notebook_output_path=os.path.join('.', 'Click_Me.ipynb'))
    return None


if __name__ == '__main__':
    main()
