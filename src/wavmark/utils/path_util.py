import pathlib
import os


def mk_parent_dir_if_necessary(img_save_path):
    folder = pathlib.Path(img_save_path).parent
    if not os.path.exists(folder):
        os.makedirs(folder)


def mk_dir_if_necessary(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
