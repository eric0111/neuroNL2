import os

def create_folder(folder):
    try:
        os.mkdir(folder)
    except OSError as error:
        print(error)