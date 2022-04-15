import os

def load_files(IMAGES_FOLDER, CONFOUNDS_FOLDER):
    print("Loading images and confounds...")
    #LOAD IMAGES
    images_filenames = os.listdir(IMAGES_FOLDER)
    try:
        images_filenames.remove(".DS_Store")
    except Exception as e:
        print(e)
    images_filenames.sort()

    images_abs_paths = []
    for atlas in images_filenames:
        images_abs_paths.append(IMAGES_FOLDER + atlas)

    #LOAD CONFOUNDS
    confounds_files = os.listdir(CONFOUNDS_FOLDER)

    try:
        confounds_files.remove(".DS_Store")
    except Exception as e:
        print(e)
    confounds_files.sort()

    abs_confounds_files = []
    for file in confounds_files:
        abs_confounds_files.append(CONFOUNDS_FOLDER + file)

    return images_filenames, images_abs_paths, abs_confounds_files