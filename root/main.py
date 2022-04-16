import time
from datetime import datetime

from clustering.rnn import rnn
from clustering.transformers import transformers
from root.clustering.svm import svm
from root.time_series_generator.components_extractor.extract_components import extract_components
from root.time_series_generator.extract_most_intense_regions import extract_most_intense_regions
from root.time_series_generator.load_files import load_files
from root.time_series_generator.generate_time_series import generate_time_series
import constants
from utils.create_folder import create_folder


# USER INPUT
FILES = "/home/eb/Desktop/stopsignal/"
CONFOUNDS = "/home/eb/Desktop/stopsignal_confounds/"
OUTPUT_FOLDER = "./bin/output/stopsignal/"
TIME_SERIES_FOLDER = "./bin/output/stopsignal/time_series/"
components_method = constants.METHOD_DICTIONARY_LEARNING

def main():
    try:
        create_folder(OUTPUT_FOLDER)
    except Exception as e:
        print(e)

    #time_series_generator
    t = time.time()
    images_filenames, images_abs_paths, abs_confounds_files = load_files(FILES, CONFOUNDS)
    components_img = extract_components(images_abs_paths, components_method)
    extractor = extract_most_intense_regions(components_img, OUTPUT_FOLDER)
    generate_time_series(images_filenames, images_abs_paths, extractor, TIME_SERIES_FOLDER, abs_confounds_files)
    elapsed = time.time() - t
    print("generate_dict_time_series - stopsignal: ", elapsed)

    #clustering
    t = time.time()
    svm(TIME_SERIES_FOLDER, OUTPUT_FOLDER)
    elapsed = time.time() - t
    print("clustering - stopsignal - svm: ", elapsed)

    t = time.time()
    rnn(TIME_SERIES_FOLDER, CNN=False, LSTM=False, GRU=True)
    elapsed = time.time() - t
    print("clustering - stopsignal - rnn: ", elapsed)

    t = time.time()
    transformers(TIME_SERIES_FOLDER)
    elapsed = time.time() - t
    print("clustering - stopsignal - transformers: ", elapsed)


main()