import time

from root.clustering import clustering
from root.generate_dict_time_series_NILEARN import generate_dict_time_series_NILEARN


def test_dict_NILEARN():
    # t = time.time()
    # IMAGES_FOLDER = "/home/eb/Desktop/stopsignal/"
    # CONFOUNDS = "/home/eb/Desktop/stopsignal_confounds/"
    # OUTPUT_FOLDER = "../bin/output/stopsignal2/"
    # generate_dict_time_series_NILEARN(IMAGES_FOLDER, OUTPUT_FOLDER, CONFOUNDS)
    # elapsed = time.time() - t
    # print("generate_dict_time_series2 - stopsignal: ", elapsed)
    # #generate_dict_time_series2 - stopsignal:  1294.1882195472717 -- 22 mins

    # t = time.time()
    # TIME_SERIES_FOLDER = "../bin/output/stopsignal2/time_series_dict/"
    # OUTPUT_FOLDER = "../bin/output/stopsignal2/"
    # clustering(TIME_SERIES_FOLDER, OUTPUT_FOLDER)
    # elapsed = time.time() - t
    # print("clustering2 - stopsignal: ", elapsed)
    # # clustering: 58.56087112426758 -- 59 secs

    t = time.time()
    IMAGES_FOLDER = "/home/eb/Desktop/bart/"
    CONFOUNDS = "/home/eb/Desktop/bart_confounds/"
    OUTPUT_FOLDER = "../bin/output/bart2/"
    generate_dict_time_series_NILEARN(IMAGES_FOLDER, OUTPUT_FOLDER, CONFOUNDS)
    elapsed = time.time() - t
    print("generate_dict_time_series2 - bart: ", elapsed)
    #generate_dict_time_series2 - bart:  2034.8166592121124


test_dict_NILEARN()