import numpy
from root.utils.create_folder import create_folder

def generate_time_series(images_filenames, images_abs_paths, extractor, TIME_SERIES_FOLDER, abs_confounds_files):
    #CREATE OUTPUT FOLDER
    try:
        create_folder(TIME_SERIES_FOLDER)
    except Exception as e:
        print(e)

    # GENERATE AND SAVE TIME SERIES
    print("Generating time-series...")
    for filename, image, confounds in zip(images_filenames, images_abs_paths, abs_confounds_files):
        # call transform from RegionExtractor object to extract timeseries signals
        print(filename)
        data = numpy.genfromtxt(fname=confounds, delimiter="\t", skip_header=1, filling_values=1)
        confounds = numpy.nan_to_num(data, copy=True)
        timeseries_each_subject = extractor.transform(image, confounds=confounds)
        numpy.save(TIME_SERIES_FOLDER + filename.split("/")[-1].split(".nii")[0] + ".npy",
                   timeseries_each_subject)