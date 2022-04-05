import os
import nilearn
import numpy
import numpy as np
from matplotlib import pyplot as plt
from nilearn import plotting
from nilearn.decomposition import CanICA, DictLearning
from root.utils.create_folder import create_folder
#https://nilearn.github.io/auto_examples/03_connectivity/plot_extract_regions_dictlearning_maps.html


def load_paths(IMAGES_FOLDER, OUTPUT_FOLDER, CONFOUNDS):
    # LOAD PATHS
    create_folder(OUTPUT_FOLDER)
    create_folder(OUTPUT_FOLDER + "time_series_dict/")

    print("Loading images and confounds...")
    images_filenames = os.listdir(IMAGES_FOLDER)
    try:
        images_filenames.remove(".DS_Store")
    except Exception as e:
        print(e)
    images_filenames.sort()

    images_abs_paths = []
    for atlas in images_filenames:
        images_abs_paths.append(IMAGES_FOLDER + atlas)

    confounds_files = os.listdir(CONFOUNDS)
    # print(confounds_files)
    try:
        confounds_files.remove(".DS_Store")
    except Exception as e:
        print(e)
    confounds_files.sort()

    abs_confounds_files = []
    for file in confounds_files:
        abs_confounds_files.append(CONFOUNDS + file)

    return images_filenames, images_abs_paths, abs_confounds_files

def extract_components(images_abs_paths, OUTPUT_FOLDER):
    #extract components from 20 images (10 from the control group and 10 from patients)
    images = nilearn.image.load_img(images_abs_paths[0:10] + images_abs_paths[-11:-1])

    # # RUN ICA
    # print("Running DictLearning...")
    # canica = CanICA(n_components=20,
    #                 memory="nilearn_cache", memory_level=2,
    #                 verbose=10,
    #                 mask_strategy='whole-brain-template',
    #                 random_state=0,
    #                 standardize=True,
    #                 # low_pass=0.08,
    #                 high_pass=0.01,
    #                 t_r=2)
    # # other filters here-> https://nilearn.github.io/modules/generated/nilearn.decomposition.CanICA.html
    # canica.fit(images)
    #
    # # Retrieve the independent components in brain space. Directly
    # # accessible through attribute `components_img_`.
    # components_img = canica.components_img_

    # RUN DICTLEARNING
    dict_learn = DictLearning(n_components=8, smoothing_fwhm=6.,
                              memory="nilearn_cache", memory_level=2,
                              random_state=0,
                              low_pass=0.08,
                              high_pass=0.01)
    # Fit to the data
    dict_learn.fit(images)
    # Resting state networks/maps in attribute `components_img_`
    components_img = dict_learn.components_img_

    # EXTRACT MOST INTENSE REGIONS
    # Import Region Extractor algorithm from regions module
    # threshold=0.5 indicates that we keep nominal of amount nonzero voxels across all
    # maps, less the threshold means that more intense non-voxels will be survived.
    from nilearn.regions import RegionExtractor
    print("extracting the most intense regions...")
    extractor = RegionExtractor(components_img, threshold=0.5,
                                thresholding_strategy='ratio_n_voxels',
                                extractor='local_regions',
                                standardize=True, min_region_size=1350)

    # Just call fit() to process for regions extraction
    extractor.fit()
    # Extracted regions are stored in regions_img_
    regions_extracted_img = extractor.regions_img_

    # Each region index is stored in index_
    regions_index = extractor.index_
    # Total number of regions extracted
    n_regions_extracted = regions_extracted_img.shape[-1]

    # Visualization of region extraction results
    title = ('%d regions are extracted from %d components.'
             '\nEach separate color of region indicates extracted region'
             % (n_regions_extracted, 8))
    plotting.plot_prob_atlas(regions_extracted_img, view_type='filled_contours',
                             title=title)

    plt.savefig(OUTPUT_FOLDER + "regions_extracted_dict_img.png")

    return extractor

def generate_time_series(images_filenames, images_abs_paths, extractor, OUTPUT_FOLDER, abs_confounds_files):
    # GENERATE AND SAVE TIME SERIES
    print("Generating time-series...")
    for filename, image, confounds in zip(images_filenames, images_abs_paths, abs_confounds_files):
        # call transform from RegionExtractor object to extract timeseries signals
        print(filename)
        data = np.genfromtxt(fname=confounds, delimiter="\t", skip_header=1, filling_values=1)
        confounds = numpy.nan_to_num(data, copy=True)
        timeseries_each_subject = extractor.transform(image, confounds=confounds)
        numpy.save(OUTPUT_FOLDER + "time_series_dict/" + filename.split("/")[-1].split(".nii")[0] + ".npy",
                   timeseries_each_subject)

def generate_dict_time_series_NILEARN(IMAGES_FOLDER, OUTPUT_FOLDER, CONFOUNDS):
    images_filenames, images_abs_paths, abs_confounds_files = load_paths(IMAGES_FOLDER, OUTPUT_FOLDER, CONFOUNDS)

    extractor = extract_components(images_abs_paths, OUTPUT_FOLDER)

    generate_time_series(images_filenames, images_abs_paths, extractor, OUTPUT_FOLDER, abs_confounds_files)
