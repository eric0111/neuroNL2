import os

import nilearn
import numpy
from matplotlib import pyplot as plt
from nilearn import plotting
from nilearn.decomposition import CanICA


def generate_ica_time_series(IMAGES_FOLDER, OUTPUT_FOLDER):
    # LOAD IMAGES
    images_filenames = os.listdir(IMAGES_FOLDER)
    try:
        images_filenames.remove(".DS_Store")
    except Exception as e:
        print(e)
    images_filenames.sort()

    print(images_filenames)

    images_abs_paths = []
    for atlas in images_filenames:
        images_abs_paths.append(IMAGES_FOLDER + atlas)

    print(images_abs_paths)

    images = nilearn.image.load_img(images_abs_paths)

    # RUN ICA
    canica = CanICA(n_components=20,
                    memory="nilearn_cache", memory_level=2,
                    verbose=10,
                    mask_strategy='whole-brain-template',
                    random_state=0,
                    standardize=True,
                    )
    # other filters here-> https://nilearn.github.io/modules/generated/nilearn.decomposition.CanICA.html
    canica.fit(images)

    # Retrieve the independent components in brain space. Directly
    # accessible through attribute `components_img_`.
    components_img = canica.components_img_

    # EXTRACT MOST INTENSE REGIONS
    # Import Region Extractor algorithm from regions module
    # threshold=0.5 indicates that we keep nominal of amount nonzero voxels across all
    # maps, less the threshold means that more intense non-voxels will be survived.
    from nilearn.regions import RegionExtractor
    print("extracting most intense regions...")
    extractor = RegionExtractor(components_img, threshold=0.5,
                                thresholding_strategy='ratio_n_voxels',
                                extractor='local_regions',
                                standardize=True, min_region_size=1350)
    # Just call fit() to process for regions extraction
    extractor.fit()
    # Extracted regions are stored in regions_img_
    regions_extracted_img = extractor.regions_img_
    regions_extracted_img.to_filename(OUTPUT_FOLDER + "regions_extracted_img.nii.gz")
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

    plt.savefig(OUTPUT_FOLDER + "regions_extracted_img.png")

    #GENERATE AND SAVE TIME SERIES
    for filename, image in zip(images_filenames, images_abs_paths):
        # call transform from RegionExtractor object to extract timeseries signals
        timeseries_each_subject = extractor.transform(image)
        numpy.save(OUTPUT_FOLDER + "time_series/" + filename.split("/")[-1].split(".nii")[0] + ".npy",
                                   timeseries_each_subject)
