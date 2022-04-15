from matplotlib import pyplot as plt
from nilearn import plotting
from nilearn.regions import RegionExtractor

from utils.create_folder import create_folder


def extract_most_intense_regions(components_img, OUTPUT_FOLDER):
    # EXTRACT MOST INTENSE REGIONS
    # Import Region Extractor algorithm from regions module
    # threshold=0.5 indicates that we keep nominal of amount nonzero voxels across all
    # maps, less the threshold means that more intense non-voxels will be survived.
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

    plt.savefig(OUTPUT_FOLDER + "regions_extracted_img.png")

    return extractor