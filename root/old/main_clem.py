import nilearn
import numpy
import matplotlib.pyplot as plt
from nilearn import plotting
import os
from nilearn.decomposition import CanICA
from nilearn.connectome import ConnectivityMeasure
import numpy as np
#https://peerherholz.github.io/workshop_weizmann/data/image_manipulation_nilearn.html

def main(task, FILES, CONFOUNDS, CLEANED_FOLDER):
    # RETRIEVE FILES AND SORT THEM ALPHABETICALLY
    atlas_files = os.listdir(FILES)
    try:
        atlas_files.remove(".DS_Store")
    except Exception as e:
        print(e)
    atlas_files.sort()
    #print(atlas_files)
    confounds_files = os.listdir(CONFOUNDS)
    try:
        confounds_files.remove(".DS_Store")
    except Exception as e:
        print(e)
    confounds_files.sort()

    # ADD ABSOULTE PATH TO FILES
    abs_atlas_files = []
    for atlas in atlas_files:
        abs_atlas_files.append(FILES + atlas)

    abs_confounds_files = []
    for confound in confounds_files:
        abs_confounds_files.append(CONFOUNDS + confound)

    bold = nilearn.image.load_img(abs_atlas_files[0])
    TR = bold.header['pixdim'][4]

    cleaned_signals = []
    for img, confounds, atlas_filename in zip(abs_atlas_files, abs_confounds_files, atlas_files):
        print("loading and cleaning: "+img)
        #load confounds.tsv into numpy array and remove nan and infs
        data = np.genfromtxt(fname=confounds, delimiter="\t", skip_header=1,filling_values=1)
        confounds = numpy.nan_to_num(data, copy=True)

        #clean image
        temp_cleaned_signal = nilearn.image.clean_img(img, detrend=False, standardize=True, t_r=TR,
                                         confounds=confounds)
        cleaned_signal = nilearn.image.clean_img(temp_cleaned_signal, high_pass=0.01, t_r=TR)

        cleaned_signal.to_filename(CLEANED_FOLDER+atlas_filename)


    ## ICA
    print("running ICA on cleaned signals....")
    func_filenames = abs_confounds_files #cleaned_signals

    canica = CanICA(n_components=20,
                    memory="nilearn_cache", memory_level=2,
                    verbose=10,
                    mask_strategy='whole-brain-template',
                    random_state=0,
                    standardize=True,
                    )
    # other filters here-> https://nilearn.github.io/modules/generated/nilearn.decomposition.CanICA.html
    canica.fit(cleaned_signals)

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

    plt.savefig("./bin/output/"+task+"/regions_extracted_img.png")

    # TIME-SERIES AND CORRELATION MATRIXES
    # First we need to do subjects timeseries signals extraction and then estimating
    # correlation matrices on those signals.
    # To extract timeseries signals, we call transform() from RegionExtractor object
    # onto each subject functional data stored in func_filenames.
    # To estimate correlation matrices we import connectome utilities from nilearn

    print("generating time-series and correlation matrixes...")
    correlations = []
    # Initializing ConnectivityMeasure object with kind='correlation'
    connectome_measure = ConnectivityMeasure(kind='correlation')
    for filename, cleaned_signal in zip(func_filenames, cleaned_signals):
        # call transform from RegionExtractor object to extract timeseries signals
        timeseries_each_subject = extractor.transform(cleaned_signal)

        # plot time series
        # plt.close()
        # plt.plot(timeseries_each_subject)
        # plt.savefig(./bin/output/"+task+"/filename+"_time-series.png")
        # plt.close()

        # call fit_transform from ConnectivityMeasure object
        correlation = connectome_measure.fit_transform([timeseries_each_subject])

        # plot correlation matrixes
        display1 = plotting.plot_matrix(correlation.reshape(n_regions_extracted, n_regions_extracted), colorbar=True)
        plt.savefig("./bin/output/"+task+"/corr_matrix_" + filename.split("/")[-1].split(".nii")[0] + ".png")
        plt.close()

        # saving each subject correlation to correlations
        correlations.append(correlation)
        numpy.save("./bin/output/"+task+"/corr_matrix_" + filename.split("/")[-1].split(".nii")[0] + ".npy", correlation)

    # MEAN OF ALL CORRELATION MATRIXES
    print("mean correlation matrix...")
    mean_correlations = np.mean(correlations, axis=0).reshape(n_regions_extracted,
                                                              n_regions_extracted)

    title = 'Correlation between %d regions' % n_regions_extracted
    display = plotting.plot_matrix(mean_correlations, vmax=1, vmin=-1,
                                   colorbar=True, title=title)
    plt.savefig("./bin/output/"+task+"/mean_corr_matrix.png")
    plt.close()
    numpy.save("./bin/output/"+task+"/mean_corr_matrix.npy", mean_correlations)

    # CONNECTOME
    # Then find the center of the regions and plot a connectome
    print("connectome")
    regions_img = regions_extracted_img
    coords_connectome = plotting.find_probabilistic_atlas_cut_coords(regions_img)
    plt.savefig("./bin/output/"+task+"/atlas.png")
    plt.close()

    plotting.plot_connectome(mean_correlations, coords_connectome,
                             edge_threshold='90%', title=title)
    plt.savefig("./bin/output/"+task+"/connectome.png")
    plt.close()

# SETTINGS
FILES  =  "/Users/eb/Desktop/test_files/files/"
CONFOUNDS =  "/Users/eb/Desktop/test_files/confounds/"
CLEANED_FOLDER = "/Users/eb/Desktop/test_files/cleaned/"
task = "test"
main(task, FILES, CONFOUNDS, CLEANED_FOLDER)

# FILES = "/Users/eb/Desktop/stopsignal/"
# CONFOUNDS =  "/Users/eb/Desktop/stopsignal_confounds/"
# task = "stopsignal"
# main(task , FILES, CONFOUNDS)
#
# FILES = "/Users/eb/Desktop/bart/"
# CONFOUNDS =  "/Users/eb/Desktop/bart_confounds/"
# task = "bart"
# main(task , FILES, CONFOUNDS)