import matplotlib.pyplot as plt
from nilearn import plotting
import os
from nilearn.decomposition import CanICA
from nilearn.connectome import ConnectivityMeasure
import numpy as np

#https://nilearn.github.io/auto_examples/03_connectivity/plot_extract_regions_dictlearning_maps.html
#This example can also be inspired to apply the same steps to even regions extraction using ICA maps.
# In that case, idea would be to replace Dictionary learning to canonical ICA decomposition using nilearn.decomposition.CanICA

#SETTINGS
FOLDER = "/Users/eb/Desktop/test_files/files/"
CONFOUNDS = "/Users/eb/Desktop/test_files/confounds/"
# n/a in stdDVARS	non-stdDVARS	vx-wisestdDVARS	FramewiseDisplacement in the first line was replaced with 0 using utils.py

#RETRIEVE FILES AND SORT THEM ALFABETICALLY
atlas_files = os.listdir(FOLDER)
atlas_files.sort()
confounds_files = os.listdir(CONFOUNDS)
confounds_files.sort()

# #PRINT FILES
# print(atlas_files)
# print(confounds_files)

#ADD ABSOULTE PATH TO FILES
abs_atlas_files = []
for file in atlas_files:
    abs_atlas_files.append(FOLDER+file)

abs_confounds_files = []
for file in confounds_files:
    abs_confounds_files.append(CONFOUNDS+file)


#ICA
func_filenames = abs_atlas_files
confounds = abs_confounds_files

canica = CanICA(n_components=20,
                memory="nilearn_cache", memory_level=2,
                verbose=10,
                mask_strategy='whole-brain-template',
                random_state=0,
                standardize = True,
                #low_pass=0.08,
                high_pass=0.01,
                t_r=2)
#other filters here-> https://nilearn.github.io/modules/generated/nilearn.decomposition.CanICA.html
canica.fit(func_filenames)

# Retrieve the independent components in brain space. Directly
# accessible through attribute `components_img_`.
components_img = canica.components_img_

#EXTRACT MOST INTENSE REGIONS
# Import Region Extractor algorithm from regions module
# threshold=0.5 indicates that we keep nominal of amount nonzero voxels across all
# maps, less the threshold means that more intense non-voxels will be survived.
from nilearn.regions import RegionExtractor

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

plt.savefig("./bin/output/regions_extracted_img.png")

#TIME-SERIES AND CORRELATION MATRIXES
# First we need to do subjects timeseries signals extraction and then estimating
# correlation matrices on those signals.
# To extract timeseries signals, we call transform() from RegionExtractor object
# onto each subject functional data stored in func_filenames.
# To estimate correlation matrices we import connectome utilities from nilearn

correlations = []
# Initializing ConnectivityMeasure object with kind='correlation'
connectome_measure = ConnectivityMeasure(kind='correlation')
for filename, confound in zip(func_filenames, confounds):
    # call transform from RegionExtractor object to extract timeseries signals
    timeseries_each_subject = extractor.transform(filename, confounds=confound)

    #plot time series
    # plt.close()
    # plt.plot(timeseries_each_subject)
    # plt.savefig(./bin/output/filename+"_time-series.png")
    # plt.close()

    # call fit_transform from ConnectivityMeasure object
    correlation = connectome_measure.fit_transform([timeseries_each_subject])

    #plot correlation matrixes
    display1 = plotting.plot_matrix(correlation.reshape(n_regions_extracted,n_regions_extracted), colorbar=True )
    plt.savefig("./bin/output/corr_matrix_"+filename.split("/")[-1].split(".nii")[0]+".png")
    plt.close()

    # saving each subject correlation to correlations
    correlations.append(correlation)

# MEAN OF ALL CORRELATION MATRIXES
mean_correlations = np.mean(correlations, axis=0).reshape(n_regions_extracted,
                                                          n_regions_extracted)

title = 'Correlation between %d regions' % n_regions_extracted
display = plotting.plot_matrix(mean_correlations, vmax=1, vmin=-1,
                               colorbar=True, title=title)
plt.savefig("./bin/output/mean_corr_matrix.png")
plt.close()

#CONNECTOME
# Then find the center of the regions and plot a connectome
regions_img = regions_extracted_img
coords_connectome = plotting.find_probabilistic_atlas_cut_coords(regions_img)
plt.savefig("./bin/output/atlas.png")
plt.close()

plotting.plot_connectome(mean_correlations, coords_connectome,
                         edge_threshold='90%', title=title)
plt.savefig("./bin/output/connectome.png")
plt.close()