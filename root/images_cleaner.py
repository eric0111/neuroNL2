from nilearn import image
import numpy
import os
import numpy as np
#https://peerherholz.github.io/workshop_weizmann/data/image_manipulation_nilearn.html

def images_cleaner(FILES, CONFOUNDS, OUTPUT_FOLDER):
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

    # ADD ABSOLUTE PATH TO FILES
    abs_atlas_files = []
    for atlas in atlas_files:
        abs_atlas_files.append(FILES + atlas)

    abs_confounds_files = []
    for confound in confounds_files:
        abs_confounds_files.append(CONFOUNDS + confound)

    bold = image.load_img(abs_atlas_files[0])
    TR = bold.header['pixdim'][4]

    #GENERATE CLEANED IMAGES AND SAVE TO CLEANED_FOLDER
    for img, confounds, atlas_filename in zip(abs_atlas_files, abs_confounds_files, atlas_files):
        print("loading and cleaning: "+img)
        #load confounds.tsv into numpy array and remove nan and infs
        data = np.genfromtxt(fname=confounds, delimiter="\t", skip_header=1,filling_values=1)
        confounds = numpy.nan_to_num(data, copy=True)

        #clean image
        temp_cleaned_signal = image.clean_img(img, detrend=False, standardize=True, t_r=TR,
                                         confounds=confounds)
        cleaned_signal = image.clean_img(temp_cleaned_signal, high_pass=0.01, t_r=TR)

        cleaned_signal.to_filename(OUTPUT_FOLDER + atlas_filename)

