import nilearn
from nilearn.decomposition import CanICA, DictLearning
import root.constants as constants

def extract_components_DICT(images_abs_paths):
    #extract components from 20 images (10 from the control group and 10 from patients)
    images = nilearn.image.load_img(images_abs_paths[0:constants.NUM_IMAGES_FOR_COMPONENTS]
                                    + images_abs_paths[(-constants.NUM_IMAGES_FOR_COMPONENTS - 1):-1])

    # RUN DICTLEARNING
    dict_learn = DictLearning(n_components=constants.NUM_COMPONENTS,
                              smoothing_fwhm=6.,
                              memory="nilearn_cache", memory_level=2,
                              random_state=0,
                              standardize=constants.STANDARDIZE,
                              low_pass=constants.LOW_PASS,
                              high_pass=constants.HIGH_PASS,
                              t_r=2)
    # Fit to the data
    dict_learn.fit(images)
    # Resting state networks/maps in attribute `components_img_`
    components_img = dict_learn.components_img_
    return components_img