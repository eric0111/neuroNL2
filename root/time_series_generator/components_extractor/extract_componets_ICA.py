import nilearn
from nilearn.decomposition import CanICA, DictLearning
import root.constants as constants

def extract_components_ICA(images_abs_paths):
    #extract components from 20 images (10 from the control group and 10 from patients)
    images = nilearn.image.load_img(images_abs_paths[0:constants.NUM_IMAGES_FOR_COMPONENTS]
                                    + images_abs_paths[(-constants.NUM_IMAGES_FOR_COMPONENTS - 1):-1])

    # # RUN ICA
    print("Running ICA...")
    canica = CanICA(n_components=constants.NUM_COMPONENTS,
                    memory="nilearn_cache", memory_level=2,
                    verbose=10,
                    mask_strategy='whole-brain-template',
                    random_state=0,
                    low_pass=constants.LOW_PASS,
                    high_pass=constants.HIGH_PASS,
                    t_r=2)

    # Fit to the data
    canica.fit(images)

    # Retrieve the independent components in brain space. Directly
    # accessible through attribute `components_img_`.
    components_img = canica.components_img_

    return components_img