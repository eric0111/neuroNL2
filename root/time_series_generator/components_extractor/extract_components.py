from root import constants
from root.time_series_generator.components_extractor.extract_components_DICT import extract_components_DICT
from root.time_series_generator.components_extractor.extract_componets_ICA import extract_components_ICA

def extract_components(images_abs_paths, method):
    return {
        constants.METHOD_ICA: extract_components_ICA(images_abs_paths),
        constants.METHOD_DICTIONARY_LEARNING: extract_components_DICT(images_abs_paths)
    }.get(method)

