from root.old.images_cleaner import images_cleaner

# FILES  =  "/Users/eb/Desktop/test_files/files/"
# CONFOUNDS =  "/Users/eb/Desktop/test_files/confounds/"
# CLEANED_FOLDER = "/Users/eb/Desktop/test_files/cleaned/"
# cleaner(FILES, CONFOUNDS, CLEANED_FOLDER)

FILES  =  "/Users/eb/Desktop/stopsignal/"
CONFOUNDS =  "/Users/eb/Desktop/stopsignal_confounds/"
OUTPUT_FOLDER = "/Users/eb/Desktop/stopsignal_cleaned/"
images_cleaner(FILES, CONFOUNDS, OUTPUT_FOLDER)

# FILES = "/Users/eb/Desktop/stopsignal/"
# CONFOUNDS =  "/Users/eb/Desktop/stopsignal_confounds/"
# task = "stopsignal"
# main(task , FILES, CONFOUNDS)