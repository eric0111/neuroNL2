import numpy as np
from root.utils.generate_labels import generate_labels


def generate_data_reshaped_and_labels(TIME_SERIES_FOLDER):
    all_subjects_data, labels = generate_labels(TIME_SERIES_FOLDER)

    print('N control:', labels.count(1))
    print('N patients:', labels.count(5))

    max_len_image = np.max([len(i) for i in all_subjects_data])
    print(max_len_image)

    all_subjects_data_reshaped = []
    for subject_data in all_subjects_data:
        # Padding
        N = max_len_image - len(subject_data)
        padded_array = np.pad(subject_data, ((0, N), (0, 0)),
                              'constant', constant_values=(0))
        subject_data = padded_array
        subject_data = np.array(subject_data)
        subject_data.reshape(subject_data.shape[0], subject_data.shape[1], 1)
        all_subjects_data_reshaped.append(subject_data)

    # shape of data
    # 40 subjects
    # 261 time stamps
    # 10 networks values
    # (40, 261, 70)

    print(np.array(all_subjects_data_reshaped).shape)
    # (171, 184, 71)

    return all_subjects_data_reshaped, labels
