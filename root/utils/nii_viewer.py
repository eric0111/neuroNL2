import nibabel as nib
import matplotlib.pyplot as plt

def show_slices(slices):
   """ Function to display row of image slices """
   fig, axes = plt.subplots(1, len(slices))
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap="gray", origin="lower")

def print_shapes_from_nii_path(file_path):
    epi_img_4D = nib.load(file_path)
    epi_img_data_4D = epi_img_4D.get_fdata()
    print("epi_img_data_4D.shape ", epi_img_data_4D.shape)

    epi_img_3D = nib.four_to_three(epi_img_4D)
    epi_img_data_3D = epi_img_3D[0].get_fdata()
    print("epi_img_data_3D.shape ", epi_img_data_3D.shape)

def print_shapes_from_nii_img(epi_img_4D):
    epi_img_data_4D = epi_img_4D.get_fdata()
    print("epi_img_data_4D.shape ", epi_img_data_4D.shape)

    epi_img_3D = nib.four_to_three(epi_img_4D)
    epi_img_data_3D = epi_img_3D[0].get_fdata()
    print("epi_img_data_3D.shape ", epi_img_data_3D.shape)

def save_t0_plot_from_nii_img(file_path, fig_name):
    epi_img_4D = nib.load(file_path)
    epi_img_3D = nib.four_to_three(epi_img_4D)
    epi_img_data_3D = epi_img_3D[0].get_fdata()
    # PLOT
    epi_img_data = epi_img_data_3D

    slice_0 = epi_img_data[26, :, :]
    slice_1 = epi_img_data[:, 30, :]
    slice_2 = epi_img_data[:, :, 16]

    show_slices([slice_0, slice_1, slice_2])
    plt.savefig(fig_name)
    plt.close()

def save_t0_plot_from_nii_path(epi_img_4D, fig_name):
    epi_img_3D = nib.four_to_three(epi_img_4D)
    epi_img_data_3D = epi_img_3D[0].get_fdata()
    # PLOT
    epi_img_data = epi_img_data_3D

    slice_0 = epi_img_data[26, :, :]
    slice_1 = epi_img_data[:, 30, :]
    slice_2 = epi_img_data[:, :, 16]

    show_slices([slice_0, slice_1, slice_2])
    plt.savefig(fig_name)
    plt.close()


