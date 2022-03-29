# PLOTS cleaned files
# bold = nilearn.image.clean_img(abs_atlas_files[0], detrend=False, standardize=True, t_r=TR,
#                                               confounds=abs_confounds_files[0])
# func_d = nilearn.image.clean_img(abs_atlas_files[0], detrend=True, standardize=True, t_r=TR,
#                                               confounds=abs_confounds_files[0])
# x, y, z = [31, 14, 7]
# plt.figure(figsize=(12, 4))
# plt.plot(np.transpose(bold.get_fdata()[x, y, z, :]))
# plt.plot(np.transpose(func_d.get_fdata()[x, y, z, :]))
# plt.legend(['Confounds', 'Confounds+ Detrend'])
# plt.savefig("detrend2.png")
# cleaned_signals = nilearn.image.clean_img(abs_atlas_files,
#                                           detrend=True, standardize=True,
#                                           confounds=abs_confounds_files)
#
# filtered_cleaned_signals = nilearn.image.clean_img(cleaned_signals,high_pass=0.01)
