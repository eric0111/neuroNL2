from matplotlib import pyplot as plt

def test():
    # OUTPUT_FOLDER = "../bin/output/stopsignal/"
    # temp_list = [1,1,1,0,0,3,3,3]
    # hello = numpy.asarray(temp_list)
    # print(temp_list[0:2] + temp_list[-3:-1])
    # #numpy.save(OUTPUT_FOLDER + "hello.npy", hello)

    path = "/root/old/output/stopsignal2/time_series_dict/sub-10159_task-stopsignal_bold_space-MNI152NLin2009cAsym_preproc.npy"
    import numpy as np
    time_series = np.load(path)
    # X = np.load("./X.npy")
    # Y = np.load("./Y.npy")

    print(time_series.shape)

    X = time_series[:,1]
    Y = time_series[:,2]
    from tsaug.visualization import plot
    plot(X, Y)
    plt.show()


    from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse
    my_augmenter = (TimeWarp() * 5  # random time warping 5 times in parallel
                            + Crop(size=184)  # random crop subsequences with length 300
                            + Quantize(n_levels=[10, 20, 30])  # random quantize to 10-, 20-, or 30- level sets
                            + Drift(
        max_drift=(0.1, 0.5)) @ 0.8  # with 80% probability, random drift the signal up to 10% - 50%
                            + Reverse() @ 0.5  # with 50% probability, reverse the sequence
                            )
    X_aug, Y_aug = my_augmenter.augment(X, Y)
    plot(X_aug, Y_aug)
    plt.show()
test()