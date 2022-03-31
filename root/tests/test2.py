import numpy


def test():
    OUTPUT_FOLDER = "../bin/output/stopsignal/"
    temp_list = [1,1,1,0,0,3,3,3]
    hello = numpy.asarray(temp_list)
    print(temp_list[0:2] + temp_list[-3:-1])
    #numpy.save(OUTPUT_FOLDER + "hello.npy", hello)

test()