import time


def get_time_process(target_function):
    start_time = time.time()
    target_function()
    end_time = time.time()

    return end_time - start_time

# import numpy as np
# x = np.array([1,2,3,4,5,6])
# print(x[[1,2,3]])
