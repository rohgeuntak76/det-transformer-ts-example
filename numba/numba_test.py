import numpy as np
from numba import jit
import time

@jit
def numba_func(input):
    sol = np.tanh(input)
    return sol 


def no_numba_func(input):
    sol = np.tanh(input)
    return sol 


if __name__  == '__main__':
    data_length = 1000000000
    input_data = np.arange(data_length)

    start_time = time.time()
    numba_func(input_data)
    end_time = time.time()
    print("Elapsed Time (with numba):",end_time-start_time)

    start_time = time.time()
    no_numba_func(input_data)
    end_time = time.time()
    print("Elapsed Time (without numba):",end_time-start_time)