import struct
import numpy as np
from matplotlib import pyplot as plt
from os.path import getsize



def main():

    dir_path = './'
    filename = 'rx_data.out'

    num_samp = getsize(filename)//8
    rx_data = np.zeros(num_samp, dtype= np.complex64)

    print('filename: ', filename)
    
    with open(dir_path+filename, "rb") as f:
        for j in range(num_samp):
            r_data = struct.unpack('f', f.read(4))[0]
            i_data = struct.unpack('f', f.read(4))[0]
            rx_data[j] = r_data+1j*i_data

if __name__ == "__main__":
    main()
