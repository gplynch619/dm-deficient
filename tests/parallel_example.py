''' Gabe Lynch
gabriel.p.lynch@gmail.com
ANL-CPAC'''

import sys
import numpy as np
import genericio as gio
from mpi4py import MPI
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

file_name = '/media/luna1/dkorytov/data/AlphaQ/core_catalog5/07_13_17.AlphaQ.401.coreproperties'

def main():
    arrx = gio.gio_read(file_name, 'x', rank, size)
    arry = gio.gio_read(file_name, 'y', rank, size)

    i = 0
    for elem in arrx[:,0]:
        if elem != 0:
            i +=1
    perc= np.around(i/float(arrx.shape[0]), decimals=3)
    print i,"/",arrx.shape[0],"(",perc*100,"%)", " nonzero elements in rank ",rank

    plt.scatter(arrx, arry)
    plt.show()
if __name__ == '__main__':
    main()
