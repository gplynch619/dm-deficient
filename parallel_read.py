''' Gabe Lynch
gabriel.p.lynch@gmail.com
ANL-CPAC'''

import sys
import numpy as np
import genericio as gio
import matplotlib.pyplot as plt
import redist as rd

from matplotlib.colors import LogNorm
from local_density_calc import getData
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

core_path = '/media/luna1/dkorytov/data/AlphaQ/core_catalog5/07_13_17.AlphaQ.401.coreproperties'
r_avg = 0.018

def main():
    
    arr = getData(core_path, rank, size, 'core_tag', 'x', 'y', 'z', 'radius')

    comm.barrier()

    #now we redistribute the arr array by position

    redist = rd.MPIGridRedistributor(comm, [2, 2, 2], [256, 256, 256])
    pos = redist.stack_position([arr['x'], arr['y'], arr['z']])
    data_cell_num = redist.get_cell_number_from_position(pos)
    arr = redist.redistribute_by_cell_number(arr, data_cell_num)

    pos = redist.stack_position([arr['x'], arr['y'], arr['z']])

    buffer_data = redist.exchange_overload_by_position(arr, pos, [2, 2, 2], periodic=False)

    if rank == 2:
        plt.figure()
       # plt.plot(arr['x'][::10], arr['y'][::10], 'xk', label = 'original data')
        plt.hist2d(arr['x'], arr['y'], bins=128, norm=LogNorm())
       # plt.plot(buffer_data['x'][::10], buffer_data['y'][::10], '.g', label = 'overflow data')
        #[::n] plots every nth data point, so this data is downsampled 90% for ease of viewing
        s = 'Rank ',rank
        plt.title(s)
        plt.colorbar()
        plt.show()
        plt.savefig('redist_core.png')

if __name__ == '__main__':
    main()
