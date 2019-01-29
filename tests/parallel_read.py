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

core_path = '/home/jphollowed/data/hacc/alphaQ/particles/downsampled_particles/STEP487/m000.mpicosmo.487'
r_avg = 0.018

def main():
    
    #arr = getData(core_path, rank, size, 'core_tag', 'x', 'y', 'z', 'radius')

    arr = getData(core_path, rank, size, 'id', 'x', 'y', 'z')
    print "Data loaded... "
    comm.barrier()

    #now we redistribute the arr array by position

    redist = rd.MPIGridRedistributor(comm, [2, 2, 2], [256, 256, 256])
    pos = redist.stack_position([arr['x'], arr['y'], arr['z']])
    arr = redist.redistribute_by_position(arr, pos,  overload_lengths=[r_avg, r_avg, r_avg])
    print "Data redistrubted..."
    pos2 = arr
    
    #pos = redist.stack_position([arr['x'], arr['y'], arr['z']])

    buffer_data = redist.exchange_overload_by_position(arr, pos2, [2, 2, 2], periodic=True)
    print "Vuffer redist.... " 
    if rank == 2:
        plt.figure()
        plt.plot(arr['x'][::10], arr['y'][::10], 'xk', label = 'original data')
       # plt.hist2d(arr['x'], arr['y'], bins=128, norm=LogNorm())
        plt.plot(buffer_data['x'][::10], buffer_data['y'][::10], '.g', label = 'overflow data')
        #[::n] plots every nth data point, so this data is downsampled 90% for ease of viewing
        s = 'Rank ',rank
        plt.title(s)
        plt.colorbar()
        plt.show()
        plt.savefig('redist_part_{}.png'.format(rank))

if __name__ == '__main__':
    main()
