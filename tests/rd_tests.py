''' Gabe Lynch
gabriel.p.lynch@gmail.com
ANL-CPAC'''

import sys
import numpy as np
import genericio as gio
import matplotlib.pyplot as plt
import redist as rd
import colormaps as cmaps

from matplotlib.colors import LogNorm
from local_density_calc import getData
from mpi4py import MPI



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

core_path = '/media/luna1/dkorytov/data/AlphaQ/core_catalog5/07_13_17.AlphaQ.401.coreproperties'
r_avg = 0.018

plt.register_cmap(name='viridis', cmap = cmaps.viridis)
plt.set_cmap(cmaps.viridis)

def main():
    
    arr = getData(core_path, rank, size, 'core_tag', 'x', 'y', 'z', 'radius')

    comm.barrier()

    #now we redistribute the arr array by position

    redist = rd.MPIGridRedistributor(comm, [2, 2, 2], [256, 256, 256])
    pos = redist.stack_position([arr['x'], arr['y'], arr['z']])
    redist_arr = redist.redistribute_by_position(arr, pos, overload_lengths=[r_avg, r_avg, r_avg])


    plt.rc('font',family='serif')
    
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=False)

    axes[0].hist2d(arr['x'], arr['y'], bins=128, norm=LogNorm())
    axes[1].hist2d(redist_arr['x'], redist_arr['y'], bins=128, norm=LogNorm())

    axes[0].set_title('Original Data')
    axes[1].set_title('Redistributed Data')

    fig.text(0.5, 0.04, "X (Mpc/h)", ha='center')
    fig.text(0.04, 0.5, "Y (Mpc/h)", ha='center', rotation='vertical')
    fig.suptitle('Rank {}'.format(rank))
    plt.draw()
    fig.savefig('redist_core_{}.png'.format(rank))

if __name__ == '__main__':
    main()
