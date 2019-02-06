'''Gabe Lynch
gabriel.p.lynch@gmail.com
ANL CPAC'''

import sys
import numpy as np
import genericio as gio
import redist as rd
import time

from mpi4py import MPI
from topology_optimize import topology_optimizer
######### MPI ##########################
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
########################################

######### Defining constants ###########
r_avg = 0.1800 #estimate of average radius
#######################################

######### Function definitions ########
def repsTimestep(string): #checks if a string represents an int in [0, 500]
    try:
        int(string)
        return True
    except ValueError:
        return False

def getTimestep(datafile):
    split_name = datafile.split('.')
    for s in split_name:
        if repsTimestep(s):
            return int(s)

def listRange(l):
    return range(l[0], l[1]+1)

def getData(dataFile, rank, size, *args): #Takes a data file and a list of attributes as input, and returns an array whose columns are the attributes from the file (e.g. mass, position, etc)
    nObjects = gio.gio_read(dataFile, args[0], rank, size).shape[0] #gets number of objects in file by checking the shape of the numpy array returned from the first arg .
    data_array = np.empty(shape=(nObjects, len(args)))#one row per object, one column per arg
    names = []
    formats = []
    for index, arg in enumerate(args):
        arr = gio.gio_read(dataFile, arg, rank, size).flatten()
        data_array[:,index] = arr
        names.append(arg)
        formats.append(type(arr[0]))
    dtype = dict(names = names, formats=formats)
    new_array = np.core.records.fromarrays(data_array.transpose(), dtype)
    return new_array

def computeMass(core, sorted_list):
    x_sorted = sorted_list[0]
    y_sorted = sorted_list[1]
    z_sorted = sorted_list[2]
    
    cx = core['x']
    cy = core['y']
    cz = core['z']

    x_range = [cx - r_avg, cx + r_avg]
    y_range = [cy - r_avg, cy + r_avg]
    z_range = [cz - r_avg, cz + r_avg]
    
    x_cut = listRange(np.searchsorted(x_sorted['x'], x_range)-[0,1])
    y_cut = listRange(np.searchsorted(y_sorted['y'], y_range)-[0,1])
    z_cut = listRange(np.searchsorted(z_sorted['z'], z_range)-[0,1])

    x = x_sorted[x_cut]
    y = y_sorted[y_cut]
    z = z_sorted[z_cut]

    local_particles = reduce(np.intersect1d, (x, y, z))

    return local_particles
#########################################################################

########################## Main Program #################################
#make dict of timestep -> file
def main():

    core_file = sys.argv[1] 
    particle_file = sys.argv[2]

    start=time.time()
    #load timestep
    cores = getData(core_file, rank, size, "core_tag", "x", "y", "z", "fof_halo_tag", "infall_mass","radius")
    m = cores['radius'] < 0.05
    cores = cores[m]
    
    comm.barrier()
   
    #Begin Redistribution#
    grid_topology = topology_optimizer(size, 3)
    redist = rd.MPIGridRedistributor(comm, grid_topology, [256, 256, 256])
    pos = redist.stack_position([cores['x'], cores['y'], cores['z']])    
    redist_cores = redist.redistribute_by_position(cores, pos, overload_lengths=[0, 0, 0])

    
    timestep = getTimestep(core_file)
    particles = getData(particle_file, rank, size, 'id', 'x', 'y', 'z')
    
    comm.barrier()

    particle_pos = redist.stack_position([particles['x'], particles['y'], particles['z']])
    redist_particles = redist.redistribute_by_position(particles, particle_pos, overload_lengths=[0,0,0])
    #End Redistribution#

    li = []
    li.append(redist_particles[redist_particles['x'].argsort()])
    li.append(redist_particles[redist_particles['y'].argsort()])
    li.append(redist_particles[redist_particles['z'].argsort()])
 
    saved_cores_list = []
    count_dict = {}
    n = 0
    for i, core in enumerate(cores):
        p = computeMass(core, li)
        m = len(p)
        #if m < 50:
	count_dict[n] = m
	saved_cores_list.append(core)
        n += 1
    #This block of code adds a new field to store the count of particles. Try to optimize       
    
    saved_cores_tmp = np.array(saved_cores_list)
    final_cores = np.empty(saved_cores_tmp.shape, dtype=saved_cores_tmp.dtype.descr + [('count',np.int16)])
    for name in saved_cores_tmp.dtype.names:
        final_cores[name] = saved_cores_tmp[name]
    for i, elem in enumerate(final_cores):
        elem['count']=count_dict[i]
    
    directory = "/home/gplynch/proj/dm-deficient/output/"
    filename = "{0}E/02_04_19.{0}.deficientcores#{1}".format(timestep, rank)
    np.save(directory+filename, final_cores)
    
    end = time.time()
    time_length = end - start
    st = "Total time was {1} seconds".format(time_length)
    if rank==0:
        paramfile=directory+'{0}Eparams.txt'
        with open(paramfile, 'w+') as f:
	    f.write('Search radius length was {0}'.format(r_avg))
	    f.write(st)

if __name__ == '__main__':
    main()
