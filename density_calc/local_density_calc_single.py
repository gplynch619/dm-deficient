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
r_avg = 0.018 #estimate of average radius
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

  
    #load timestep
    start = time.time()
    cores = getData(core_file, rank, size, "core_tag", "x", "y", "z", "fof_halo_tag", "radius")
    m = cores['radius'] < 0.05
    cores = cores[m]
    
    end = time.time()
    time_length = end - start
    st = "load cores {0} {1}".format(rank, time_length)
    print st
    comm.barrier()
   
    
    grid_topology = topology_optimizer(size, 3)

    redist = rd.MPIGridRedistributor(comm, grid_topology, [256, 256, 256])
    
    pos = redist.stack_position([cores['x'], cores['y'], cores['z']])
    
    start = time.time()
    redist_cores = redist.redistribute_by_position(cores, pos, overload_lengths=[0, 0, 0])
    end = time.time()
    time_length = end - start
    st = "redist cores {0} {1}".format(rank, time_length)
    print st

    
    timestep = getTimestep(core_file)
    start = time.time()
    particles = getData(particle_file, rank, size, 'id', 'x', 'y', 'z')
    end = time.time()
    
    time_length = end - start
    st = "load particles {0} {1}".format(rank, time_length)
    print st

    comm.barrier()

    particle_pos = redist.stack_position([particles['x'], particles['y'], particles['z']])
   
    start = time.time()
    redist_particles = redist.redistribute_by_position(particles, particle_pos, overload_lengths=[0,0,0])
    end = time.time()
  
    time_length = end - start
    st = "redist particles {0} {1}".format(rank, time_length)
    print st    

    start = time.time()
    li = []
    li.append(redist_particles[redist_particles['x'].argsort()])
    li.append(redist_particles[redist_particles['y'].argsort()])
    li.append(redist_particles[redist_particles['z'].argsort()])
    end = time.time()
    time_length = end - start
    print "sorting {0} {1}".format(rank, time_length)

#    s = "/home/gplynch/proj/output/{0}/MPI_{0}.out#{1}".format(timestep,rank)
#    f = open(s, "w+")
#    f.write('Core Tag | Local particles | (x,y,z) | Radius')
 
    start = time.time()
    saved_cores_tmp = []
    for i, core in enumerate(cores):
        p = computeMass(core, li)
        m = len(p)
        if m < 50:
	    saved_cores_tmp.append(core)
    saved_cores = np.array(saved_cores_tmp)
    s = "/home/gplynch/proj/output/{0}/01_22_19.{0}.deficientcores#{1}".format(timestep, rank)
    np.save(s, saved_cores)
    end = time.time()
    time_length = end - start
    st = "calc {0} {1}".format(rank, time_length)

if __name__ == '__main__':
    main()
