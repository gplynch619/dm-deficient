'''Gabe Lynch
gabriel.p.lynch@gmail.com
ANL CPAC'''

import sys
import numpy as np
import genericio as gio
import redist as rd
import matplotlib.pyplot as plt
import time

from mpi4py import MPI
######### MPI ##########################
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
########################################

######### Defining constants ###########
#core_file = '/media/luna1/dkorytov/data/AlphaQ/core_catalog5/07_13_17.AlphaQ.487.coreproperties'
core_file = sys.argv[1]
particle_file = sys.argv[2]
#particle_file = '/home/jphollowed/data/hacc/alphaQ/particles/downsampled_particles/STEP487/m000.mpicosmo.487'
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
    data_array = np.empty(shape=(nObjects, len(args)+1), dtype=float)#one row per object, one column per arg
    names = []
    formats = []
    for index, arg in enumerate(args):
        data_array[:,index] = gio.gio_read(dataFile, arg, rank, size).flatten()
        names.append(arg)
        formats.append(type(gio.gio_read(dataFile, arg, rank, size).flatten()[0]))
    data_array[:, len(args)] = np.full((nObjects,), getTimestep(dataFile))
    names.append('timestep')
    formats.append('i4')
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

    return len(local_particles)
#########################################################################

########################## Main Program #################################
#make dict of timestep -> file
def main():
   
    #load timestep
    cores = getData(core_file, rank, size, "core_tag", "x", "y", "z", "radius")
    print "Data for rank ",rank," loaded" 
    m = cores['radius'] < 0.05
    cores = cores[m]
    
    comm.barrier()
   
    print "Loading and redistributing data..."
    
    grid_partition = int(np.cbrt(size))

    redist = rd.MPIGridRedistributor(comm, [grid_partition,grid_partition, grid_partition], [256, 256, 256])
    
    pos = redist.stack_position([cores['x'], cores['y'], cores['z']])
    
    redist_cores = redist.redistribute_by_position(cores, pos, overload_lengths=[r_avg, r_avg, r_avg])
    
    str_out = "Rank {0} has Min{1} of {2} and Max{1} of {3}, with {4}|{5} split"
    
    xmin = np.amin(redist_cores_at_timestep[0]['x'])
    xmax = np.amax(redist_cores_at_timestep[0]['x'])
    xless = np.sum(redist_cores_at_timestep[0]['x'] < 128)
    xmore =  np.sum(redist_cores_at_timestep[0]['x'] >= 128)

    ymin = np.amin(redist_cores_at_timestep[0]['y'])
    ymax = np.amax(redist_cores_at_timestep[0]['y'])
    yless = np.sum(redist_cores_at_timestep[0]['y'] < 128)
    ymore =  np.sum(redist_cores_at_timestep[0]['y'] >= 128)
    
    zmin = np.amin(redist_cores_at_timestep[0]['z'])
    zmax = np.amin(redist_cores_at_timestep[0]['z'])
    zless = np.sum(redist_cores_at_timestep[0]['z'] < 128)
    zmore =  np.sum(redist_cores_at_timestep[0]['z'] >= 128)
   
    #wait a bit based on rank so outputs aren't all mangled
    #sleep(0.5*rank) 
    #print(str_out.format(rank, "X", xmin, xmax, xless, xmore))
    #print(str_out.format(rank, "Y", ymin, ymax, yless, ymore))
    #print(str_out.format(rank, "Z", zmin, zmax, zless, zmore))
    #print('\n')

    #at this point, if all has gone well, the core data should be loaded and spatially distributed across ranks. We now want to do the same with the particles


    timestep = cores[0]['timestep']
    particles = getData(particle_file, rank, size, 'id', 'x', 'y', 'z')
    particle_pos = redist.stack_position([particles['x'], particles['y'], particles['z']])

    redist_particles = redist.redistribute_by_position(particles, particle_pos, overload_lengths=[r_avg, r_avg, r_avg])
   
    print "Rank {0} has loaded particles for timestep {1}".format(rank, timestep)

    li = []
    li.append(redist_particles[redist_particles['x'].argsort()])
    li.append(redist_particles[redist_particles['y'].argsort()])
    li.append(redist_particles[redist_particles['z'].argsort()])
    s = "outputs/487/MPI_{0}.out#{1}".format(timestep,rank)
    f = open(s, "w")
    f.write('Core Tag | Local particles | (x,y,z) | Radius')
    for i, core in enumerate(cores):
        m = computeMass(core, li)
        if m < 50:
            s1 = "{0} {1} ({2},{3},{4}) {5}\n".format(core['core_tag'], m, core['x'], core['y'], core['z'], core['radius'])
            f.write(s1) 
    f.close()

if __name__ == '__main__':
    main()
