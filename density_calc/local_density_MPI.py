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
from time import sleep
start = time.time()
######### MPI ##########################
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
########################################

######### Defining constants ###########
tsteps = [401, 411, 421, 432, 442, 453, 464, 475, 487]
core_path = '/media/luna1/dkorytov/data/AlphaQ/core_catalog5/'
particle_path= '/home/jphollowed/data/hacc/alphaQ/particles/downsampled_particles/'
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
    lparticle_files = []
    lcore_files = []
    for t in tsteps:
        lparticle_files.append((t, particle_path+'STEP{0}/m000.mpicosmo.{0}'.format(t)))
        lcore_files.append((t,core_path+'07_13_17.AlphaQ.%s.coreproperties' % t))
    particle_files = dict(lparticle_files)
    core_files = dict(lcore_files)

    cores_at_timestep = []
    for core_file in core_files.values(): #returns list of numpy arrays, one for each timestep, with disrupted cores stipped out
        arr = getData(core_file, rank, size, "core_tag", "x", "y", "z", "radius")
        print "Data for rank ",rank," loaded" 
        m = arr['radius'] < 0.05
        arr = arr[m]
        cores_at_timestep.append(arr)
       # print "Cores from timestep ",arr[0]['timestep']," have been stripped..."
    
    comm.barrier()
    
    if rank == 0:
        print "Getting ready to redistribute"


    redist = rd.MPIGridRedistributor(comm, [2,2,2], [256, 256, 256])
    redist_cores_at_timestep = []
    buffers = []
    for i, cores in enumerate(cores_at_timestep):
        pos = redist.stack_position([cores['x'], cores['y'], cores['z']])
        data_cell_num = redist.get_cell_number_from_position(pos)
        new_cores = redist.redistribute_by_cell_number(cores, data_cell_num)
        pos2 = redist.stack_position([new_cores['x'], new_cores['y'], new_cores['z']])
        buffer_data = redist.exchange_overload_by_position(new_cores, pos2, [2,2,2], periodic=True)
        buffers.append(buffer_data)

        redist_cores_at_timestep.append(redist.redistribute_by_position(cores, pos, overload_lengths=[1, 1, 1]))
    


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
    sleep(0.5*rank) 
    print(str_out.format(rank, "X", xmin, xmax, xless, xmore))
    print(str_out.format(rank, "Y", ymin, ymax, yless, ymore))
    print(str_out.format(rank, "Z", zmin, zmax, zless, zmore))
    print('\n')

    if rank ==0:
        plt.figure()
        plt.plot(cores_at_timestep[0]['x'], cores_at_timestep[0]['y'], 'xk', label = 'original data')
        plt.plot(redist_cores_at_timestep[0]['x'], redist_cores_at_timestep[0]['y'], '.b', label ='local data')
        plt.plot(buffers[0]['x'], buffers[0]['y'], '.g', label = 'overload')
        plt.xlim([0, 256])
        plt.ylim([0, 256])
        plt.title("Rank {}".format(rank))
        plt.ylabel("Mpc h^-1")
        plt.xlabel("Mpc h^-1")
        plt.legend()

    #at this point, if all has gone well, the core data should be loaded and spatially distributed across ranks. We now want to do the same with the particles


    for cores in cores_at_timestep:
        timestep = cores[0]['timestep']
        particles = getData(particle_files[timestep], rank, size, 'id', 'x', 'y', 'z')
        particle_pos = redist.stack_position([particles['x'], particles['y'], particles['z']])
        particle_cell_num = redist.get_cell_number_from_position(particle_pos)

        redist_particles = redist.redistribute_by_position(particles, particle_pos, overload_lengths=[r_avg, r_avg, r_avg])
       
        print "Rank {0} has loaded particles for timestep {1}".format(rank, timestep)

        li = []
        li.append(redist_particles[redist_particles['x'].argsort()])
        li.append(redist_particles[redist_particles['y'].argsort()])
        li.append(redist_particles[redist_particles['z'].argsort()])
        s = "outputs/MPI_{0}.out#{1}".format(timestep,rank)
        f = open(s, "w")
        f.write('Core Tag | Local particles | (x,y,z) | Radius')
        for i, core in enumerate(cores):
            m = computeMass(core, li)
            s1 = "{0} {1} ({2},{3},{4}) {5}\n".format(core['core_tag'], m, core['x'], core['y'], core['z'], core['radius'])
            if i%1000 == 0:
                print "Core {} written".format(core['core_tag'])
                f.write(s1)
        end = time.time()
        print(end - start)
        f.close()
        quit()
'''
    for core_array in core_data:#arrays contain the non-disrupted cores at each timestep
        time_step = core_array[0]['timestep']
        particles = getData(particle_files[time_step], rank, size, 'id', 'x', 'y', 'z')
        print "Particles for timestep ",time_step," are loaded..."
        li = []
        li.append(particles[particles['x'].argsort()])
        li.append(particles[particles['y'].argsort()])
        li.append(particles[particles['z'].argsort()])
        s = str(time_step)+".txt"
        f = open(s, "w")
        for core in core_array:
           # print "Computing local density for core ",core['core_tag']," ..."
            m = computeMass(core, li)
            s1 = str(core['core_tag'])+' '+str(m)
            f.write(s1+'\n')
        f.close()
'''
if __name__ == '__main__':
    main()
