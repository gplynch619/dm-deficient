'''Gabe Lynch
gabriel.p.lynch@gmail.com
ANL CPAC'''

import sys
import numpy as np
import genericio as gio
import matplotlib.pyplot as plt

#Defining constants
tsteps = [401, 411, 421, 432, 442, 453, 464, 475, 487]
core_path = '/media/luna1/dkorytov/data/AlphaQ/core_catalog5/'
particle_path= '/home/jphollowed/data/hacc/alphaQ/particles/downsampled_particles/'
r_avg = 0.018 #estimate of average radius
#########

#####Function definitions#########################
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

def getData(dataFile, *args): #Takes a data file and a list of attributes as input, and returns an array whose columns are the attributes from the file (e.g. mass, position, etc)
    nObjects = gio.gio_read(dataFile, args[0]).shape[0] #gets number of objects in file by checking the shape of the numpy array returned from the first arg .
    data_array = np.empty(shape=(nObjects, len(args)+1), dtype=float)#one row per object, one column per arg
    names = []
    formats = []
    for index, arg in enumerate(args):
        data_array[:,index] = gio.gio_read(dataFile, arg).flatten()
        names.append(arg)
        formats.append(type(gio.gio_read(dataFile, arg).flatten()[0]))
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

#####Main Program#########################################################
#make dict of timestep -> file
def main():
    lparticle_files = []
    lcore_files = []
    for t in tsteps:
        lparticle_files.append((t, particle_path+'STEP{0}/m000.mpicosmo.{0}'.format(t)))
        lcore_files.append((t,core_path+'07_13_17.AlphaQ.%s.coreproperties' % t))
    particle_files = dict(lparticle_files)
    core_files = dict(lcore_files)

    print "Dictionaries made..."

    core_data = []
    for core_file in core_files.values(): #returns list of numpy arrays, one for each timestep, with disrupted cores stipped out
        arr = getData(core_file, "core_tag", "x", "y", "z", "radius")
        m = arr['radius'] < 0.05
        arr = arr[m]
        core_data.append(arr)
        print "Cores from timestep ",arr[0]['timestep']," have been stripped..."

    for core_array in core_data:#arrays contain the non-disrupted cores at each timestep
        time_step = core_array[0]['timestep']
        particles = getData(particle_files[time_step], 'id', 'x', 'y', 'z')
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
if __name__ == '__main__':
    main()
