'''Gabe Lynch
gabriel.p.lynch@gmail.com
ANL CPAC'''

import sys
import numpy as np
import genericio as gio
import matplotlib.pyplot as plt

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

def computeMass(core):
    time_step = core['timestep']
    particles = getData(particle_files[time_step], 'id', 'x', 'y', 'z')
    print "Loaded particles for timestep ",time_step
    for c in ['x','y','z']:
        m = particles[c] < core[c]+1.5*r_avg
        particles = particles[m]
        m = particles[c] > core[c] - 1.5*r_avg
    particles = particles[m]
    if len(particles) != 0:
        return len(particles)
    elif len(particles) == 0:
        return -1

tsteps = [401, 411, 421, 432, 442, 453, 464, 475, 487]
core_path = '/media/luna1/dkorytov/data/AlphaQ/core_catalog5/'
particle_path= '/home/jphollowed/data/hacc/alphaQ/particles/downsampled_particles/'
r_avg = 0.018 #estimate of average radius

#make dict of timestep -> file
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

#candidate_cores = []
f=open('output.txt', 'w')
for core_array in core_data:
    cores_this_time = []
    print "Finding candidate cores for timestep ",core_array[0]['timestep']," ..."
    for core in core_array:
        m = computeMass(core)
        print "Calculating mass for Core ",core['core_tag']
        mass_ratio = 20./m
        if m != -1:
            if (mass_ratio < 1.):
                cores_this_time.append((core['core_tag'], mass_ratio, core['timestep']))
        elif m == -1:
            cores_this_time.append((core['core_tag'], 'No DM', core['timestep']))
    f.write("Number of candidate cores this timestep: ",len(cores_this_time))
    f.write("Core tag            Mass Ratio")
    for candidate_core in cores_this_time:
        f.write(candidate_core[0],"     ",candidate_core[1],"      ",candidate_core[2])
f.close()
#fig, axes = plt.subplots(1, 2, sharey=True, tight_layout=True)

