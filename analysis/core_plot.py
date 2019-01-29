import sys
import numpy as np
import genericio as gio
import time
import matplotlib.pyplot as plt
import colormaps as cmaps

sys.path.insert(0, '/home/lynchg/dm-deficient/')
from local_density_MPI import getData
from matplotlib.colors import LogNorm

fof_file = "/media/luna1/dkorytov/data/AlphaQ/fof/m000-499.fofproperties"
fof_particle_file = "/media/luna1/dkorytov/data/AlphaQ/halo_prtcls/m000-499.haloparticles"
core_file = "/home/lynchg/dm-deficient/outputs/499/01_22_19.499.deficientcores#0.npy"


print "Loading fof data"
fof_data = getData(fof_file, 'fof_halo_tag', 'fof_halo_count')
print "Done loading!"

print "Loading core data"
cores = np.load(core_file)
print "Done loading!"

m = (-fof_data['fof_halo_count']).argsort()
fof_data = fof_data[m] #sorted in decreasing order of fof count

fof_data = fof_data[np.isin(fof_data['fof_halo_tag'], cores['fof_halo_tag'])] #strips out out fof haloes that do not have dm deficient cores from core file.

srt_key = {f: i for i, f in enumerate(fof_data['fof_halo_tag'])}



m = fof_particles['fof_halo_tag'] == fof_halo
new_particles = fof_particles[m]
fig, axes = plt.subplots(nrows=1, ncols=1)
Greys = plt.get_cmap('Greys')
axes.hist2d(new_particles['x'], new_particles['y'], bins=100, cmap = Greys)  #norm=LogNorm())
#axes.plot(core['x'], core['y'], 'ro')
axes.set_title("FoF Halo {0}, Coun of {1}".format(fof_halo, len(new_particles)))
plt.show()

'''
for core in cores:
    fof_halo = core['fof_halo_tag']
    m = fof_particles['fof_halo_tag'] == fof_halo
    new_particles = fof_particles[m]
    fig, axes = plt.subplots(nrows=1, ncols=1)
    Greys = plt.get_cmap('Greys')
    axes.hist2d(new_particles['x'], new_particles['y'], bins=128, cmap = Greys,  norm=LogNorm())
    axes.plot(core['x'], core['y'], 'ro')
    axes.set_title("Core {}".format(core['core_tag']))
    plt.show(block=False)

    key = raw_input("What would you like to do? \n 1: Continue 2: Save and continue 3: Save and quit 4: Quit")
    if key == '1':
        plt.close()
    elif key == '2':
        plt.savefig('/home/lynchg/dm-deficient/figures/core_{}.png'.format(core['core_tag']))
        plt.close()
    elif key == '3':
        plt.savefig('/home/lynchg/dm-deficient/figures/core_{}.png'.format(core['core_tag']))
        plt.close()
        break 
    elif key == '4':
        break
''' 
