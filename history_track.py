import sys
import numpy as np
import genericio as gio
import time
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/lynchg/dm-deficient/density_calc/')
from local_density_MPI import getData
from matplotlib.colors import LogNorm

fof_particle_file = "/media/luna1/dkorytov/data/AlphaQ/halo_prtcls/m000-487.haloparticles"

fof_particles = getData(fof_particle_file, 0, 1, "x", "y", "z", "fof_halo_tag")

fof_tag = fof_particles['fof_halo_tag'][0]

mask = fof_particles['fof_halo_tag'] != 1064323159

fof_particles = fof_particles[mask]
print fof_particles

fig, axes = plt.subplots(nrows=1, ncols=1)

axes.scatter(fof_particles['x'], fof_particles['y'], norm=LogNorm())
axes.set_title("FoF halo {}".format(fof_tag))

plt.draw()
plt.show()
