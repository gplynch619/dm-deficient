import sys
import numpy as np
import matplotlib.pyplot as plt
import colormaps as cmaps

def toStellarMass(haloMass):
    #taken from arxiv.org/abs/0909.4305
    x=haloMass/(10**11.4)
    frac = 0.129*(x**-0.926 + x**0.261)**-2.440
    stellar=haloMass*frac

    return stellar

def main():
    sys.path.insert(0, '/home/lynchg/dm-deficient/')
    from matplotlib.colors import LogNorm

    directory = "/home/lynchg/dm-deficient/outputs/"
    filename = sys.argv[1]

    core_file = directory + filename

    particleMass=1e9 

    cores = np.load(core_file)

    fig, axes = plt.subplots(nrows=1, ncols=1)

    plt.rc('text', usetex=True)
    plt.rc('font', **{'family':'serif', 'serif':['Computer Modern'], 'weight':'light'})

    plt.register_cmap(name='viridis', cmap=cmaps.viridis)
    plt.set_cmap(cmaps.viridis)

    #Viridis = plt.get_cmap('viridis')

    #count = np.delete(cores['count'], np.where(cores['count']==0))
    count = cores['count'][cores['count']!=0.0]

    stellarMass=toStellarMass(cores['infall_mass'][cores['count']!=0.0])
    localMass=particleMass*count

    print len(stellarMass)
    print len(localMass)

    axes.hist2d(np.log10(localMass), np.log10(stellarMass), bins=30, norm=LogNorm())

    plt.xlabel(r'$\log(M_{499}) [M_\odot]$', fontsize=16)
    plt.ylabel(r'$\log(M_{infall}) [M_\odot]}$', fontsize=16)
    plt.title(r'Local mass to stellar mass (lonely cores removed), r=.1800', fontsize=16)


    plt.show()
    try:
        if sys.argv[2] == 's':
            fig.savefig('/home/lynchg/dm-deficient/figures/local_stellar_big2.png')
    except:
        pass
if __name__ == "__main__":
    main()
