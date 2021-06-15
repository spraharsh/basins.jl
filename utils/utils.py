import os
import numpy as np
from pele.potentials import HS_WCA
from pele.distance import Distance


def read_xyzdr(fname, etol=1.0, bdim=3):
    coords = []
    radii = []
    stable_atoms = []
    if not os.path.isfile(fname):
        raise IOError("The xyzdr file '{}' does not exist.".format(fname))
    f = open(fname, "r")
    while True:
        xyzdr = f.readline()
        if not xyzdr:
            break
        x, y, z, d, r = xyzdr.split()
        coords.extend([float(x), float(y), float(z)])
        radii.extend([float(d)])
        stable = float(float(r) >= etol)
        for _ in range(bdim):
            stable_atoms.extend([stable])
    return np.array(coords, dtype='d'), np.array(radii, dtype='d'), np.array(stable_atoms, dtype='d')

def read_xyzd(fname):
    coords = []
    radii = []
    if not os.path.isfile(fname):
        raise IOError("The xyzd file '{}' does not exist.".format(fname))
    f = open(fname, "r")
    while True:
        xyzd = f.readline()
        if not xyzd:
            break
        x, y, z, d = xyzd.split()
        coords.extend([float(x), float(y), float(z)])
        radii.extend([float(d)])
    return np.array(coords, dtype='d'), np.array(radii, dtype='d')



if __name__=="__main__":
    s = read_xyzdr("/home/praharsh/Dropbox/research/bv-libraries/basins.jl/potentials/jammed_packings/jammed_packing0.xyzdr")
    print(s)

    th = read_xyzd("/home/praharsh/Dropbox/research/bv-libraries/basins.jl/potentials/hs_wca_start/packing0.xyzd")
    print(th)
    # coords = np.array([-1.26049482, -0.95521073,  1.26534317, -0.3978736 , -2.03585937,
    #    -2.12901647,  2.31972795, -2.13775773,  0.38452674,  1.23081331,
    #     1.69439858, -1.14643869, -0.22386022,  0.22083366, -2.16366322,
    #     2.1621745 ,  0.27984705,  0.40418538, -0.03676139,  0.32428513,
    #     0.06877512, -1.2921155 ,  1.62574596,  1.37820726,  0.86374639,
    #    -1.08976784,  1.48627568, -2.50258403, -2.04333088, -2.47876631,
    #     0.9743414 ,  1.78943571,  1.63232619, -1.63667732, -0.96949313,
    #    -0.83257371, -0.11080114, -2.18505503,  0.06158633, -1.56067732,
    #     1.60050288, -0.88748628,  1.25802853, -1.10362087, -1.0863623 ,
    #     2.50590404,  0.19892436, -2.37223693])
    # radii = np.array([1.92391561, 1.97588283, 1.79582555, 2.05414624, 2.10975251,
    #    2.0876029 , 1.94621551, 2.08321329, 1.85513953, 1.85981893,
    #    2.05620311, 1.75396169, 1.8238525 , 2.03632431, 1.97105859,
    #    2.05234171])
    # rattler_coords = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    #                            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.,
    #                            0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

    coords, radii = th
    


    # coords = np.array([1, 0, 0, 1.1, 0, 0])
    # radii = np.array([1.0, 1.0])
    # box_vec =[6., 6., 6.]
    eps = 1.0
    sca = 0.1186889420813968
    radii = radii/(1+sca)
    box_vec = [5.0353449208573426, 5.0353449208573426, 5.0353449208573426] 
    potential = HS_WCA(eps, sca, radii, 3, box_vec, distance_method = Distance.PERIODIC)
    energy = potential.getEnergy(coords)
    print(energy)