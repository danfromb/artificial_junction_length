# -*- coding: utf-8 -*-
"""
Used to generate the energy phase relations of the Josephson junction including a channel or dot
"""

import os
import kwant
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sla
import scipy.linalg as lin
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
name = MPI.Get_processor_name()

outputFileName = 'SuperconductingPhaseEnergyRelation.txt'

L = 500
W = 10
t = 2/np.sqrt(3)
spinOrbit = 1
superconductingGap = 0.01
SCGap = 10
chemicalPotential = 0
# chain
decorationOnsite = 1 * 0.1
hybridization = 0.1
# dot
#decorationOnsite = 0 * superconductingGap
#hybridization = np.sqrt(SCGap) * 5 * superconductingGap

sig_0 = np.array([[1, 0],[0, 1]])
sig_x = np.array([[0, 1],[1, 0]])
sig_y = np.array([[0, -1j],[1j, 0]])
sig_z = np.array([[1, 0],[0, -1]])

tau_x0 = np.kron(sig_x, sig_0)
tau_y0 = np.kron(sig_y, sig_0)
tau_z0 = np.kron(sig_z, sig_0)
tau_zz = np.kron(sig_z, sig_z)

lat = kwant.lattice.honeycomb()
a, b = lat.sublattices
hoppings = (((0, 0), a, b), ((0, 1), a, b), ((-1, 1), a, b))
hoppingsSpinOrbit = (((-1, 1), a, a), ((0, 1), a, a), ((1, 0), a, a), 
                     ((1, -1), a, a), ((0, -1), a, a), ((-1, 0), a, a), 
                     ((-1, 1), b, b), ((0, 1), b, b), ((1, 0), b, b), 
                     ((1, -1), b, b), ((0, -1), b, b), ((-1, 0), b, b))

def onsite(site, phase = np.pi):
    return (chemicalPotential * tau_z0 
            + masstermSublaticeSign(site) * staggeredPotential(site) * tau_z0
            + np.cos(superconductingPhase(site, phase)) * superconductingRegion(site) * superconductingGap * tau_x0
            + np.sin(superconductingPhase(site, phase)) * superconductingRegion(site) * superconductingGap * tau_y0)

def onsiteDecoration(site, phase = np.pi):
    return ((chemicalPotential + decorationOnsite) * tau_z0 
            + np.cos(superconductingPhase(site, phase)) * superconductingRegion(site) * superconductingGap * tau_x0
            + np.sin(superconductingPhase(site, phase)) * superconductingRegion(site) * superconductingGap * tau_y0)
    
def staggeredPotential(site):
    tag = site.tag
    (x, y) = site.pos
    #push channels away from left and right sample edges
    distance = 5
    if x <= a(distance, tag[1]).pos[0] or x >= a(L - distance, tag[1]).pos[0]:
        return 2
    else:
        return 0
    
def masstermSublaticeSign(site):
    if site.family == a:
        return 1
    else:
        return -1

def spinOrbitHopping(site1, site2, phase):
    return 1j * spinOrbit / (3 * np.sqrt(3)) * haldanePhases(site1, site2) * tau_zz

def haldanePhases(site1, site2):
    delta = site1.tag - site2.tag
    if site1.family == a:
        if delta == (-1, 1) or delta == (1, 0) or delta == (0, -1):
            return -1
        else:
            return 1
    else:
        if delta == (-1, 1) or delta == (1, 0) or delta == (0, -1):
            return 1
        else:
            return -1
 
def superconductingPhase(site, phase):
    tag = site.tag
    (x, y) = site.pos
    if x <= a(L/2, tag[1]).pos[0]:
        return 0
    else:
        return phase
       
def superconductingRegion(site):
    tag = site.tag
    (x, y) = site.pos
    if x > a(L/2, tag[1]).pos[0] - SCGap/2 and x < a(L/2, tag[1]).pos[0] + SCGap/2:
        return 0
    else:
        return 1

def calcWaveFunctions(sys, args, n = 10, sig = 0):
    ham_mat = sys.hamiltonian_submatrix(args, sparse=True)
    return sla.eigsh(ham_mat.tocsc(), k = n, which = "LM", sigma = sig, return_eigenvectors = True)
    
def calcLowestEigenEnergies(sys, args):
    ev, evecs = calcWaveFunctions(sys, args, n = 16, sig = 0)
    return ev

def calcAndExportSuperconductingPhaseRelation(sys):
    numberOfPhases = 44
    if rank == 0:
        superconductingPhases = np.linspace(0, 2 * np.pi, numberOfPhases, endpoint = False)
        chunks = np.array_split(superconductingPhases, comm.size)
    else:
        superconductingPhases = None
        chunks = None
    superconductingPhases = comm.scatter(chunks, root=0)
    exportData = []
    i = 0
    for phi in superconductingPhases:
        numberOfPhases = superconductingPhases.size
        t0 = time.clock()
        ev = calcLowestEigenEnergies(sys, [phi])
        evList = ev.tolist()
        evList.insert(0, phi)
        exportData.append(evList)
        print("{:.2f} sec for phase {} of {} in rank {} on {}".format(time.clock() - t0, i, numberOfPhases, rank, name))
        i = i + 1
    data = comm.gather(exportData, root=0)
    if rank == 0:
        exportData = []
        for i in range(comm.size):
            for list in data[i]:
                exportData.append(list)
        head = ("\n System length: " + str(L)
                + "\n System width: " + str(W)
                + "\n hopping: " + str(t)
                + "\n spin Orbit: " + str(spinOrbit)
                + "\n superconducting gap: " + str(superconductingGap)
                + "\n spatial gap in superconductor: " + str(SCGap)
                + "\n\n superconducting Phase - lowest energy")
        np.savetxt(outputFileName, exportData, header = head)

def calcAndPlotTotalWaveFunctionAroundEnergy(sys, args):
    ev, evecs = calcWaveFunctions(sys, args, n = 1, sig = 0)#0.089)
    print(ev)
    print(ev/superconductingGap)
    for i in range(1):
        vec = np.sqrt(np.abs(evecs[0::4, i])**2 + np.abs(evecs[1::4, i])**2)
        plotWaveFunction(sys, vec)
        
def plotWaveFunction(sys, vec):
    plt.figure()
    axes = plt.axes()
    kwant.plotter.map(sys, np.abs(vec)**2, colorbar=True, cmap='jet', ax=axes, oversampling=1,)# file = 'a.pdf')
    #plt.savefig('a.pdf')

def makeSystem():
    sys = kwant.Builder()
    # do not jet implement the edges
    sys[(a(i, j) for i in range(L) for j in range(1, W))] = onsite
    sys[(b(i, j) for i in range(L) for j in range(W - 1))] = onsite
    sys[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings]] = t * tau_z0
    sys[[kwant.builder.HoppingKind(*hopping) for hopping in hoppingsSpinOrbit]] = spinOrbitHopping
    # edges / passivation attoms and coupling to them
    sys[(a(i, 0) for i in range(L))] = onsiteDecoration  #chemicalPotential * tau_z0
    sys[(b(i, W - 1) for i in range(L))] = onsiteDecoration #chemicalPotential * tau_z0
    sys[((a(i, 0), b(i, 0)) for i in range(L))] = hybridization * tau_z0
    sys[((a(i, W - 1), b(i, W - 1)) for i in range(L))] = hybridization * tau_z0
    
#    sys[a(L/2, 0)] = onsiteDecoration
#    sys[b(L/2, W - 1)] = onsiteDecoration
#    sys[a(L/2, 0), b(L/2, 0)] = hybridization * tau_z0
#    sys[a(L/2, W - 1), b(L/2, W - 1)] = hybridization * tau_z0
    
#    sys[a(L/2 + 1, 0)] = onsiteDecoration
#    sys[b(L/2 + 1, W - 1)] = onsiteDecoration
#    sys[a(L/2 + 1, 0), b(L/2 + 1, 0)] = hybridization * tau_z0
#    sys[a(L/2 + 1, W - 1), b(L/2 + 1, W - 1)] = hybridization * tau_z0
    return sys

def main():
    if rank == 0:
        t = time.clock()
    sys = makeSystem()
    sysF = sys.finalized()
    #kwant.plot(sys)
    calcAndExportSuperconductingPhaseRelation(sysF)
    #calcAndPlotTotalWaveFunctionAroundEnergy(sysF, [1 * np.pi])
    if rank == 0:
        print("\n overall time: {:.2f}".format(time.clock() - t))
    
if __name__ == '__main__':
    main()
