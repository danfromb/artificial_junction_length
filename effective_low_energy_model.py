# -*- coding: utf-8 -*-
"""
An effective low energy model of the Josephson junction capturing much of the same physics.
Not used in the publication and only used for further insight into the system.
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

outputFileName = 'SuperconductingPhaseEnergyRelation.txt'

L = 50
W = 50
t = -1
spinOrbit = -0.75
superconductingGap = 0*0.1
SCGap = 0
offSet = 0*0.01
hybridization = 0.5

sig_0 = np.array([[1, 0],[0, 1]])
sig_x = np.array([[0, 1],[1, 0]])
sig_y = np.array([[0, -1j],[1j, 0]])
sig_z = np.array([[1, 0],[0, -1]])

tau_00 = np.kron(sig_0, sig_0)
tau_0x = np.kron(sig_0, sig_x)
tau_0y = np.kron(sig_0, sig_y)
tau_0z = np.kron(sig_0, sig_z)
tau_x0 = np.kron(sig_x, sig_0)
tau_y0 = np.kron(sig_y, sig_0)
tau_z0 = np.kron(sig_z, sig_0)
tau_zz = np.kron(sig_z, sig_z)

tau_x00 = np.kron(sig_x, tau_00)
tau_y00 = np.kron(sig_y, tau_00)
tau_z00 = np.kron(sig_z, tau_00)
tau_z0x = np.kron(sig_z, tau_0x)
tau_z0y = np.kron(sig_z, tau_0y)
tau_z0z = np.kron(sig_z, tau_0z)
tau_zzz = np.kron(sig_z, tau_zz)

lat = kwant.lattice.square()

def onsite(site, phase = np.pi):
    return (offSet * tau_z00
            + 4 * t * tau_z0z
            + spinOrbit * tau_zzz
            + np.cos(superconductingPhase(site, phase)) * superconductingRegion(site) * superconductingGap * tau_x00
            + np.sin(superconductingPhase(site, phase)) * superconductingRegion(site) * superconductingGap * tau_y00)
 
def hopping():
    return -t * tau_z0z
    
def hoppingX():
    return -1j * t * tau_z0x
    
def hoppingY():
    return -1j * t * tau_z0y
    
def superconductingPhase(site, phase):
    (x, y) = site.pos
    if x <= L/2:
        return 0
    else:
        return phase
       
def superconductingRegion(site):
    (x, y) = site.pos
    if x > L/2 - SCGap/2 and x < L/2 + SCGap/2:
        return 0
    else:
        return 1

def calcWaveFunctions(sys, args, n = 10, sig = 0):
    ham_mat = sys.hamiltonian_submatrix(args, sparse=True)
    return sla.eigsh(ham_mat.tocsc(), k = n, which = "LM", sigma = sig, return_eigenvectors = True, tol = 0.001)
    
def calcLowestEigenEnergies(sys, args):
    ev, evecs = calcWaveFunctions(sys, args, n = 32, sig = 0)
    return ev

def plotWaveFunction(sys, vec):
    kwant.plotter.map(sys, np.abs(vec)**2, colorbar=True, oversampling=1)

def calcAndPlotTotalWaveFunctionAroundEnergy(sys, args):
    ev, evecs = calcWaveFunctions(sys, args, n = 16, sig = 0.0)
    print(ev)
    print(ev/superconductingGap)
    for i in range(1):
        vec = np.sqrt(np.abs(evecs[0::8, i])**2 + np.abs(evecs[1::8, i])**2 + np.abs(evecs[2::8, i])**2 + np.abs(evecs[3::8, i])**2)
        #vec = np.sqrt((np.abs(evecs[0::16, i])**2 + np.abs(evecs[1::16, i])**2 + np.abs(evecs[2::16, i])**2 + np.abs(evecs[3::16, i])**2
        #            + np.abs(evecs[4::16, i])**2 + np.abs(evecs[5::16, i])**2 + np.abs(evecs[6::16, i])**2 + np.abs(evecs[7::16, i])**2))    
        plotWaveFunction(sys, vec)

def calcAndExportSuperconductingPhaseRelation(sys):
    numberOfPhases = 50
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
        print("{:.2f} sec for phase {} of {} in rank {}".format(time.clock() - t0, i, numberOfPhases, rank))
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

def makeSystem():
    sys = kwant.Builder()
    # do not jet implement the edges
    sys[(lat(i, j) for i in range(L) for j in range(W))] = onsite
    sys[[kwant.builder.HoppingKind((1, 0), lat, lat)]] = hopping() + hoppingX()
    sys[[kwant.builder.HoppingKind((0, 1), lat, lat)]] = hopping() + hoppingY()
    return sys

def makeContinuusSystem():
    sym = kwant.lattice.TranslationalSymmetry((1, 0))
    sysContinuum = kwant.Builder(sym)
    for j in range(-W, W, 1):
        sysContinuum[lat(0, j)] = onsite
        sysContinuum[lat(1, j), lat(0, j)] = hopping() + hoppingX()
        if j > -W:
            sysContinuum[lat(0, j), lat(0, j - 1)] = hopping() + hoppingY()
    return sysContinuum

def main():
    if rank == 0:
        t = time.clock()
    sys = makeSystem()
    sysF = sys.finalized()
    sysCon = makeContinuusSystem()
    sysConF = sysCon.finalized()
    #kwant.plot(sys)
    #calcAndPlotTotalWaveFunctionAroundEnergy(sysF, [np.pi])
    #calcAndExportSuperconductingPhaseRelation(sysF)
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-1, 1)
    kwant.plotter.bands(sysConF, ax = ax)
    if rank == 0:
        print("\n overall time: {:.2f}".format(time.clock() - t))
    
if __name__ == '__main__':
    main()
