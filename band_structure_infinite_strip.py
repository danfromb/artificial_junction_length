# -*- coding: utf-8 -*-
"""
Generate band structure of an infinite strip with additional atoms coupled to the edge.
Not used in the publication and only for additional insight into the system.
"""

import kwant
import numpy as np
import matplotlib.pyplot as plt


W = 10
t = -1
spinOrbit = -0.75
chemicalPotential = 0.0
decorationOnsite = 0
hybridization = 0.5

sig_0 = np.array([[1, 0],[0, 1]])
sig_x = np.array([[0, 1],[1, 0]])
sig_y = np.array([[0, -1j],[1j, 0]])
sig_z = np.array([[1, 0],[0, -1]])

lat = kwant.lattice.honeycomb()
a, b = lat.sublattices
hoppings = (((0, 0), a, b), ((0, 1), a, b), ((-1, 1), a, b))
hoppingsSpinOrbit = (((-1, 1), a, a), ((0, 1), a, a), ((1, 0), a, a), 
                     ((1, -1), a, a), ((0, -1), a, a), ((-1, 0), a, a), 
                     ((-1, 1), b, b), ((0, 1), b, b), ((1, 0), b, b), 
                     ((1, -1), b, b), ((0, -1), b, b), ((-1, 0), b, b))

def onsite(site):
    return (chemicalPotential * sig_0)

def onsiteDecoration(site):
    return ((chemicalPotential + decorationOnsite) * sig_0)

def spinOrbitHopping(site1, site2):
    return 1j * spinOrbit / (3 * np.sqrt(3)) * haldanePhases(site1, site2) * sig_z

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
        
def calcAndPlotWaveFunctionsForSpecificMomentum(sys, unitCell):  
    coords = kwant.plotter.sys_leads_pos(unitCell, kwant.plotter.sys_leads_sites(unitCell, 0)[0]) 
    pointArray = getPointArray(coords)   
   
    modes = sys.modes(energy = 0.15)
    print(modes[0].momenta)
    
    y = np.arange(W)
        
    plt.figure()
    for i in range(4):
        vec = modes[0].wave_functions[0::4,i]
        func = []
        for j in y:
            func.append(vec[pointArray[0][j]])
        #plt.plot(y, np.abs(func)**2, label = "{}".format(i))
        plt.plot(y, np.log(np.abs(func)**2), label = "{}".format(i))
    #plt.legend(loc = 'upper left')
    plt.legend(loc = 'upper center')
    
def getIndex(coords, point, tol = 0.1):
    #print(point)
    for i in range(coords.shape[0]):
        if np.abs(coords[i][0] - point[0]) < tol and np.abs(coords[i][1] - point[1]) < tol:
            return i
            
def getPointArray(coords):
    pointArray = np.empty([1, W], dtype=int)
    #print(coords)
    for j in range(W):
        pointArray[0][j] = getIndex(coords, a(0, j).pos)
        #pointArray[i][0] = getIndex(coords, b(i, 0).pos)
    return pointArray

def makeSystem():
    sym = kwant.lattice.TranslationalSymmetry((1, 0))
    sys = kwant.Builder(sym)
    unitCell = kwant.Builder()
    # do not jet implement the edges
    sys[(a(0, j) for j in range(1, W))] = onsite
    sys[(b(0, j) for j in range(W - 1))] = onsite
    unitCell[(a(0, j) for j in range(W))] = 0
    unitCell[(b(0, j) for j in range(W))] = 0
    sys[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings]] = t * sig_0
    sys[[kwant.builder.HoppingKind(*hopping) for hopping in hoppingsSpinOrbit]] = spinOrbitHopping
    # edges / passivation attoms and coupling to them
    sys[a(0, 0)] = onsiteDecoration
    sys[b(0, W - 1)] = onsiteDecoration
    sys[a(0, 0), b(0, 0)] = hybridization * sig_0
    sys[a(0, W - 1), b(0, W - 1)] = hybridization * sig_0
    return sys, unitCell

def main():
    sys, unitCell = makeSystem()
    sysF = sys.finalized()
    unitCellF = unitCell.finalized()
    #kwant.plot(sys)
    kwant.plotter.bands(sysF)
    #calcAndPlotWaveFunctionsForSpecificMomentum(sysF, unitCellF)
    
main()
