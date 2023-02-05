# -*- coding: utf-8 -*-
"""
Used to process the numerical data and generate plots
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

labelsize = 20
ticksize = 20

rc('text', usetex=True)        

xticks = [0, np.pi, 2*np.pi]
xticklabels = [r'$ 0 $', r'$ \pi $', r'$ 2\pi $']
yticks = [-.5, 0, .5]
yticklabels = [r'$ -0.5 $', r'$ 0 $', r'$ 0.5 $']
ylim = 1
    
fig = plt.figure()
ax = fig.gca()

superconductingGap = 0.01
numerOfPhases = 44

def plotData():
    vals = np.loadtxt('SuperconductingPhaseEnergyRelationChainHighEpsilon.txt', unpack = False)
    vals1 = np.loadtxt('SuperconductingPhaseEnergyRelationChainLowEpsilon.txt', unpack = False)
    phase = vals.T[0]
    energies = np.sort([vals.T[i] for i in range(1, len(vals.T))], axis = 0)
    energies1 = np.sort([vals1.T[i] for i in range(1, len(vals.T))], axis = 0)
    for i in range(0, len(energies)):
        ax.plot(phase, energies[i]/superconductingGap, 'o', color = 'blue', markersize = 5)
        ax.plot(phase, energies1[i]/superconductingGap, 'o', color = 'red', markersize = 5)
    ax.annotate(r'$ |\epsilon_0 -\mu| = t $', fontsize = labelsize, color = 'blue', rotation = 0, va = 'bottom',
                xy=(0, 0), xycoords='data',
                xytext=(2.4, 0.4), textcoords='data'
               )
    ax.annotate(r'$ |\epsilon_0 -\mu| = t/5 $', fontsize = labelsize, color = 'red', rotation = 0, va = 'bottom',
                xy=(0, 0), xycoords='data',
                xytext=(4.2, 0.05), textcoords='data'
               )
    global ylim
    ylim = 0.985
    
def plotDot():
    vals = np.loadtxt('SuperconductingPhaseEnergyRelationDotHighEpsilon.txt', unpack = False)
    vals1 = np.loadtxt('SuperconductingPhaseEnergyRelationDotLowEpsilon.txt', unpack = False)
    phase = vals.T[0]
    energies = np.sort([vals.T[i] for i in range(1, len(vals.T))], axis = 0)
    energies1 = np.sort([vals1.T[i] for i in range(1, len(vals.T))], axis = 0)
    for i in range(0, len(energies)):
        ax.plot(phase, energies[i]/superconductingGap, 'o', color = 'blue', markersize = 5)
        ax.plot(phase, energies1[i]/superconductingGap, 'o', color = 'red', markersize = 5)
    ax.annotate(r'$ \epsilon_0 = t $', fontsize = labelsize, color = 'blue', rotation = 0, va = 'bottom',
                xy=(0, 0), xycoords='data',
                xytext=(3.2, 0.4), textcoords='data'
               )
    ax.annotate(r'$ \epsilon_0 = 0 $', fontsize = labelsize, color = 'red', rotation = 0, va = 'bottom',
                xy=(0, 0), xycoords='data',
                xytext=(4.5, 0.1), textcoords='data'
               )
    
def plotChannelThreshold():
    phase, vals = np.loadtxt('numericalResults/EPRThresholdChannelHighEnergy.txt', unpack = True, skiprows = 7, delimiter = ',')
    phase1, vals1 = np.loadtxt('numericalResults/EPRThresholdChannelLowEnergy.txt', unpack = True, skiprows = 7, delimiter = ',')
    ax.plot(phase, vals, 'o', color = 'blue', markersize = 1)
    ax.plot(phase1, vals1, 'o', color = 'red', markersize = 1)
    ax.annotate(r'$ |\epsilon_0 -\mu| = t $', fontsize = labelsize, color = 'blue', rotation = 0, va = 'bottom',
                xy=(0, 0), xycoords='data',
                xytext=(2.4, 0.4), textcoords='data'
               )
    ax.annotate(r'$ |\epsilon_0 -\mu| = t/5 $', fontsize = labelsize, color = 'red', rotation = 0, va = 'bottom',
                xy=(0, 0), xycoords='data',
                xytext=(4.2, 0.02), textcoords='data'
               )
    global ylim
    ylim = 0.985
    
def plotDotThreshold():
    phase, vals = np.loadtxt('numericalResults/EPRThresholdDotHighEnergy.txt', unpack = True, skiprows = 7, delimiter = ',')
    phase1, vals1 = np.loadtxt('numericalResults/EPRThresholdDotLowEnergy.txt', unpack = True, skiprows = 7, delimiter = ',')
    ax.plot(phase, vals, 'o', color = 'blue', markersize = 1)
    ax.plot(phase1, vals1, 'o', color = 'red', markersize = 1)
    ax.annotate(r'$ \epsilon_0 = t $', fontsize = labelsize, color = 'blue', rotation = 0, va = 'bottom',
                xy=(0, 0), xycoords='data',
                xytext=(3.2, 0.4), textcoords='data'
               )
    ax.annotate(r'$ \epsilon_0 = 0 $', fontsize = labelsize, color = 'red', rotation = 0, va = 'bottom',
                xy=(0, 0), xycoords='data',
                xytext=(4.5, 0.1), textcoords='data'
               )
    
def plotTest():
    vals = np.loadtxt('SuperconductingPhaseEnergyRelation.txt', unpack = False)
    phase = vals.T[0]
    energies = np.sort([vals.T[i] for i in range(1, len(vals.T))], axis = 0)
    for i in range(0, len(energies)):
        ax.plot(phase, energies[i]/superconductingGap, 'o', color = 'blue', markersize = 5)
    x=np.linspace(0, 2 * np.pi, 100)
    ax.plot(x, 0.5*(x - np.pi) / (1 + 0.1 ))
    ax.plot(x, 0.5*(x - np.pi) / (1 + 0.1 * (1 + 16) ))
        
def plotTotalEnergy(phase, energies):
    fig = plt.figure()
    ax = fig.gca()
    energy = []
    for i in range(len(phase)):
        sum = 0
        for e in energies[:, i]:
            if e > 0:
                sum = sum + e
        energy.append(sum/superconductingGap)
    ax.plot(phase, energy, 'o', color = 'blue', markersize = 5)
    fig = plt.figure()
    ax = fig.gca()
    ax.set_ylim([-1, 1])
    ax.plot(phase, np.gradient(energy)*numerOfPhases/2/np.pi, 'o', color = 'blue', markersize = 5) 
        
#plotData()
#plotDot()
#plotChannelThreshold()
plotDotThreshold()
#plotTest()

ax.tick_params(labelsize = ticksize)

ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, size = ticksize)
ax.set_xlabel(r'$ \phi $', fontsize = labelsize)
ax.set_xlim(0, 2*np.pi)

ax.set_ylabel(r' $ E / \Delta $', fontsize = labelsize, rotation = 90)
ax.set_yticks(yticks)
ax.set_ylim(0, ylim)


plt.tight_layout(pad = .5)
#plt.savefig('out.pdf')
plt.show()
