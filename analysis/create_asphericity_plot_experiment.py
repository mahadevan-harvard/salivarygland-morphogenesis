"""
Code for manuscript "*Basement membrane dynamics governs salivary gland budding morphogenesis" 
Soft Math Lab (Harvard University)
Author: Justin Tauber
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib

import PlotLibrary as plotlib

from matplotlib.ticker import (MultipleLocator)


cmap = matplotlib.colormaps['rainbow']

# Plot Figure settings
figSpecs = plotlib.FigureSettings()
figSpecs.set_journal('PNAS')
figSpecs.set_figureHeight(30)

# What is the size of this figure relative to the journal specs
xFraction = 60/178
yFraction = 1.0

width = xFraction*figSpecs.doubleColumn
height = yFraction*figSpecs.figureHeight

# Figure Settings
fig = plt.figure(figsize=(width,height)) 
ax = plt.subplot()

######################################################################
# Script for plotting data
######################################################################

def sphericity(A, V):
	return (np.pi**(1/3) * (6*V)**(2/3)) / A

def asphericity(A, V):
	return 1/sphericity(A, V) - 1

rootFolder = "/mnt/d/PD2_BuddingMorphogenesis/PaperData"

# PLOT CONTROL

Folders = ["241104-SMG2", "241104-SMG4", "241104-SMG5", "241105-SMG1", "241105-SMG2", "241105-SMG3", "241105-SMG4_SG1"]
color = cmap(0.25)
label = r'\SI{0}{\micro\Molar}'

all_curves = []
all_times = []

for i,root in enumerate(Folders):

	folder  = f"{root}_cp_model0907_v2"
	fileName = f"{rootFolder}/Results_MeshVoxelization/{folder}/volume_area_{folder}"
	data = np.genfromtxt(f"{fileName}.txt",skip_header =1)

	time = data[:,1]/60
	V = data[:,2]
	A = data[:,3]
	DeltaA = asphericity(A,V)
	y = 100 * (DeltaA - DeltaA[0])

	all_curves.append(y)
	all_times.append(time)

	ax.plot(time, y, color=color, lw=0.5, alpha=0.5)

# --- Average ---
max_len = max(len(c) for c in all_curves)

Y = np.full((len(all_curves), max_len), np.nan)

for i, c in enumerate(all_curves):
	Y[i, :len(c)] = c

mean_curve = np.nanmean(Y, axis=0)
std_curve = np.nanstd(Y, axis=0)

# Since all time axes have the same spacing, use the longest one
time_avg = max(all_times, key=len)

ax.plot(time_avg, mean_curve, color=color, lw=1.0, label=label)
ax.fill_between(time_avg,mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.2, lw=0)

# PLOT 1 uM DATA

Folders = ["241114-SMG1-BB94-1uM","241114-SMG2-BB94-1uM","241114-SMG3-BB94-1uM","241114-SMG4-BB94-1uM","241114-SMG5-BB94-1uM"] #), "241105-SMG4_SG2"] ##, "241105-SMG1_remeshed"]
color = cmap(0.55)
label = r'\SI{1}{\micro\Molar}'

all_curves = []
all_times = []

for i,root in enumerate(Folders):

	folder  = f"{root}_cp_model0907_v2"
	fileName = f"{rootFolder}/Results_MeshVoxelization/{folder}/volume_area_{folder}"
	data = np.genfromtxt(f"{fileName}.txt",skip_header =1)

	time = data[:,1]/60
	V = data[:,2]
	A = data[:,3]
	DeltaA = asphericity(A,V)
	y = 100 * (DeltaA - DeltaA[0])

	all_curves.append(y)
	all_times.append(time)

	ax.plot(time, y, color=color,  lw=0.5, alpha=0.5)

# --- Average ---
max_len = max(len(c) for c in all_curves)

Y = np.full((len(all_curves), max_len), np.nan)

for i, c in enumerate(all_curves):
	Y[i, :len(c)] = c

mean_curve = np.nanmean(Y, axis=0)
std_curve = np.nanstd(Y, axis=0)

# Since all time axes have the same spacing, use the longest one
time_avg = max(all_times, key=len)

ax.plot(time_avg, mean_curve, color=color, lw=1.0, label=label)
ax.fill_between(time_avg,mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.2, lw=0)

# PLOT 5 uM 

Folders = ["250401-SMG1-BB94-5uM","250401-SMG2-BB94-5uM","250401-SMG3-BB94-5uM","250401-SMG4-BB94-5uM","250401-SMG5-BB94-5uM","250401-SMG6-BB94-5uM","250401-SMG7-BB94-5uM","250401-SMG8-BB94-5uM"]#,
color = cmap(0.99)
label = r'\SI{5}{\micro\Molar}'

all_curves = []
all_times = []

for i,root in enumerate(Folders):

	folder  = f"{root}_cp_model0907_v2"
	fileName = f"{rootFolder}/Results_MeshVoxelization/{folder}/volume_area_{folder}"
	data = np.genfromtxt(f"{fileName}.txt",skip_header =1)

	time = data[:,1]/60
	V = data[:,2]
	A = data[:,3]
	DeltaA = asphericity(A,V)
	y = 100 * (DeltaA - DeltaA[0])

	all_curves.append(y)
	all_times.append(time)

	ax.plot(time, y, color=color, lw=0.5, alpha=0.5)

# --- Average ---
max_len = max(len(c) for c in all_curves)

Y = np.full((len(all_curves), max_len), np.nan)

for i, c in enumerate(all_curves):
	Y[i, :len(c)] = c

mean_curve = np.nanmean(Y, axis=0)
std_curve = np.nanstd(Y, axis=0)

# Since all time axes have the same spacing, use the longest one
time_avg = max(all_times, key=len)

ax.plot(time_avg, mean_curve, color=color, lw=1.0, label=label)
ax.fill_between(time_avg,mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.2, lw=0)

######################################################################
# Final layout settings
######################################################################

# Final adjustments to the figure
plotlib.set_box(ax, alt_style=True)

ax.set_xlabel(r'$t$ [h]')
ax.set_ylabel(r'$\Delta\chi_{3D}$ [\%]')

leg = plotlib.set_legend(ax,pos=2)
leg.set_title("[BB-94]:")
ax.set_xlim([0,16])
ax.set_ylim([-10,100])

ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(50))
ax.yaxis.set_minor_locator(MultipleLocator(10))

plotlib.set_position(ax,x=0.17,y=0.27,width=0.78,height=0.67)

# Save as PDF
plt.savefig(f"results/asphericitychange_evolution_BB94_cleanmesh.pdf",format="pdf", bbox_inches="tight",transparent=True)

plt.show()