import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import (MultipleLocator)

import PlotLibrary as plotlib

cmap = matplotlib.colormaps['Dark2']

# Plot Figure settings
figSpecs = plotlib.FigureSettings()
figSpecs.set_journal('PNAS')
figSpecs.set_figureHeight(35)

# What is the size of this figure relative to the journal specs
xFraction = 0.5
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

# PLOT WASHOUT DATA

Folders = ["250916-SMG1-BB94-washout","250916-SMG2-BB94-washout","250916-SMG3-BB94-washout","250916-SMG4-BB94-washout","250916-SMG5-BB94-washout","250916-SMG6-BB94-washout","250916-SMG7-BB94-washout","250916-SMG8-BB94-washout"]
color = "firebrick"
label = r'BB-94 washout'

all_curves = []
all_times = []

for i,root in enumerate(Folders):

	folder  = f"{root}_cp_model0907_v2"
	fileName = f"{rootFolder}/Results_MeshVoxelization//{folder}/volume_area_{folder}"
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

# PLOT CONTROL
Folders = ["241104-SMG2", "241104-SMG4","241104-SMG5", "241105-SMG1","241105-SMG2", "241105-SMG3", "241105-SMG4_SG1"]
color = "darkgrey"
label = r'Control'

all_curves = []
all_times = []

for i,root in enumerate(Folders):

	folder  = f"{root}_cp_model0907_v2"
	fileName = f"{rootFolder}/Results_MeshVoxelization//{folder}/volume_area_{folder}"
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
plotlib.set_box(ax,alt_style=True)

ax.set_xlabel(r'$t$ [h]')
ax.set_ylabel(r'$\Delta \chi_{3D}$ [\%]')

leg = plotlib.set_legend(ax,pos=2)

ax.set_xlim([0,16])
ax.set_ylim([-10,200])

ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(50))
ax.yaxis.set_minor_locator(MultipleLocator(10))

plotlib.set_position(ax,x=0.10,y=0.20,width=0.85,height=0.70)

# Save as PDF
plt.savefig(f"results/asphericitychange_evolution_washout_cleanmesh.pdf",format="pdf", bbox_inches="tight",transparent=True)

plt.show()