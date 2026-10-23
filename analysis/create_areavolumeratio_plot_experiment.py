import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import PlotLibrary as plotlib

from matplotlib.ticker import (MultipleLocator)


cmap = matplotlib.colormaps['rainbow']

# Plot Figure settings
figSpecs = plotlib.FigureSettings()
figSpecs.set_journal('PNAS')
figSpecs.set_figureHeight(35)

# What is the size of this figure relative to the journal specs
xFraction = 1.00
yFraction = 1.0

width = xFraction*figSpecs.singleColumn
height = yFraction*figSpecs.figureHeight

# Figure Settings
fig = plt.figure(figsize=(width,height)) 
ax = plt.subplot()

######################################################################
# Script for plotting data
######################################################################

rootFolder = "/mnt/d/PD2_BuddingMorphogenesis/PaperData"

Folders = ["241104-SMG2", "241104-SMG4", "241104-SMG5", "241105-SMG1","241105-SMG2", "241105-SMG3", "241105-SMG4_SG1"]

all_curves_V = []
all_curves_A = []
all_times = []

for i,root in enumerate(Folders):

	folder  = f"{root}_cp_model0907_v2"
	fileName = f"{rootFolder}/Results_MeshVoxelization/{folder}/volume_area_{folder}"	
	data = np.genfromtxt(f"{fileName}.txt",skip_header =1)

	time = data[:,1]/60
	V = data[:,2]
	A = data[:,3]

	dV = (V/V[0])**(2/3) - 1
	dA = (A/A[0]) - 1

	all_curves_V.append(dV)
	all_curves_A.append(dA)	
	all_times.append(time)

	ax.plot(time, 100 * dA, color='C02', lw=.5,alpha=.5)
	ax.plot(time, 100 * dV, color='C01', lw=.5,alpha=.5)

# --- Average ---
max_len = max(len(c) for c in all_curves_V)

YV = np.full((len(all_curves_V), max_len), np.nan)
YA = np.full((len(all_curves_A), max_len), np.nan)
for i, c in enumerate(all_curves_V):
	YV[i, :len(c)] = c
for i, c in enumerate(all_curves_A):
	YA[i, :len(c)] = c

mean_curve_V = np.nanmean(100*YV, axis=0)
std_curve_V = np.nanstd(100*YV, axis=0)
mean_curve_A = np.nanmean(100*YA, axis=0)
std_curve_A = np.nanstd(100*YA, axis=0)

# Since all time axes have the same spacing, use the longest one
time_avg = max(all_times, key=len)

ax.plot(time_avg, mean_curve_A, color='C02', lw=1.0, label=r"$\Delta \tilde{S}$")
ax.fill_between(time_avg,mean_curve_A - std_curve_A, mean_curve_A + std_curve_A, color='C02', alpha=0.2, lw=0)

ax.plot(time_avg, mean_curve_V, color='C01', lw=1.0, label=r'$\Delta \tilde{S}_{\rm iso}$')
ax.fill_between(time_avg,mean_curve_V - std_curve_V, mean_curve_V + std_curve_V, color='C01', alpha=0.2, lw=0)

######################################################################
# Final layout settings
######################################################################
# Final adjustments to the figure
plotlib.set_box(ax, alt_style=True)

ax.set_xlabel(r'$t$ [h]')
ax.set_ylabel(r'$\Delta \tilde{S}$ [\%] ')

plotlib.set_position(ax,x=0.15,y=0.20,width=0.80,height=0.70)

ax.set_xlim([0,16])
ax.set_ylim([-5,75])

ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(25))
ax.yaxis.set_minor_locator(MultipleLocator(5))

leg = plotlib.set_legend(ax,pos=0)

# Save as PDF
plt.savefig(f"results/area_volume_evolution_control_cleanmesh_plus.pdf",format="pdf", bbox_inches="tight",transparent=True)

plt.show()