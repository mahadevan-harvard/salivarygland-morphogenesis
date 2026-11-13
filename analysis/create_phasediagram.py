"""
Code for manuscript "*Basement membrane dynamics governs salivary gland budding morphogenesis" 
Soft Math Lab (Harvard University)
Author: Justin Tauber
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib

import PlotLibrary as plotlib

from matplotlib.ticker import (LogLocator)

cmap = matplotlib.colormaps['rainbow']
from matplotlib.offsetbox import DrawingArea, AnnotationBbox
from matplotlib.patches import Wedge

from scipy.interpolate import splprep, splev
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

def calculate_signed_curvature_for_aperiodic_curve(x, y):
	"""
	Calculate signed curvature at each point on an aperiodic curve.
	Curvature is calculated as the signed angle between consecutive segments,
	divided by the average segment length around each point.

	Parameters:
	- x: 1D array of x-coordinates of the curve.
	- y: 1D array of y-coordinates of the curve.

	Returns:
	- signed_curvature: Array of signed curvature values along the curve.
	"""
	# Forward and backward segment vectors
	dx_forward = np.diff(x, append=x[-1])  # No wrapping at the end
	dy_forward = np.diff(y, append=y[-1])
	dx_backward = np.diff(x, prepend=x[0])  # No wrapping at the start
	dy_backward = np.diff(y, prepend=y[0])

	# Stack forward and backward vectors
	forward_vectors = np.column_stack((dx_forward, dy_forward))
	backward_vectors = np.column_stack((dx_backward, dy_backward))

	# Segment lengths
	lengths_forward = np.linalg.norm(forward_vectors, axis=1)
	lengths_backward = np.linalg.norm(backward_vectors, axis=1)

	# Dot products for angle magnitude
	dot_products = np.sum(forward_vectors * backward_vectors, axis=1)
	magnitudes_product = lengths_forward * lengths_backward

	cos_angles = np.divide(
		dot_products,
		magnitudes_product,
		out=np.ones_like(dot_products),
		where=magnitudes_product > 1e-12
	)

	angles = np.arccos(
		np.clip(cos_angles, -1.0, 1.0)
	)

	cross_products = (
		backward_vectors[:, 0] * forward_vectors[:, 1]
		- backward_vectors[:, 1] * forward_vectors[:, 0]
	)

	# Average segment lengths
	average_lengths = (lengths_forward + lengths_backward) / 2

	# Calculate signed curvature
	signed_curvature = angles * np.sign(cross_products) / average_lengths  # Radians per unit length

	# Set curvature to 0 at boundaries where neighbors are undefined
	signed_curvature[0] = 0.0
	signed_curvature[-1] = 0.0

	return signed_curvature

def add_pie(ax, x, y, fractions, colors, size=20):

	da = DrawingArea(size, size, 0, 0)

	radius = size / 2
	center = (radius, radius)

	start = 0
	for frac, color in zip(fractions, colors):
		if frac > 0.01:
			da.add_artist(
				Wedge(
					center,
					radius,
					start * 360,
					(start + frac) * 360,
					facecolor=color,
					edgecolor='k',
					linewidth=0.0
				)
			)
			start += frac

	ab = AnnotationBbox(
		da,
		(x, y),
		frameon=False,
		box_alignment=(0.5, 0.5)
	)

	ax.add_artist(ab)

# Plot Figure settings
figSpecs = plotlib.FigureSettings()
figSpecs.set_journal('PNAS')
figSpecs.set_figureHeight(30)

# What is the size of this figure relative to the journal specs
xFraction = 50/180
yFraction = 1.0

width = xFraction*figSpecs.doubleColumn
height = yFraction*figSpecs.figureHeight

# Figure Settings
fig = plt.figure(figsize=(width,height)) 
ax = plt.subplot()

######################################################################
# Script for plotting data
######################################################################
r0 = 5
h0 = 1
Ey = 1
d=1
kp_array = np.arange(1e-4,5e-1,1e-4)
tau_g = 1e6
tau_g_i = 2*tau_g
t = 1.5*tau_g
kn = 5e-6

L0 = 2*np.pi*(r0+h0)

# ballooning transition
z_upper = kp_array * ( (1 - np.exp(-t/tau_g))/ (np.sqrt( np.exp(t/tau_g_i) + ((4*np.pi*d)/L0)*(np.exp(t/tau_g) - np.exp(t/tau_g_i)) ) - 1) - np.exp(-t/tau_g))
k_upper = Ey*z_upper 

ax.plot(k_upper,kp_array,color="black",lw=1,zorder=100)

ax.fill_between(k_upper,1e-4*np.ones(len(k_upper)),kp_array,color="#f2efcfff",zorder=0)
ax.fill_between(k_upper,kp_array,1e1*np.ones(len(k_upper)),color="#b8deffff",zorder=0)


d = 1                  # Cell size [um]
M = 1e-7

# kn = 5e-6
z0_array = np.array([0.001,0.002,0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03])
kp_array = np.array([0.001, 0.002,0.003, 0.005, 0.008, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1])


for cnt1, k_P in enumerate(kp_array):
	for cnt, z in enumerate(z0_array):
		Nbuds_array = []
		for cnt2,V in enumerate(np.array([1,2,3])):
			tg = tau_g
			tgi = 2*tg
			if z <= 0.002:
				dt = 0.01
			else:
				dt = 0.1
			# Load data from npz file
			data = np.load(f"data_selected/output/phasediagram_kp_z/shape_array_r_5_z_{z}_kP_{k_P}_kn_5e-06_tg_{tg:.2g}_tgi_{tgi:.2g}_M_{M}_dt_{dt}_exp_V{V:03d}_k12.npz")

			shapes = data["data"]  # Shape: (Nnodes, 2, Niterations)

			Nframes = np.shape(shapes)[2]
			Nbud_array = np.zeros(Nframes)

			i = Nframes - 1
			shape = shapes[:,:,i]

			invalid_numbers = np.any(np.isnan(shape))
			if invalid_numbers:
				print("NaNs numbers encountered.")
				continue

			# Fit a B-spline to the filament
			tck, u = splprep([shape[:, 0], shape[:, 1]], s=0.1, per=False)  # `s=0` for exact fit
			path_length = u / np.max(u)  # Normalize path length (0 to 1)

			u_fine = np.linspace(0, 1, 1000, endpoint=False)  # Fine parameter range for smooth B-spline
			x_spline, y_spline = splev(u_fine, tck)  # Evaluate B-spline at fine points
			
			# Calculate curvature (κ) at each point using the B-spline derivatives
			raw_curvature = calculate_signed_curvature_for_aperiodic_curve(x_spline, y_spline) #calculate_curvature_from_angles_and_lengths(angles, segment_lengths)
			curvature = gaussian_filter1d(raw_curvature, sigma=1, mode='constant',cval=-1)  # Adjust sigma as neede

			# Detect local minima and maxima
			minima, _ = find_peaks(-curvature, prominence=1)

			# keep only negative minima
			minima = minima[curvature[minima] < 0]

			# Exclude endpoints
			endpoint_margin = 12

			minima = minima[
				(minima > endpoint_margin)
				& (minima < len(curvature)-endpoint_margin)
			]

			# anchors
			anchors = np.concatenate((
				[0],
				np.sort(minima),
				[len(curvature)-1]
			))

			Nbuds = len(anchors) - 1
			Nbuds_array += [Nbuds]
			print(f"{z} {k_P}:\t {Nbuds}")

		Nbuds_array = np.array(Nbuds_array)
		balloon_fraction = np.sum(Nbuds_array == 1) / len(Nbuds_array)
		bud_fraction = np.sum(Nbuds_array > 1) / len(Nbuds_array)
		
		resistivity = z
		expansivity = k_P

		add_pie(ax, resistivity, expansivity, [bud_fraction,balloon_fraction],["#0f2c87ff", "#dca927ff"],size=3)

######################################################################
# Final layout settings
######################################################################

# Final adjustments to the figure
plotlib.set_box(ax,alt_style=True)

ax.set_xlabel(r'$E z$')
ax.set_ylabel(r'$k_P$')

ax.set_xlim([3e-4,1e-1])
ax.set_ylim([4e-4,4e-1])

ax.set_xscale("log")
ax.set_yscale("log")

ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1, 10) * 0.1, numticks=10))
ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1, 10) * 0.1, numticks=10))

plotlib.set_position(ax,x=0.26,y=0.25,width=0.65,height=0.70)

# Save as PDF
plt.savefig(f"results/phasediagram_z_kp_k12.pdf",format="pdf", bbox_inches="tight",transparent=True)

plt.show()