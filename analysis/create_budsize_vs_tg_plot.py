"""
Code for manuscript "*Basement membrane dynamics governs salivary gland budding morphogenesis" 
Soft Math Lab (Harvard University)
Author: Justin Tauber
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import PlotLibrary as plotlib
from matplotlib.ticker import (MultipleLocator, LogLocator)

from scipy.interpolate import splprep, splev
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import scipy.optimize as opt

cmap = matplotlib.colormaps['plasma']

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
# Functions
######################################################################

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

######################################################################
# Script for plotting data
######################################################################

tg_list = np.array([1e3, 2e3, 3e3, 1e4, 2e4, 3e4, 5e4, 1e5, 2e5, 3e5, 1e6, 2e6, 3e6, 5e6, 1e7])
dt_list = np.array([0.0001, 0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,0.1,0.1,0.1,0.1])
V_list = np.array([1,2,3])

kn = 5e-6
r0 = 5
h0 = 1
L0 = 2*np.pi*(r0+h0)

z_range = np.array([0.001])

for z in z_range:

	xi = 1
	k_eff = ((1/12)*z**3 + kn) / L0**4
	tau_r = xi / k_eff

	N = len(tg_list)
	lambda_dict = {}

	for i, tg in enumerate(tg_list):
		lambda_dict[tg] = []
		for j, V in enumerate(V_list):
			tgi = 2*tg

			# Load data from npz file
			data = np.load(f"data_selected/growthrate_sweep/shape_array_r_5_z_0.001_kP_0.01_kn_5e-06_tg_{tg:.2g}_tgi_{tgi:.2g}_M_1e-07_dt_{dt_list[i]}_exp_V{V:03d}_k12.npz")
			shapes = data["data"]  # Shape: (Nnodes, 2, Niterations)

			if len(shapes) < 150:
				print("Simulation not completed.")
				continue

			invalid_numbers = np.any(np.isnan(shapes))
			if invalid_numbers:
				print("NaNs detected.")
				continue

			Nframes = np.shape(shapes)[2]
			Nbud_array = np.zeros(Nframes)

			for frame in range(Nframes):
				shape = shapes[:,:,frame]

				# Fit a B-spline to the filament
				tck, u = splprep([shape[:, 0], shape[:, 1]], s=0, per=False)  # `s=0` for exact fit
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
				
				Nbud_array[frame] = Nbuds

			Nbuds = Nbud_array[-1]
			
			lambda_dict[tg].append(L0/Nbuds)

	means = [np.mean(lambda_dict[tg]) for tg in tg_list]
	stds = [np.std(lambda_dict[tg], ddof=1) for tg in tg_list]

	for i, tg in enumerate(tg_list):
		values = lambda_dict[tg]
		jitter = np.random.normal(0, 0.1, len(values))
		ax.scatter(
			np.full(len(values), tg_list[i]/tau_r),
			values,
			color='grey',
			s=10,
			lw=0
		)

	# Plot Mean and std
	ax.errorbar(tg_list/tau_r,means,yerr=stds,marker="_",color='k',capsize=3,lw=0,elinewidth=.5,markersize=3,markeredgewidth=.5,capthick=1.5)

	# Fit power-law to data
	if N > 1:
		def lambda_fit(tau_g, lambda_0, alpha):
			return (lambda_0 * tau_g**alpha) / np.sqrt(1 + (lambda_0 * tau_g**alpha)**2)

		p0 = [1.0, 0.5] # Initial guess
		bounds=([0, 0], [2, 1])

		popt, pcov = opt.curve_fit(lambda_fit, tg_list/tau_r, np.array(means)/L0, p0, bounds=bounds)
		a_fit, b_fit = popt
		perr = np.sqrt(np.diag(pcov))		
		sigma_a = perr[0]
		sigma_b = perr[1]		
		print(f"Fitted parameters: a = {a_fit:.3f}, b = {b_fit:.3f}")
		label = (rf"$\alpha={b_fit:.2f}\pm{sigma_b:.2f}$")

		fit_kn = np.logspace(-9,-4,50)
		ax.plot(fit_kn, L0*lambda_fit(fit_kn, *popt), lw=1, color='#333333',zorder=0,label=label)

######################################################################
# Final layout settings
######################################################################

# Final adjustments to the figure
plotlib.set_box(ax, alt_style=True)

ax.set_xlabel(r'$\tau_g/ \tau_r$')
ax.set_ylabel(r'$\lambda$ [$d$]')

ax.set_xscale("log")

ax.set_xlim([1e-9,1e-4])
ax.set_ylim([0,10])

leg = plotlib.set_legend(ax,pos=2)
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(1))
ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1, 10) * 0.1, numticks=10))

plotlib.set_position(ax,x=0.26,y=0.25,width=0.65,height=0.70)

# Save as PDF
plt.savefig(f"results/Nbuds_tg_exp_k12.pdf",format="pdf", bbox_inches="tight",transparent=True)

plt.show()