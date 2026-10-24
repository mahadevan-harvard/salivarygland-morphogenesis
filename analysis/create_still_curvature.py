"""
Code for manuscript "*Basement membrane dynamics governs salivary gland budding morphogenesis" 
Soft Math Lab (Harvard University)
Author: Justin Tauber
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import matplotlib as mpl

cmap = mpl.colormaps['coolwarm']

def angle_fn(R):     
	"""
	Calculate signed angle along an open filament.
	"""
	# Compute vectors between consecutive points
	avec = R[:-2] - R[1:-1]  # Vector from i+1 to i
	bvec = R[2:] - R[1:-1]   # Vector from i+1 to i+2

	# Compute magnitudes of vectors
	amag = np.sqrt(np.sum(avec**2, axis=1))
	bmag = np.sqrt(np.sum(bvec**2, axis=1))

	# Compute cosine of the angle
	adotb = np.sum(avec * bvec, axis=1)
	cosangle = adotb / (amag * bmag)

	# Clamp cosine values to avoid numerical issues
	cosangle = np.clip(cosangle, -1.0, 1.0)

	# Compute angle using arccos
	acosangle = np.arccos(cosangle)

	# Compute sine of the angle (for sign correction)
	acrossb = avec[:, 0] * bvec[:, 1] - avec[:, 1] * bvec[:, 0]
	sinangle = acrossb / (amag * bmag)

	# Correct angle sign based on sine
	theta = np.where(sinangle < 0.0, 2.0 * np.pi - acosangle, acosangle)
	
	return theta

# Load data from npz files
data = np.load("./output/standard_sweep/shape_array_r_5_z_0.001_kP_0.01_kn_0_tg_1e+06_tgi_2e+06_M_1e-07_dt_0.01_expV0.1.npz")

shapes = data["data"]  # Shape: (Nnodes, 2, Niterations)
edge_list = data["edge_list"]  # List of edges, shape: (Nedges, 2)

l_init = np.linalg.norm(shapes[1:, :, 0] - shapes[:-1, :, 0], axis=1)

# Select a specific frame
frame = 149

# Initialize the plot
fig, ax = plt.subplots(figsize=(1,1))
ax.set_xlim(-12, 12)
ax.set_ylim(-10, 14)
ax.set_aspect('equal')

edges_plot = mc.LineCollection([], cmap=cmap, norm=plt.Normalize(0, 1))
ax.add_collection(edges_plot)

# Current nodes positions
nodes = shapes[:, :, frame]

# Calculate curvature
theta = angle_fn(nodes)
dtheta = (theta - np.pi)
curvature = dtheta / (l_init[:-1] + l_init[1:]) # NB. Excludes left and right nodes

bond_dtheta = np.zeros(len(l_init))
bond_dtheta[0] = dtheta[0]
bond_dtheta[-1] = dtheta[2]
bond_dtheta[1:-1] = (dtheta[:-1] + dtheta[1:]) / 2.0
rel_dtheta = (bond_dtheta / (0.1 * np.pi)) + 0.5

bond_curvature = np.zeros(len(l_init))
bond_curvature[0] = curvature[0]
bond_curvature[-1] = curvature[2]
bond_curvature[1:-1] = (curvature[:-1] + curvature[1:]) / 2.0

rel_curvature = (bond_curvature / 6.0) + 0.5

# Create edge segments
segments = []
for edge in edge_list:
	segments.append([nodes[edge[0]], nodes[edge[1]]])

# Update LineCollection
edges_plot.set_segments(segments)
edges_plot.set_array(np.array(rel_curvature))  # Set edge colors
edges_plot.set_linewidth(1.0)

ax.axis('off')

# Save as PDF
plt.savefig(f"results/snapshot_frame_{frame}_z_0.001.pdf", format="pdf", bbox_inches="tight",transparent=True)

plt.show()