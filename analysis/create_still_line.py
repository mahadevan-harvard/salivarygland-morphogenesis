import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import matplotlib as mpl

cmap = mpl.colormaps['plasma']

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
z = 0.01
kP = 0.01
kn=5e-6
dt = 0.1
tg=1e6
folder = "standard_sweep_k12"

data = np.load(f"./output/{folder}/shape_array_r_5_z_{z}_kP_{kP}_kn_{kn}_tg_{tg:.2g}_tgi_{2*tg:.2g}_M_1e-07_dt_{dt}_exp_V001_k12.npz")

shapes = data["data"]  # Shape: (Nnodes, 2, Niterations)
edge_list = data["edge_list"]  # List of edges, shape: (Nedges, 2)

l_init = np.linalg.norm(shapes[1:, :, 0] - shapes[:-1, :, 0], axis=1)

# Select a specific frame
frame = 150

# Initialize the plot
fig, ax = plt.subplots(figsize=(1,1))
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.set_aspect('equal')

edges_plot = mc.LineCollection([], color="k")
ax.add_collection(edges_plot)

# Current nodes positions
nodes = shapes[:, :, frame]

ax.plot(nodes[:,0],nodes[:,1],lw=.5,c="k")
ax.axis('off')

# Save as PDF
plt.savefig(f"./results/{folder}_snapshot_frame_{frame}_z_{z}_kp_{kP}_kn_{kn}_line.pdf", format="pdf", transparent=True, bbox_inches="tight")

plt.show()