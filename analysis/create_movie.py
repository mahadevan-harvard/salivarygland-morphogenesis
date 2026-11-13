
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.collections as mc
import matplotlib as mpl

cmap = mpl.colormaps['coolwarm']

def angle_fn(R):     
	"""
	Calculate signed angle along an open filament.
	
	Parameters
	----------
	R : ndarray
		Array of shape (N, 2), where each row is a node position [x, y].

	Returns
	-------
	theta : ndarray
		Array of shape (N-2,), containing the signed angles at interior nodes.
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

# Load data from npz file
foldername = "washout"
filename = "shape_array_r_5_z_0.015_kP_0.01_kn_5e-06_tg_1e+06_tgi_2e+06_M_1e-07_dt_0.01_exp_V001_washout"

foldername = "phasediagram_kp_z"
filename = "shape_array_r_5_z_0.015_kP_0.01_kn_5e-06_tg_1e+06_tgi_2e+06_M_1e-07_dt_0.1_exp_V001_k12"
filename = "shape_array_r_5_z_0.001_kP_0.01_kn_5e-06_tg_1e+06_tgi_2e+06_M_1e-07_dt_0.01_exp_V001_k12"


data = np.load(f"data_selected/output/{foldername}/{filename}.npz")

shapes = data["data"]  # Shape: (Nnodes, 2, Niterations)
edge_list = data["edge_list"]  # List of edges, shape: (Nedges, 2)


l_init = np.linalg.norm(shapes[1:, :, 0] - shapes[:-1, :, 0], axis=1)

# Parameters
Niterations = shapes.shape[2]
hold_frames = 15  # ~1 second at fps=15
frame_sequence = list(range(Niterations))

# Initialize the plot
fig, ax = plt.subplots(figsize=(4,4))
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.set_xlim(-13,13)
ax.set_ylim(-9,16)

ax.set_aspect('equal')  # Set the aspect ratio to equal
ax.axis('off')
edges_plot = mc.LineCollection([], cmap=cmap, norm=plt.Normalize(0, 1), linewidths=3)
ax.add_collection(edges_plot)

def update(frame):
	"""Update function for each frame."""
	ax.set_title(r"$t/\tau_{g} = $" + f"{(frame+1)*0.01:.2f}", y=0.92)

	# Current nodes positions
	nodes = shapes[:, :, frame]

	theta = angle_fn(nodes)
	dtheta = (theta - np.pi)

	bond_dtheta = np.zeros(len(l_init))
	bond_dtheta[0] = dtheta[0]
	bond_dtheta[-1] = dtheta[2]
	bond_dtheta[1:-1] = (dtheta[:-1] + dtheta[1:]) / 2.0
	rel_dtheta = (bond_dtheta / (0.1 * np.pi)) + 0.5

	curvature = dtheta / (l_init[:-1] + l_init[1:]) # Excludes left and right nodes
	bond_curvature = np.zeros(len(l_init))
	bond_curvature[0] = curvature[0]
	bond_curvature[-1] = curvature[2]
	bond_curvature[1:-1] = (curvature[:-1] + curvature[1:]) / 2.0

	rel_curvature = (bond_curvature / 2.0) + 0.5

	dl = np.linalg.norm(nodes[1:] - nodes[:-1], axis=1)
	rel_dl = 3* (dl - l_init) / l_init

	# Create edge segments
	segments = []
	for edge in edge_list:
		segments.append([nodes[edge[0]], nodes[edge[1]]])

	# Update LineCollection
	edges_plot.set_segments(segments)
	edges_plot.set_array(np.array(rel_curvature))  # Set edge colors

	return edges_plot,

# Create animation
anim = FuncAnimation(fig, update, frames=frame_sequence, interval=50, blit=False)

# Save as a movie
anim.save(f"./results/{filename}_z15_k12.mp4", writer="ffmpeg", dpi=300, fps=15)

# Optional: Display the animation
plt.show()