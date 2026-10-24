import numpy as np

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

################################################
# Function definitions
################################################

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

def compute_area(R):
	"""
	Compute the area enclosed by the filament and the chord connecting its ends.
	
	Parameters:
	R : ndarray
		Array of shape (N, 2), where each row is [x, y] representing a node position.

	Returns:
	Area : float
		Area of the region enclosed by the filament and the chord.
	"""
	# Extract x and y coordinates
	x = R[:, 0]
	y = R[:, 1]

	# Append the starting node to close the loop with the chord
	x = jnp.append(x, x[0])
	y = jnp.append(y, y[0])

	# Compute area using the shoelace formula
	Area = 0.5 * jnp.abs(jnp.sum(x[:-1] * y[1:]) - jnp.sum(y[:-1] * x[1:]))

	return Area

def compute_area_chain(R):
	"""
	Compute the area enclosed by the filament and the chord connecting its ends.
	
	Parameters:
	R : ndarray
		Array of shape (N, 2), where each row is [x, y] representing a node position.

	Returns:
	Area : float
		Area of the region enclosed by the filament and the chord.
	"""
	h = 5.0

	# Extract x and y coordinates
	x = R[:, 0]
	y = R[:, 1]

	# Close the loop with the chord
	x = jnp.append(x, jnp.array([x[-1], x[0], x[0]]))
	y = jnp.append(y, jnp.array([-h, -h, y[0]]))

	# Compute area using the shoelace formula
	Area = 0.5 * jnp.abs(jnp.sum(x[:-1] * y[1:]) - jnp.sum(y[:-1] * x[1:]))

	return Area


def compute_perimeter(R):
	"""
	Compute the length of an open filament by summing distances between consecutive nodes.

	Parameters
	----------
	R : array
		Shape of the filament (N, 2), where each row is a node position [x, y].

	Returns
	-------
	Perimeter : float
		Total length of the filament.
	"""
	# Compute distances between consecutive nodes
	distances = jnp.sqrt(jnp.sum((R[1:] - R[:-1])**2, axis=1))

	# Sum up the distances to get the total length
	Perimeter = jnp.sum(distances)

	return Perimeter

def compute_Inn(R, neighbor_list,N):
	"""
	Computes the nearest segment index (Inn) for each node using a neighbor list.

	Parameters
	----------
	R : array
		Shape of the filament (N, 2), where each row is a node position [x, y].
	neighbor_list : jax_md.partition.NeighborList
		Precomputed neighbor list for efficient pairwise calculations.

	Returns
	-------
	Inn : array
		Indices of the nearest segments for each node.
	"""
	Inn = jnp.full(N, -1, dtype=jnp.int32)  # Initialize Inn with invalid indices

	def process_node(i, Inn):

		node_neighbors = neighbor_list.idx[i]  # Neighbors for node `i`
		min_dist = jnp.inf  # Initialize minimum distance
		Inn_i = jnp.where(i + 2 < N, i + 2, i - 2)  # Drel_rangeefault Inn for edges
		Inn_i = jnp.int32(Inn_i)

		def process_neighbor(j, carry_inner):
			Inn_i, min_dist = carry_inner
			neighbor = node_neighbors[j]

			# Check if neighbor is valid and not a direct neighbor
			valid = (neighbor < N) & (jnp.abs(neighbor - i) > 1)
			dist = jnp.linalg.norm(R[neighbor] - R[i])

			# Update Inn and min_dist if this neighbor is closer
			new_min_dist = jnp.where(valid & (dist < min_dist), dist, min_dist)
			new_Inn_i = jnp.where(valid & (dist < min_dist), neighbor, Inn_i)

			return new_Inn_i, new_min_dist

		# Loop over neighbors for node `i`
		Inn_i, _ = jax.lax.fori_loop(0, node_neighbors.size, process_neighbor, (Inn_i, min_dist))

		# Update the global Inn array
		Inn = Inn.at[i].set(Inn_i)
		return Inn

	# Loop over all nodes
	Inn = jax.lax.fori_loop(0, N, process_node, Inn)

	return Inn

def force_repulsion_segment_full(R, neighbor_list, l0, N, rel_range=1.0, verb=False) :
	"""
	Computes the normal repulsion force.
	
	Each node is repelled from the closest segment adjacent to its nearest
	node (excluding its neighbours). The repulsion force is then projected
	on the local normal to the filament.
	
	Parameters
	----------
	X : array
		Shape of the filament.
	rel_range : float
		Range of the interaction, relative to the segment length.
	
	Returns
	-------
	array
		Repulsion force.
	"""
	a = l0
	r = rel_range*a
	
	# Compute pairwise distances (excluding self and neighbors)
	Inn = compute_Inn(R, neighbor_list,N)
	
	# Initialize forces
	Tg = jnp.zeros((N, 2))
	Frep = jnp.zeros((N, 2))
	Dnn  = jnp.inf*np.ones(N)
	
	# Compute repulsion forces for nearest-neighbors
	for k in [-1, 1] :
		Inn1 = Inn + k		
		Inn1 = jnp.where(Inn1 > 0, Inn1, Inn + 1)  # If Inn1 < 0, use Inn + 1
		Inn1 = jnp.where(Inn1 < N, Inn1, Inn - 1)  # If Inn1 >= N, use Inn - 1
		L = jnp.sum((R[Inn1] - R[Inn]) * (R - R[Inn]), axis=1) / jnp.sum((R[Inn1] - R[Inn]) ** 2, axis=1)
		L = jnp.clip(L, 0.0, 1.0)
		Rnn = R[Inn] + L[:, None] * (R[Inn1] - R[Inn])
		Dnn1 = jnp.sqrt(jnp.sum((Rnn - R) ** 2, axis=1))
		Dnn1 = jnp.where(Dnn1 > r, jnp.inf, Dnn1)
		
		# Update forces for nodes within interaction range
		# Returning indices of the places where Dnn1 < Dnn and calculating the repulsion force for those
		mask = Dnn1 < Dnn
		Dnn = jnp.where(mask, Dnn1, Dnn)
		Frep = jnp.where(
			mask[:, None],
			(R - Rnn) * ((r / Dnn[:, None]) ** 7 - (r / Dnn[:, None]) ** 4) / Dnn[:, None],
			Frep,
		)

	# Compute tangent vectors for the filament
	Tg = jnp.zeros((N, 2)).at[1:-1].set(R[2:] - R[:-2])
	Tg = Tg.at[0].set(R[1] - R[0])
	Tg = Tg.at[-1].set(R[-1] - R[-2])
	Tg = Tg / jnp.linalg.norm(Tg, axis=1, keepdims=True)

	# Project forces onto the normal direction
	Frep -= jnp.sum(Frep * Tg, axis=1, keepdims=True) * Tg

	return Frep

def make_force_repulsion_segment(*, l0, N):	
	def force_repulsion_segment(R, neighbor_list):
			return force_repulsion_segment_full(R, neighbor_list, l0=l0, N=N)
	return force_repulsion_segment

def area_energy_full(R, k, A0): 
		"""
		Computes the energy penalty based on the deviation of the area
		enclosed by the filament from a target area A0.

		Parameters
		----------
		R : array
			Shape of the filament (N, 2), where each row is a node position [x, y].
		k : float
			Energy penalty coefficient.
		A0 : float
			Target area for the filament.

		Returns
		-------
		Energy : float
			Area energy penalty.
		"""
			
		# Extract x and y coordinates
		x = R[:, 0]
		y = R[:, 1]

		# Close the loop with the chord
		x = jnp.append(x, x[0])
		y = jnp.append(y, y[0])

		# Compute the area using the shoelace formula
		Area = 0.5 * jnp.abs(jnp.sum(x[:-1] * y[1:]) - jnp.sum(y[:-1] * x[1:]))

		# Compute the energy penalty for area deviation
		return 0.5 * (k/A0) * (Area - A0) ** 2

def make_area_energy(*, k_A):	
	def area_energy(R, A0):
			return area_energy_full(R, A0=A0, k=k_A)
	return area_energy

def area_energy_full_chain(R, k, A0): 
		"""
		Computes the energy penalty based on the deviation of the area
		enclosed by the filament from a target area A0.

		Parameters
		----------
		R : array
			Shape of the filament (N, 2), where each row is a node position [x, y].
		k : float
			Energy penalty coefficient.
		A0 : float
			Target area for the filament.

		Returns
		-------
		Energy : float
			Area energy penalty.
		"""

		h = 5.0

		# Extract x and y coordinates
		x = R[:, 0]
		y = R[:, 1]

		# Close the loop with the chord
		x = jnp.append(x, jnp.array([x[-1], x[0], x[0]]))
		y = jnp.append(y, jnp.array([-h, -h, y[0]]))

		# Compute the area using the shoelace formula
		Area = 0.5 * jnp.abs(jnp.sum(x[:-1] * y[1:]) - jnp.sum(y[:-1] * x[1:]))

		# Compute the energy penalty for area deviation
		return 0.5 * (k/A0) * (Area - A0) ** 2

def make_area_energy_chain(*, k_A):	
	def area_energy(R, A0):
			return area_energy_full_chain(R, A0=A0, k=k_A)
	return area_energy

def perimeter_energy_full(R, k, P0):
	"""
	Computes the energy penalty based on the deviation of the filament's
	perimeter from a target perimeter P0.

	Parameters
	----------
	R : array
		Shape of the filament (N, 2), where each row is a node position [x, y].
	k : float
		Energy penalty coefficient.
	P0 : float
		Target perimeter for the filament.

	Returns
	-------
	Energy : float
		Perimeter energy penalty.
	"""
	# Compute distances between consecutive points
	l = jnp.sqrt(jnp.sum((R[1:] - R[:-1]) ** 2, axis=1))
	
	# Sum up distances to get the perimeter
	Perimeter = jnp.sum(l)

	# Compute the energy penalty for perimeter deviation
	return 0.5 * (k/P0) * (Perimeter - P0) ** 2

def make_perimeter_energy(*, k_P):	
	def perimeter_energy(R, P0):
			return perimeter_energy_full(R, P0=P0, k=k_P)
	return perimeter_energy

def stretching_energy_full(R, E, z, l0):
	"""
	Computes the stretching energy of an open filament.

	Parameters
	----------
	R : array
		Shape of the filament (N, 2), where each row is a node position [x, y].
	E : float
		Elastic modulus.
	z : array
		Thickness of the filament at each segment.
	l0 : array
		Rest lengths of the segments.

	Returns
	-------
	Energy : float
		Stretching energy of the filament.
	"""
	# Compute distances between consecutive points
	l = jnp.sqrt(jnp.sum((R[1:] - R[:-1]) ** 2, axis=1))

	# Compute spring stiffness for each segment
	mu = z * E
	k = mu / l0

	# Compute stretching energy
	return jnp.sum(0.5 * k * (l - l0) ** 2)

def make_stretching_energy(*, E):	
	def stretching_energy(R, z, l0):
			return stretching_energy_full(R, z=z, l0=l0, E=E)
	return stretching_energy

# Energy definition bending energy
def bending_energy_full(R, E, z, kappa_nucleus, l0, theta0):
	"""
	Computes the bending energy of an open filament.

	Parameters
	----------
	R : array
		Shape of the filament (N, 2), where each row is a node position [x, y].
	E : float
		Elastic modulus.
	z : array
		Thickness of the filament at each segment.
	kappa_nucleus : float
		Contribution from nucleus stiffness.
	l0 : array
		Rest lengths of the segments.
	theta0 : float
		Target angle between consecutive segments.

	Returns
	-------
	Energy : float
		Bending energy of the filament.
	"""
	# Compute vectors between consecutive nodes
	avec = R[:-2] - R[1:-1]  # Vectors from i+1 to i
	bvec = R[2:] - R[1:-1]   # Vectors from i+1 to i+2

	# Compute magnitudes of vectors
	amag = jnp.sqrt(jnp.sum(avec ** 2, axis=1))
	bmag = jnp.sqrt(jnp.sum(bvec ** 2, axis=1))

	# Compute dot products and angles
	adotb = jnp.sum(avec * bvec, axis=1)
	cosangle = adotb / (amag * bmag)

	# Clamp cosine values to avoid numerical issues
	cosangle = jnp.where(cosangle>=1.0, 1.0, cosangle)
	cosangle = jnp.where(cosangle<=-1.0, -1.0, cosangle) 
#	cosangle = jnp.clip(cosangle, -1.0, 1.0)	# @@@ different
	acosangle = jnp.arccos(cosangle)  # Angle between vectors

	# Compute signed angles using cross product
	acrossb = avec[:, 0] * bvec[:, 1] - avec[:, 1] * bvec[:, 0]
	sinangle = acrossb / (amag * bmag)
	theta = jnp.where(sinangle < 0.0, 2.0 * jnp.pi - acosangle, acosangle)

	# Compute angle deviation
	dtheta = theta - theta0
	dtheta = jnp.where(jnp.abs(dtheta) > jnp.pi, 2.0 * jnp.pi - dtheta, dtheta)

	# Compute bending stiffness for each angle using properties of adjacent edges
	z_av = (z[:-1] + z[1:])/2
	l0_sum = l0[:-1] + l0[1:]

	# Compute the effective stiffness for each angle as the average of left and right edges
	kappa = E * (z_av**3 / (12 * l0_sum))

	# Incorporate nucleus stiffness, averaged over both edges
	kappa_nucleus_norm = kappa_nucleus / (l0_sum)
	kappa = jnp.where(dtheta > 0.0, kappa + kappa_nucleus_norm, kappa)

	# Compute bending energy
	return jnp.sum(kappa * dtheta ** 2)

def make_bending_energy(*, E, kappa_nucleus, theta0):	
	def bending_energy(R, l0, z):
			return bending_energy_full(R, l0=l0, z=z, E=E, kappa_nucleus=kappa_nucleus, theta0=theta0)
	return bending_energy