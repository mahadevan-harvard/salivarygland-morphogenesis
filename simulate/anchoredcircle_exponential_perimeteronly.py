import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import jax_md

import interactions

import json

import time
import sys


def run_simulation(inputFile, outputPath=""):

		################################################
		# Import simulation parameters
		################################################
	
		try:
			with open(inputFile) as f:
				p = json.load(f)
		except FileNotFoundError:
			exit("File not found")
	
		# Unpack
		Ey = p["scaling"]["Ey"]
		mu = p["scaling"]["mu"] # Damping coefficient
		d  = p["scaling"]["d"] # Cell size [-]
	
		r0 = p["geometry"]["r0"] / d # Inner radius
		h0 = p["geometry"]["h0"] / d # Outer radius
		z0_basis = p["geometry"]["z0_basis"] / d # Now set to 1/1000 of a cell
		z0 = z0_basis*p["geometry"]["z"] / d
	
	
		kappa_nucleus = p["material"]["kappa_nucleus"]
		motility = p["material"]["motility"]
		k_A = p["material"]["k_A"]
		k_P = p["material"]["k_P"]
	
		N = p["discretization"]["N"]
	
		dt = p["time"]["dt"]
		tau_g = p["time"]["tau_g"]
		Nsave = p["time"]["Nsave"]
		Nupdate = p["time"]["Nupdate"]
		seed = p["time"]["seed"]
		Version = p["time"]["V"]

		# set seeds
		key = jax.random.PRNGKey(seed)  

		# Derived quantities (ALWAYS computed)
		R0 = r0 + h0
		y_cut = -0.9 * R0
		theta_cut = 2 * np.arccos(y_cut / R0)
		L0 = R0 * theta_cut
		l0 = L0 / N
	
		segment_motility = motility * l0
		segment_mu = mu * l0	
	
		# Number of cells
		Ntot = R0**2 * (theta_cut - np.sin(theta_cut))/2
		N_O = h0*L0
	
		theta00 = np.pi
		A0 = Ntot
		AO_0 = N_O
		P0 = L0
	
		# Simulation parameters
		segment_motility = motility * l0
		segment_mu = mu * l0
	
		# time related parameters
		tau_g_i = 2*tau_g
		t_final = 1.50*tau_g
	
		################################################
		# Initial conditions
		################################################
	
		# Number of bonds (segments along the filament)
		N0 = int(np.round(L0 / l0))  # Total number of points along the arc
		Nbonds = N0 - 1             # Number of bonds (connections)
	
		np_r = np.zeros((N0,2))
		bonds = np.zeros((Nbonds,2),dtype=int)
		l = np.zeros((Nbonds))
	
		# Generate initial structure (circle of connected springs)
		for i in range(N0):
			# Compute angle for the current node along the top arc
			theta = -0.5*np.pi + np.pi - theta_cut / 2 + (i / (N0 - 1)) * theta_cut  # Angles span the top arc
	
			np_r[i,0] = R0*np.cos(theta)
			np_r[i,1] = R0*np.sin(theta)
	
	
		# Define bonds (connections between nodes)
		for i in range(Nbonds):
			bonds[i, 0] = i       # Start of bond
			bonds[i, 1] = i + 1   # End of bond
			# Compute bond lengths
			l[i] = np.linalg.norm(np_r[bonds[i, 1]] - np_r[bonds[i, 0]])
	
		P0 = np.sum(l)
		A0 = interactions.compute_area(np_r)
		AO_0 = P0*h0
		AI_0 = A0 - AO_0
	
		# Define constrained nodes (ends of the filament)
		np_constrained = np.zeros((N0, 2), dtype=bool)
		np_constrained[0, :] = True  # Constrain the first node
		np_constrained[1, :] = True  # Constrain the first node
		np_constrained[-1, :] = True  # Constrain the last node
		np_constrained[-2, :] = True  # Constrain the last node
	
		# Initial angles and thickness
		theta0 = np.ones(N0-1)*theta00
		z = np.ones(Nbonds)*z0 #
	
		# Convert to jnp lists
		bond_list = jnp.copy(bonds)
		l0_list = jnp.copy(l)
		z_list = jnp.copy(z)
		r = jnp.copy(np_r)
		constrained = jnp.copy(np_constrained)
		unconstrained = ~constrained
	
		# Construct the neighbour list
		displacement_fn, shift_fn = jax_md.space.free()
		neighbor_fn = jax_md.partition.neighbor_list(displacement_fn, box=30, r_cutoff=20*l0) # was 30
		neighbor_list = neighbor_fn.allocate(r)
	
		################################################
		# Construct the model
		################################################
	
		# Define energy functions
		area_energy_fn = interactions.make_area_energy(k_A=k_A)
		perimeter_energy_fn = interactions.make_perimeter_energy(k_P=k_P)
		stretching_energy_fn = interactions.make_stretching_energy(E=Ey)
		bending_energy_fn = interactions.make_bending_energy(E=Ey,kappa_nucleus=kappa_nucleus, theta0=np.pi)
	
		# Combined energy
		def combined_energy_fn(R, length, thickness, target_area, target_perimeter):
			return stretching_energy_fn(R, l0=length, z=thickness) + bending_energy_fn(R, l0=length, z=thickness) + area_energy_fn(R, A0=target_area) + perimeter_energy_fn(R, P0=target_perimeter)
	
		print('Initial energy of the system, U = {:f}'.format(combined_energy_fn(r, l0_list, z_list, A0, P0)))
		print('Area energy of the system, U = {:f}'.format(area_energy_fn(r, A0=A0)))
		print('Bond energy of the system, U = {:f}'.format(stretching_energy_fn(r, l0=l0_list, z=z_list)))
		print('Bending energy of the system, U = {:f}'.format(bending_energy_fn(r, l0=l0_list, z=z_list)))
		print('Perimeter energy of the system, U = {:f}'.format(perimeter_energy_fn(r,P0=P0)))
	
		# Define force functions
		bending_force = jax.grad(bending_energy_fn)
		stretching_force = jax.grad(stretching_energy_fn)
		area_force = jax.grad(area_energy_fn)
		perimeter_force = jax.grad(perimeter_energy_fn)
		repulsion_force = interactions.make_force_repulsion_segment(l0=l0,N=N)
	
		@jax.jit
		def combined_force_fn(R, length, thickness, target_area, target_perimeter, neighbor_list):
			force =  (
				- bending_force(R, l0=length, z=thickness) 
				- stretching_force(R, l0=length, z=thickness) 
				- area_force(R, A0=target_area) 
				- perimeter_force(R, P0=target_perimeter) 
				+ repulsion_force(R, neighbor_list)
			)
			return force
	
		# Create simulation stepper
		@jax.jit
		def simulation_step(r, key, neighbor_list, length, thickness, target_area, target_perimeter, unconstrained, dt, mu, motility):
			
			# Compute deterministic forces
			F = combined_force_fn(r, length, thickness, target_area, target_perimeter, neighbor_list)
			F = jnp.where(unconstrained, F, 0)
	
			# Generate active stochastic force
			key, subkey = jax.random.split(key)
			noise = jax.random.normal(subkey, shape=r.shape,dtype=jnp.float64)
			dr_active = jnp.sqrt(2 * motility * dt / mu) * noise
			dr_active = jnp.where(unconstrained, dr_active, 0)
	
			# Position update
			r = r + (F / mu) * dt + dr_active
	
			return r, key
	
		@jax.jit
		def update_neighbor_list(neighbor_list, r):
			neighbor_list = neighbor_list.update(r)
			return neighbor_list
	
		@jax.jit
		def scan_step(carry, _):
			r, key, t, neighbor_list, l0_list, z_list, unconstrained, dt, segment_mu, segment_motility = carry
	
			# Compute target area and perimeter
			target_area = AI_0 + AO_0
			target_perimeter = P0 * jnp.exp(t / tau_g)
	
			# Perform a simulation step
			r, key = simulation_step(
				r, key, neighbor_list, l0_list, z_list, target_area, target_perimeter,
				unconstrained, dt, segment_mu, segment_motility
			)
	
			# Increase time
			t += dt
	
			return (r, key, t, neighbor_list, l0_list, z_list, unconstrained, dt, segment_mu, segment_motility), jnp.zeros(())
	
		# Run Nbatch steps using jax.lax.scan
		def run_batch(r, key, t, neighbor_list, l0_list, z_list, unconstrained, dt, segment_mu, segment_motility, Nbatch):
			carry = (r, key, t, neighbor_list, l0_list, z_list, unconstrained, dt, segment_mu, segment_motility)
			carry, _ = jax.lax.scan(scan_step, carry, jnp.arange(Nbatch))
			return carry  # Updated carry values
	
		####################################
		# Run simulation
		####################################
	
		Nsteps = int(np.round(t_final/dt))						# Number of simulations steps 
		saveSteps = (np.floor(Nsteps/Nsave)).astype(int)		# Number steps between savepoints
		if saveSteps < Nupdate:									# If saveSteps / Nupdate
			Nupdate = saveSteps
		batchSteps = (np.floor(Nsteps/Nupdate)).astype(int)		# Number of batchStep - with Nupdate the number of simulation steps between updates

		# array to store output	
		shapes = np.zeros([N,2,Nsave+1])
		gstep = 0

		start = time.time()	
		t = 0.0
		for i in range(batchSteps):

			if (i*Nupdate)%saveSteps == 0:
				shapes[:,:,gstep] = np.copy(r)
				print(f"{i*Nupdate}/{Nsteps}")
	
				target_area = AI_0 + AO_0
				target_perimeter = P0 * jnp.exp(t / tau_g) 
	
				print(f"{target_perimeter} {interactions.compute_perimeter(r)} {target_area} {interactions.compute_area(r)}" )
	
				gstep+=1

			(r, key, t, neighbor_list, _, _, _, _, _, _) = run_batch(r, key, t, neighbor_list, l0_list, z_list, unconstrained, dt, segment_mu, segment_motility, Nupdate)
	
			# Update the neighbor list with the new positions
			neighbor_list = update_neighbor_list(neighbor_list, r)
	
			if neighbor_list.did_buffer_overflow:
				print("reallocate")
				neighbor_list = neighbor_fn.allocate(r)

		end = time.time()

		# Save final config		
		shapes[:,:,gstep] = np.copy(r)
		print(f"{i*Nupdate}/{Nsteps}")		
		target_area = AI_0 + AO_0
		target_perimeter = P0 * jnp.exp(t / tau_g)
		print(f"{target_perimeter} {interactions.compute_perimeter(r)} {target_area} {interactions.compute_area(r)} \n" )
	
		print('Energy of the system, U = {:f}'.format(combined_energy_fn(r, l0_list, z_list, target_area, target_perimeter)))
		print(f"Execution time: {end - start:.4f} seconds")
	
		np.savez_compressed(f"{outputPath}/shape_array_r_{r0:.2g}_z_{z0:.2g}_kP_{k_P:.2g}_kn_{kappa_nucleus:.2g}_tg_{tau_g:.2g}_tgi_{tau_g_i:.2g}_M_{motility:.2g}_dt_{dt}_exp_V{Version:03d}.npz",data=shapes, edge_list=bond_list)

		jax.clear_caches()

if __name__ == "__main__":

	try:
		inputFile = sys.argv[1]
	except IndexError:
		exit("No input filename provided")

	run_simulation(inputFile)