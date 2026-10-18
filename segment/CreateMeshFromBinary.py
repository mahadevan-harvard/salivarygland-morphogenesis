"""
Potential source of good algorithms in python:
https://github.com/LukeMcCulloch/Python-discrete-differential-geometry


https://brickisland.net/DDGSpring2021/category/assignments/?order=desc

I also install the library pydec, which is a python implementation for discrete exterior calculus


Notes on computing curvature:
https://graphics.stanford.edu/courses/cs468-13-spring/assets/lecture8-nguyen.pdf

https://team.inria.fr/titane/files/2017/04/discrete-surfaces-new.pdf

Potentially Usefull:
https://examples.itk.org/src/nonunit/review/segmentbloodvesselswithmultiscalehessianbasedmeasure/documentation

https://stackoverflow.com/questions/38197212/mesh-decimation-in-python

"""

import numpy as np
import scipy
from scipy import ndimage as ndi
from scipy import linalg
from skimage import measure
from skimage import filters
import matplotlib.pyplot as plt

import trimesh
from trimesh.curvature import discrete_gaussian_curvature_measure, discrete_mean_curvature_measure, sphere_ball_intersection
import open3d as o3d

from scipy.sparse import csc_matrix, linalg as sla
from scipy.spatial import cKDTree, KDTree

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
							   AutoMinorLocator, LogLocator)

import napari



def showNapari(image, spacing):
	aspect_xz = spacing[0]/spacing[1]
	aspect_yz = spacing[0]/spacing[2]

	# Create an empty viewer
	viewer = napari.Viewer()
	
	# Add a new layer containing an image
	viewer.add_image(image);
	
	# Remove all layers to start from scratch
	for l in viewer.layers:
		viewer.layers.remove(l)
		
	# add the image again with scaling
	viewer.add_image(image, scale=(aspect_xz, 1, 1))

	napari.run()

def pad_with(vector, pad_width, iaxis, kwargs):

	pad_value = kwargs.get('padder', 0.0)
	vector[:pad_width[0]] = pad_value
	vector[-pad_width[1]:] = pad_value

def taubin_smoothing(mesh, iterations=1, lambda_=0.5, mu=-0.53):
	"""Apply Taubin smoothing to a mesh."""
	for it in range(iterations):
		vertices = mesh.vertices.copy()
		
		# First pass (Laplacian smoothing)
		laplacian = np.zeros_like(vertices)
		for i, v in enumerate(vertices):
			neighbors = mesh.vertex_neighbors[i]
			laplacian[i] = np.mean(vertices[neighbors], axis=0) - v
		mesh.vertices += lambda_ * laplacian
		
		# Second pass (inflation)
		vertices = mesh.vertices.copy()
		inflation = np.zeros_like(vertices)
		for i, v in enumerate(vertices):
			neighbors = mesh.vertex_neighbors[i]
			inflation[i] = np.mean(vertices[neighbors], axis=0) - v
		mesh.vertices += mu * inflation

	return mesh

# Convert trimesh mesh to open3d mesh
def trimesh_to_open3d(mesh):
	vertices = np.asarray(mesh.vertices)
	triangles = np.asarray(mesh.faces)
	o3d_mesh = o3d.geometry.TriangleMesh()
	o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
	o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)
	return o3d_mesh

# Convert back from open3d mesh to trimesh
def open3d_to_trimesh(o3d_mesh):
	vertices = np.asarray(o3d_mesh.vertices)
	triangles = np.asarray(o3d_mesh.triangles)
	return trimesh.Trimesh(vertices=vertices, faces=triangles)

# Function to estimate principal curvatures
def estimate_principal_curvatures(pcd):
	kdtree = o3d.geometry.KDTreeFlann(pcd)
	points = np.asarray(pcd.points)
	normals = np.asarray(pcd.normals)
	principal_curvatures = np.zeros((len(points), 2))
	
	for i in range(len(points)):
		[k, idx, _] = kdtree.search_knn_vector_3d(points[i], 30)
		if k < 3:
			continue
		neighbors = points[idx[1:], :]
		covariance_matrix = np.cov((neighbors - points[i]).T)
		eigenvalues, _ = linalg.eigh(covariance_matrix)
		principal_curvatures[i] = eigenvalues[-2:]
	
	return principal_curvatures

def area_weighted_curvature(mesh, mean_curvature):
	vertex_areas = np.zeros(len(mesh.vertices))
	for face in mesh.faces:
		face_vertices = mesh.vertices[face]
		face_area = trimesh.triangles.area([face_vertices])[0]
		for vertex in face:
			vertex_areas[vertex] += face_area / 3.0
	
	weighted_curvature = mean_curvature / (vertex_areas + 1e-12)  # Avoid division by zero
	return weighted_curvature

def estimate_mean_curvature(mesh):
	L = trimesh.smoothing.laplacian_calculation(mesh)
	L_csr = csr_matrix(L)
	
	squared_norms = L_csr.multiply(L).sum(axis=1).A1  # Compute the squared norm of each row
	mean_curvature = np.sqrt(squared_norms)       # Take the square root to get the norm
	return mean_curvature

# Adjust neighborhood radius for curvature estimation
def adjusted_curvature(mesh, radius=1.0):
	vertices = np.asarray(mesh.vertices)
	tree = KDTree(vertices)
	adjusted_curvatures = np.zeros(len(vertices))
	
	for i in range(len(vertices)):
		indices = tree.query_ball_point(vertices[i], radius)
		if len(indices) > 1:
			local_vertices = vertices[indices]
			local_mean = np.mean(local_vertices, axis=0)
			adjusted_curvatures[i] = np.linalg.norm(local_mean - vertices[i])
	
	return adjusted_curvatures

def estimate_signed_mean_curvature(mesh):
	L = trimesh.smoothing.laplacian_calculation(mesh)
	L_csr = csr_matrix(L)
	
	laplacian_vectors = L_csr @ mesh.vertices
	signed_curvature = np.einsum('ij,ij->i', laplacian_vectors, mesh.vertex_normals)
	return signed_curvature


def skimage_to_open3d(verts, faces):
	mesh = o3d.geometry.TriangleMesh()   
	mesh.vertices = o3d.utility.Vector3dVector(verts)
	mesh.triangles = o3d.utility.Vector3iVector(faces)
	mesh.compute_vertex_normals()
	return mesh

def showNapari2Im(image1, image2, spacing):
	aspect_xz = spacing[0] / spacing[1]
	aspect_yz = spacing[0] / spacing[2]

	# Create an empty viewer
	viewer = napari.Viewer()

	# Add the first image layer (using the red channel)
	viewer.add_image(image1, scale=(aspect_xz, 1, 1), colormap='magenta', blending='additive', name='Channel 1')

	# Add the second image layer (using the blue channel)
	viewer.add_image(image2, scale=(aspect_xz, 1, 1), colormap='green', blending='additive', name='Channel 2')

	napari.run()

if __name__ == "__main__":

	voxel_size_x = 1.0 #155.72/94
	voxel_size_y = 1.3811 #385.15/512
	voxel_size_z = 1.3811 #385.15/512

	# 1.6 um steps -> z


	original_spacing = np.array([voxel_size_x, voxel_size_y, voxel_size_z])
#	spacing = np.array([1.0, 1.0, 1.0])
	normalized_spacing = original_spacing / 1.0 #voxel_size_y
	
	# 1 = cells
	# 2 = membrane
	
	Nframes = 218
			
	start = 0
	stop = Nframes
	count = start
	step = 1
	rotate_mesh = False

#	image = np.zeros([zdim, 2*xdim, 2*ydim])

	folder = "250916-SMG8-BB94-washout_cp_model0907"
	# For some reason the spacing seems of: check in the metadata using fiji

	for frame in range(start,stop+1,step):

		print("Loading File")
		try:
			FileName = f"./Results/{folder}/thresholded_epithelial_{frame:03d}.npz"
			epithelialCells = np.load(FileName)['data']
		except FileNotFoundError:
			print(f"Could not find frame {frame}")
			continue
		bg = np.invert(epithelialCells)

		sigma = 0.5  # Adjust sigma to control the amount of smoothing
		epithelialCells = filters.gaussian(epithelialCells, sigma= 1.0 / normalized_spacing) # Correct for spacing#ndi.gaussian_filter(epithelialCells.astype(np.float32), sigma=sigma)
		bg = filters.gaussian(bg, sigma= 1.0 / normalized_spacing) # Correct for spacing#ndi.gaussian_filter(epithelialCells.astype(np.float32), sigma=sigma)
		smoothed_image = np.clip(epithelialCells - bg, 0, None)

		# # Create a surface mesh
		padded = np.pad(epithelialCells, 1, pad_with)
		verts, faces, normals, values = measure.marching_cubes(padded,0.63,spacing = normalized_spacing, gradient_direction="ascent")
		verts = np.asarray(verts, dtype=np.float64)
		faces = np.asarray(faces, dtype=np.int32)
		  
		# Step 3: Extract Largest Component
		o3d_mesh = skimage_to_open3d(verts, faces)
		labels = np.array(o3d_mesh.cluster_connected_triangles()[0])
		largest_component = np.argmax(np.bincount(labels))
		triangles_to_remove = [i for i, label in enumerate(labels) if label != largest_component]
		o3d_mesh.remove_triangles_by_index(triangles_to_remove)

		# temp = open3d_to_trimesh(o3d_mesh)
		# temp.show()

		# Step 3.5: Apply initial Taubin smoothing
		smoothing_iterations = 10  # Adjust as needed
		taubin_lambda = 0.5  # Adjust as needed
		taubin_mu = -0.53  # Adjust as needed
		smoothed_mesh = o3d_mesh.filter_smooth_taubin(number_of_iterations=smoothing_iterations, lambda_filter=taubin_lambda, mu=taubin_mu)

		# temp = open3d_to_trimesh(smoothed_mesh)
		# temp.show()

		n_iterations = 11
		decimate_factor = 0.75
		smoothing_iterations = 5  # Adjust as needed          
		for i in range(n_iterations):
			# Decimate Mesh with Quadric Decimation
			target_number_of_triangles = int(decimate_factor*np.asarray(smoothed_mesh.triangles).shape[0])  # Adjust as needed
			decimated_mesh = smoothed_mesh.simplify_quadric_decimation(target_number_of_triangles)          

			# Apply Taubin Smoothing
			smoothed_mesh = decimated_mesh.filter_smooth_taubin(number_of_iterations=smoothing_iterations, lambda_filter=taubin_lambda, mu=taubin_mu)
			print(f"smooth {i}")
		# smoothed_mesh = open3d_to_trimesh(decimated_mesh)
		smoothed_mesh = open3d_to_trimesh(smoothed_mesh)
		print(smoothed_mesh.is_watertight)
		#smoothed_mesh.show()

		# # Calculate genus
		Nvertices = len(smoothed_mesh.vertices)
		Nedges = len(smoothed_mesh.edges)/2
		Nfaces = len(smoothed_mesh.faces)
			
		chi = (Nvertices - Nedges + Nfaces)
		g = (2.-chi)/2

		print(g)

		# Save the mesh with vertex attributes
		output_file = f'./Results/{folder}/mesh_epithelial_{frame:03d}.ply'
		smoothed_mesh.export(output_file)

		print(frame)

		count +=1

