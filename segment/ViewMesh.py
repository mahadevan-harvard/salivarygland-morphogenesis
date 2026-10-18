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

"""

import numpy as np
import scipy
from scipy import ndimage as ndi
from scipy import linalg
from skimage import measure
import matplotlib.pyplot as plt

import trimesh
from trimesh.curvature import discrete_gaussian_curvature_measure, discrete_mean_curvature_measure, sphere_ball_intersection
import open3d as o3d

from scipy.sparse import csc_matrix, linalg as sla
from scipy.spatial import cKDTree, KDTree

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
							   AutoMinorLocator, LogLocator)

import napari

#import pyrender

import io
from PIL import Image

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


def compute_mean_curvature_libigl(mesh, radius=5):
	"""
	Compute the mean curvature using libigl.
	"""
	# Convert the mesh to numpy arrays
	v = np.array(mesh.vertices, dtype=np.float64)
	f = np.array(mesh.faces, dtype=np.int32)

	v1, v2, k1, k2 = igl.principal_curvature(v, f, radius=radius)
	mean_curvature = 0.5 * (k1 + k2)

	return mean_curvature

if __name__ == "__main__":

	voxel_size_x = 152.1729/97
	voxel_size_y = 389.0332/512
	voxel_size_z = 389.0332/512

	# 1.6 um steps -> z


	original_spacing = np.array([voxel_size_x, voxel_size_y, voxel_size_z])
	spacing = np.array([1.0, 1.0, 1.0])
	normalized_spacing = original_spacing / voxel_size_x
	
	# 1 = cells
	# 2 = membrane
	
	Nframes = 120
			
	start = 0
	stop = start+1#Nframes
	count = start
	step = 1
	rotate_mesh = True

	zdim = 151
	xdim = 512
	ydim = 512

	image_size_x = 512
	image_size_y = 512	
	image_size_z = 151

	# For some reason the spacing seems of: check in the metadata using fiji

	for frame in range(start,stop,step):

		print("Loading File")
		FileName = f"./Results/250401-SMG1-BB94-5uM_cp/mesh_epithelial_{frame:03d}.ply"
		mesh = trimesh.load(FileName)

		mesh.apply_translation(np.array([-0.5*image_size_z,-0.5*image_size_x,-0.5*image_size_y]))

		angle = np.pi / 2
		direction = [0, 1, 0]
		center = [0, 0, 0]

		rot_matrix = trimesh.transformations.rotation_matrix(angle, direction, center)
		mesh.apply_transform(rot_matrix)

		angle = -np.pi / 2
		direction = [0, 0, 1]
		center = [0, 0, 0]

		rot_matrix = trimesh.transformations.rotation_matrix(angle, direction, center)
		mesh.apply_transform(rot_matrix)

		if rotate_mesh:
			angle = np.pi
			direction = [0, 1, 0]
			center = [0, 0, 0]

			rot_matrix = trimesh.transformations.rotation_matrix(angle, direction, center)
			mesh.apply_transform(rot_matrix)

		# Create a scene with the mesh
		scene = trimesh.Scene(mesh)
		scene.set_camera(center=(0,0,0))

		print(scene.lights[0].color)

		# Adjust the camera parameters directly
		scene.camera.fov = [60, 60]  # Field of view, adjust if needed
		scene.camera.z_far = 10000.0  # Set far clipping plane distance

		light_distance = 500

		# Define the light positions and intensities
		light_positions = [
			[-light_distance, light_distance, 0.20*light_distance],   # Light in front of the mesh
			[-light_distance, -light_distance, -0.15*light_distance]   # Light behind the mesh
		]

		# Reduce the light intensity to avoid overexposure
		light_intensity = 0.0001 # Adjust intensity to a lower value


		# Create and add the lights to the scene
		for i, position in enumerate(light_positions):
			direction = np.array(position) / np.linalg.norm(position)  # Normalize the direction
			light = trimesh.scene.lighting.DirectionalLight(
				name=f'light_{i}',
				color=np.array([25, 25, 25, 255], dtype=np.uint8),  # White light
				intensity=light_intensity
			)
			translation_matrix = trimesh.transformations.translation_matrix(position)
			scene.graph.update(frame_from=scene.graph.base_frame, frame_to=light.name, matrix=translation_matrix)
			scene.lights.append(light)

		# Print light positions
		for light in scene.lights:
			transform = scene.graph[light.name][0]
			position = trimesh.transformations.translation_from_matrix(transform)
			print(f"Light '{light.name}' position: {position}")


		# Get and print camera position
		if scene.has_camera:
			camera_transform = scene.camera_transform
			camera_position = trimesh.transformations.translation_from_matrix(camera_transform)
			print(f"Camera position: {camera_position}")


		# Show the scene
		scene.show()

		# if rotate_mesh:

		# 	with open(f'scene_screenshot_back_{frame}.png', 'wb') as f:
		# 		f.write(scene.save_image(resolution=(1024, 1024)))

		# else:

		# 	with open(f'scene_screenshot_{frame}.png', 'wb') as f:
		# 		f.write(scene.save_image(resolution=(1024, 1024)))

		count +=1

