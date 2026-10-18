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

def add_bounding_box(viewer, shape, scale):
	# shape = (z, y, x), scale = (sz, sy, sx)
	z, y, x = shape
	sz, sy, sx = scale

	corners = np.array([
		[0,   0,   0],
		[0,   0,   x* sx],
		[0,   y* sy, 0],
		[0,   y* sy, x* sx],
		[z* sz, 0,   0],
		[z* sz, 0,   x* sx],
		[z* sz, y* sy, 0],
		[z* sz, y* sy, x* sx],
	])

	edges = [
		[0, 1], [0, 2], [1, 3], [2, 3],  # bottom
		[4, 5], [4, 6], [5, 7], [6, 7],  # top
		[0, 4], [1, 5], [2, 6], [3, 7],  # verticals
	]

	lines = [corners[e] for e in edges]
	viewer.add_shapes(lines, shape_type='line', edge_color='yellow', edge_width=1)

def showNapari(image, spacing):
	aspect_xz = spacing[0]/spacing[1]
	aspect_yz = spacing[0]/spacing[2]

	# Create an empty viewer
	viewer = napari.Viewer(ndisplay=3)
	
	# Add the image with scaling
	layer = viewer.add_image(
		image,
		scale=(aspect_xz, 1, 1),
		rendering='attenuated_mip'  # <-- set rendering mode
	)
	
	# Optional: make sure the correct layer is selected
	viewer.layers.selection.active = layer

	# Add bounding box
	add_bounding_box(viewer, image.shape, (aspect_xz, 1, 1))


	napari.run()

if __name__ == "__main__":

	voxel_size_x = 1.0 #155.72/94
	voxel_size_y = 1.3811#0.8545552 #385.15/512
	voxel_size_z =  1.3811# 0.8545552 #385.15/512

	# 1.6 um steps -> z


	original_spacing = np.array([voxel_size_x, voxel_size_y, voxel_size_z])
	spacing = np.array([1.0, 1.0, 1.0])
	normalized_spacing = original_spacing / voxel_size_x
	
	# 1 = cells
	# 2 = membrane
	
	Nframes = 283
			
	start = 165
	stop = start+1#Nframes
	count = start
	step = 1
	rotate_mesh = False

	zdim = 166
	xdim = 512
	ydim = 512

#	image = np.zeros([zdim, 2*xdim, 2*ydim])

	# For some reason the spacing seems of: check in the metadata using fiji

	for frame in range(start,stop,step):

		print("Loading File")
		FileName = f"./Results/250916-SMG2-BB94-washout_cp_model0907/thresholded_epithelial_{frame:03d}.npz"
		epithelialCells = np.load(FileName)['data']

		showNapari(epithelialCells,original_spacing)

		count +=1

