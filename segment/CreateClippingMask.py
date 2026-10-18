"""
Quick annotation script
Soft Math Lab (Harvard University)
Author: Justin Tauber

Collaboration with Shaohe Wang
"""

import time

import numpy as np
import pandas as pd
from pathlib import Path

import tifffile as tiff

import napari

from skimage.draw import polygon2mask


if __name__ == "__main__":

	FileFolder = "250916-SMG2-BB94-washout"
	FileName = "250916-Col4a2-mNG-BB94-washout-003_2_t"
	ImageName = FileName #
	FilePath1 = f"./Data/{FileFolder}/{FileName}split_c1_rcan/"
	FilePath2 = f"./Data/{FileFolder}/{FileName}split_c2_rcan/"
	FilePath3 = f"./Data/{FileFolder}/{FileName}split_seg_model0907/"

	AnnotationPath = f"./Annotation/{FileFolder}/"
	annotationDir = Path(AnnotationPath)
	annotationDir.mkdir(exist_ok=True)

	# Dimensions of the voxels @@@ Can we read this from tiff?
	voxel_size_x = 1.0
	voxel_size_y = 1.3811#1.5071#0
	voxel_size_z = 1.3811#1.5071#0.8545552#1.5071

	# Spacing used for plotting in Napari
	original_spacing = np.array([voxel_size_x, voxel_size_y, voxel_size_z])
	normalized_spacing = original_spacing / voxel_size_y
#	normalized_spacing = np.array([1,1.,1.])

	# Range for analysis			
	start = 0
	stop = 218
	count = 0
	step = 50
	prev_flag = False
	restart_flag = False

# To consider:
# Check if we should run large components at some point
#  See if we can parralelize sections of the code that are very intensive

# Remove bright regions on inside?
# Enhance the primary membrane segmentation somehow

##########################################################
# LOAD-FRAME
##########################################################
global_start_time = time.time()

# Get the last image 
i = stop
print(f"\nLoading frame {i}/{stop}")

tif_data3 = tiff.imread(f"{FilePath3}{ImageName}{i:03d}-seg.tif")					
image3 = tif_data3[:,:,:] / (2**16 - 1)
z_mean_image_stop = image3.mean(axis=0)

# Get the first image
i = start
print(f"\nLoading frame {i}/{stop}")
tif_data3 = tiff.imread(f"{FilePath3}{ImageName}{i:03d}-seg.tif")					
image3 = tif_data3[:,:,:] / (2**16 - 1)
z_mean_image_start = image3.mean(axis=0)

zdim = np.shape(image3)[0]
xdim = np.shape(image3)[2]
ydim = np.shape(image3)[1]

# Create an empty viewer
viewer = napari.Viewer()

# Show 2D image
viewer.add_image(z_mean_image_start, name='start', colormap='magenta', blending='additive')
viewer.add_image(z_mean_image_stop, name='stop', colormap='green', blending='additive')


# Add shapes layer for drawing polygons
viewer.add_shapes(name='polygon_regions', shape_type='polygon')

napari.run()

if 'polygon_regions' in viewer.layers:
	# polygons = viewer.layers['polygon_regions'].data

	# if len(polygons) > 0:
	# 	# Assume only one shape for now
	# 	poly = polygons[0]

	# 	# Create a 2D mask from the polygon
	# 	mask_2d = polygon2mask((ydim, xdim), poly)

	# 	# Extrude to full 3D (same shape as your z-stack)
	# 	mask_3d = np.repeat(mask_2d[np.newaxis, :, :], zdim, axis=0)

	# 	# Save mask
	# 	tiff.imwrite(annotationDir / f"reference_mask.tif", mask_3d.astype(np.uint8))

	layer = viewer.layers['polygon_regions']
	if len(layer.data) > 0:
		# one 2D mask per shape, then union
		masks = layer.to_masks(mask_shape=(ydim, xdim))   # napari handles shape types
		mask_2d = masks.any(axis=0)

		mask_3d = np.repeat(mask_2d[None, :, :], zdim, axis=0)
		tiff.imwrite(annotationDir / "reference_mask.tif", mask_3d.astype(np.uint8))