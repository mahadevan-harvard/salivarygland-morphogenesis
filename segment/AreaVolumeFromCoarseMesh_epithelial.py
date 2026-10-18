import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw, ImageFont
import imageio.v2 as imageio
import os

import trimesh

from scipy import ndimage as ndi

from skimage.filters import threshold_otsu
from skimage import morphology
from skimage import measure
from skimage import filters
from skimage import segmentation

import tifffile as tiff
import itk

from skimage import filters

def load_frame(filepath):
	"""Load a single frame from the specified .npz file."""
	return np.load(filepath)['data']

def pad_to_target_size(image, target_size=(1024, 1024), padding_color=(0, 0, 0)):
	"""
	Pads the input image to the target size without resizing, while preserving the aspect ratio.
	
	Args:
		image (PIL.Image): The input image to be padded.
		target_size (tuple): The target size for the image (width, height).
		padding_color (tuple): The color used for padding (default is black).
	
	Returns:
		PIL.Image: The padded image.
	"""
	# Calculate padding required for both width and height
	width_padding = (target_size[0] - image.width) // 2
	height_padding = (target_size[1] - image.height) // 2

	# Pad the image on all sides (left, right, top, bottom)
	padded_image = ImageOps.expand(image, (width_padding, height_padding, 
										   target_size[0] - image.width - width_padding, 
										   target_size[1] - image.height - height_padding), 
								   fill=padding_color)
	
	return padded_image

# def add_scale_and_time(image, time_in_minutes, spacing):
# 	"""
# 	Adds a scale bar and time indicator to an image.
# 	Args:
# 		image (PIL.Image): The input image.
# 		time_in_minutes (int): The time in minutes since the start.
# 		spacing (list): The voxel spacing for the 3D data.
# 	Returns:
# 		PIL.Image: Image with the scale bar and time indicator added.
# 	"""
# 	draw = ImageDraw.Draw(image)
# 	#font = ImageFont.load_default()  # Load default font
# 	font = ImageFont.truetype("/usr/share/fonts/truetype/msttcorefonts/arial.ttf", size=20)  # Set font with TrueType font


# 	# Scale bar parameters
# 	scale_bar_length_um = 100
# 	spacing = [0.854, 0.854]  # Example spacing
# 	scale_bar_length_pixels = int((612/512)*scale_bar_length_um / spacing[1])
# 	scale_bar_start = 30
# 	scale_bar_y = image.height - 30
# 	scale_bar_end = scale_bar_start + scale_bar_length_pixels

# 	# Draw scale bar
# 	draw.line([(scale_bar_start, scale_bar_y), (scale_bar_end, scale_bar_y)], fill="white", width=5)

# 	# Center scale bar text
# 	scale_text = f"{scale_bar_length_um} µm"
# 	text_bbox = draw.textbbox((0, 0), scale_text, font=font)  # Get the bounding box
# 	text_width = text_bbox[2] - text_bbox[0]
# 	text_height = text_bbox[3] - text_bbox[1]
# 	text_x = scale_bar_start + (scale_bar_length_pixels - text_width) // 2
# 	text_y = scale_bar_y - text_height - 10  # Place above scale bar

# 	draw.text((text_x, text_y), scale_text, fill="white", font=font)


# 	# Time indicator
# 	hours, minutes = divmod(time_in_minutes, 60)
# 	time_text = f"{hours:02d}:{minutes:02d} HH:MM"
# 	time_position = (image.width - 150, 30)  # Top-right corner
# 	draw.text(time_position, time_text, fill="white", font=font)

# 	return image

def overlay_images_green_magenta(img1, img2):
	"""Create an RGB overlay where img1 is green and img2 is magenta."""
	rgb = np.zeros((*img1.shape, 3))
	#rgb[..., 0] = img1  # Red
	rgb[..., 1] = img2  # Green
	#rgb[..., 2] = img1  # Blue (Magenta = Red + Blue)
	
	return rgb

def add_scale_and_time(ax, time_in_minutes, pixel_size=1.0, bar_length_um=100, position=(0.05, 0.05)):
	"""Add scale bar and timestamp to the Matplotlib figure."""
	# Scale bar calculation
	bar_length_pixels = bar_length_um / pixel_size  # Convert μm to pixels
	bar_length_rel = bar_length_pixels/512

	# Add scale bar
	ax.plot([position[0], position[0] + bar_length_rel], [position[1], position[1]], 
			color='white', linewidth=2, transform=ax.transAxes, clip_on=False)

	# Scale bar label
	ax.text(position[0] + bar_length_rel / 2, position[1]+0.075, f"{bar_length_um} µm", 
			color='white', fontsize=6, ha="center", va="top", transform=ax.transAxes)


	# Add time label
	hours, minutes = divmod(time_in_minutes, 60)
	time_text = f"{hours:02d}:{minutes:02d} HH:MM"
	ax.text(0.93, 0.90, f"{time_text}", color='white', fontsize=6,
			transform=ax.transAxes, ha="right", va="bottom")

def rescale_image(image,bottom_cap,top_cap):
	Ifloor = np.percentile(image, 1.0)
	Icap = np.percentile(image, 99.5)

	image[image < bottom_cap*Icap] = bottom_cap*Icap
	image[image > top_cap*Icap] = top_cap*Icap
	image = (image - bottom_cap*Icap) 
	image = image / (Icap*(top_cap - bottom_cap))
	return image

def plateness_computation_spacing(image, spacing):
	"""
	Apply the Hessian-based objectness filter for membrane enhancement,
	adjusted for non-cubic voxels with specified spacing.

	Parameters:
		image (numpy.ndarray): Input 3D grayscale image (scaled 0 to 1).
		spacing (tuple): Tuple specifying voxel spacing (z, y, x).
		
	Returns:
		numpy.ndarray: Output image with plateness enhancement.
	"""
	Dimension = 3
	PixelType = itk.D
	ImageType = itk.Image[PixelType, Dimension]
	HessianPixelType = itk.SymmetricSecondRankTensor[itk.D, Dimension]
	HessianImageType = itk.Image[HessianPixelType, Dimension]

	# Convert NumPy array to ITK image
	itk_image = itk.GetImageFromArray(image.astype(np.float64))
	
	# Set the voxel spacing to account for non-cubic voxels
	itk_image.SetSpacing(spacing)

	# Set up the objectness filter
	objectness_filter = itk.HessianToObjectnessMeasureImageFilter[HessianImageType, ImageType].New()
	objectness_filter.SetBrightObject(True)
	objectness_filter.SetScaleObjectnessMeasure(False)
	objectness_filter.SetAlpha(1.0)
	objectness_filter.SetBeta(1.0)
	objectness_filter.SetGamma(0.05) # 0.01
	objectness_filter.SetObjectDimension(2)  # For plane-like structures

	# Set the multiscale Hessian filter
	multi_scale_filter = itk.MultiScaleHessianBasedMeasureImageFilter[ImageType, HessianImageType, ImageType].New()
	multi_scale_filter.SetInput(itk_image)
	multi_scale_filter.SetHessianToMeasureFilter(objectness_filter)
	multi_scale_filter.SetSigmaMinimum(0.5)
	multi_scale_filter.SetSigmaMaximum(1.5)
	multi_scale_filter.SetNumberOfSigmaSteps(3)

	# Rescale output intensity to the range of unsigned short (16-bit)
	OutputPixelType = itk.US
	OutputImageType = itk.Image[OutputPixelType, Dimension]

	rescale_filter = itk.RescaleIntensityImageFilter[ImageType, OutputImageType].New()
	rescale_filter.SetInput(multi_scale_filter)

	# Get the result
	rescale_filter.Update()
	result_image = rescale_filter.GetOutput()
	plateness = itk.GetArrayFromImage(result_image)

	return plateness

def make_divisible_by_16(value):
	return int(np.ceil(value / 16.0) * 16)

def PNA_based_preprocessing(image, image2, normalized_spacing):

	image_vasculature = np.clip(image - image2, 0, None)	 # image_vasculature = np.clip(image - image2, 0, None)

	# Extract the membrane
	image_membrane = np.clip(image - image_vasculature, 0, None)
	return image_membrane

def largest_component(image):
	# Label connected components
	labels = measure.label(image, connectivity=3)

	# Find the largest component
	props = measure.regionprops(labels)
	largest_component = max(props, key=lambda x: x.area)

	# Create a mask for the largest component
	largest_component_mask = labels == largest_component.label

	return largest_component_mask

def largest_component2D(image):
	# Label connected components
	labels = measure.label(image)

	# Find the largest component
	props = measure.regionprops(labels)
	largest_component = max(props, key=lambda x: x.area)

	# Create a mask for the largest component
	largest_component_mask = labels == largest_component.label

	return largest_component_mask

# Calculate surface area
def surface_area(verts, faces):
	triangles = verts[faces]
	a = np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=1)
	b = np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=1)
	c = np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=1)
	s = 0.5 * (a + b + c)
	return np.sum(np.sqrt(s * (s - a) * (s - b) * (s - c)))


def PNA_based_background_mask(image_membrane, image_PNA, normalized_spacing, preprocess_background=True):

	global_mask = np.ones_like(image_PNA, dtype=bool)

	if preprocess_background:
		# TRY TO CUT OUT THE GLAND
		image2_filtered = filters.difference_of_gaussians(image_PNA,2,10)
		image2_filtered = np.clip(image2_filtered,0,1)

		# Coarse segmentation of the gland area 
		# (for analysis of image series, check if this doesn't cause problems).
		global_threshold = threshold_otsu(image2_filtered[30:,:,:])
		reference_image = image2_filtered > 0.30*global_threshold 
		reference_image[:30,:,:] = False
		binary_mask = largest_component(reference_image)
		image2_f = np.copy(image2_filtered)
		image2_f[~binary_mask] = 0
		project = np.max(image2_f[30:,:,:], axis=0)
		project_median = filters.median(project,morphology.disk(10))
		mask = largest_component2D(project_median > 0)
		mask = morphology.binary_closing(mask,morphology.disk(10))
		global_mask = np.logical_and(mask[np.newaxis, :, :],global_mask)

		# Extract the gland
		distance = ndi.distance_transform_edt(~mask)
		feathered_mask = np.clip( (10 - distance) / 10, 0, 1)
		feathered_mask[mask] = 1  # Preserve the original mask
		smooth_mask = feathered_mask[np.newaxis, :, :]		
		image_membrane = image_membrane*smooth_mask
		image_PNA = image_PNA*smooth_mask
		image2_filtered = image2_filtered*smooth_mask	
	#		showNapari2Im(image_membrane,image_PNA,normalized_spacing)

		# Apply the same procedure to the x,z projection
		reference_image = image2_filtered > 0.30*global_threshold
		binary_mask = largest_component(reference_image)
		image2_f = np.copy(image2_filtered)
		image2_f[~binary_mask] = 0
		project = np.max(image2_f, axis=1)
		project_median = filters.median(project,morphology.disk(10))
		mask = largest_component2D(project_median > 0)
		mask = morphology.binary_closing(mask,morphology.disk(10))
		global_mask = np.logical_and(mask[:, np.newaxis, :],global_mask)

		# Extract the gland
		distance = ndi.distance_transform_edt(~mask)
		feathered_mask = np.clip( (10 - distance) / 10, 0, 1)
		feathered_mask[mask] = 1  # Preserve the original mask
		smooth_mask = feathered_mask[:, np.newaxis, :]
		image_membrane = image_membrane*smooth_mask
		image_PNA = image_PNA*smooth_mask		
		image2_filtered = image2_filtered*smooth_mask
	#		showNapari2D2Im(np.max(image_PNA,axis=1),mask,normalized_spacing)

		# Apply the same procedure to the y,z projection
		reference_image = image2_filtered > 0.30*global_threshold
		binary_mask = largest_component(reference_image)
		image2_f = np.copy(image2_filtered)
		image2_f[~binary_mask] = 0		
		project = np.max(image2_f, axis=2)
		project_median = filters.median(project,morphology.disk(10))
		mask = largest_component2D(project_median > 0)
		mask = morphology.binary_closing(mask,morphology.disk(10))
		global_mask = np.logical_and(mask[:, :, np.newaxis],global_mask)

		# Extract the gland
		distance = ndi.distance_transform_edt(~mask)
		feathered_mask = np.clip( (10 - distance) / 10, 0, 1)
		feathered_mask[mask] = 1  # Preserve the original mask
		smooth_mask = feathered_mask[:, :, np.newaxis]	
		image_membrane = image_membrane*smooth_mask
		image_PNA = image_PNA*smooth_mask
	
	background_mask = np.invert(global_mask)

	return image_membrane, image_PNA, background_mask

def plot_frames_with_savefig(start, stop, step, spacing, FileFolder):
	"""
	Render frames using Mayavi, save each frame to a temporary image file, 
	and save the final movie as an MP4 using imageio.
	
	Args:
		start (int): Start frame index.
		stop (int): Stop frame index.
		step (int): Step size for frames.
		spacing (list or np.ndarray): Spacing for the 3D image.
		output_movie (str): Path to save the output movie.
	"""
	pixel_sizes = spacing   # Voxel spacing
	results = []

	for frame in range(start, stop+1, step):
		print(f"Processing frame {frame}")

#		filepath = f"./Results/241104-SMG5/thresholded_epithelial_{frame:03d}.npz"
#		epithelial_cells = load_frame(filepath)
#		image = epithelial_cells.astype(float)  # Convert bool to float

	 	# Load and normalize the binary image
		FilePath1 = f"./Results/{FileFolder}/thresholded_membrane_{frame:03d}.npz"
		FilePath2 = f"./Results/{FileFolder}/thresholded_epithelial_{frame:03d}.npz"		 

		membrane = load_frame(FilePath1)
		epithelial_cells = load_frame(FilePath2)
				
		image = membrane.astype(float)			
		image2 = epithelial_cells.astype(float)

		voxel_volume = pixel_sizes[0]*pixel_sizes[1]*pixel_sizes[2]

		V = np.sum(image2)*voxel_volume

		verts, faces, _, _ = measure.marching_cubes(image2, level=0.5, spacing=(1, 0.85, 0.85))
		A = surface_area(verts, faces)

		time = frame * 5  # minutes

		results.append((frame, time, V, A))
	
	np.savetxt(f'volume_area_{FileFolder}.txt', results, fmt='%d\t%d\t%.6f\t%.6f', header='frame\ttime_min\tvolume_um3\tsurface_area_um2', comments='')


# Parameters
voxel_size_x = 1.0  # Original spacing in x
voxel_size_y = 0.8545552  # Original spacing in y
voxel_size_z = 0.8545552  # Original spacing in z
spacing = np.array([voxel_size_x, voxel_size_y, voxel_size_z])

# start = 0
# stop = 272  # Change this to your desired stop frame
# step = 1

# # Run the rendering function
# plot_frames_with_savefig(start, stop, step, spacing, "241104-SMG5var1")

Folder = "Results"
#subFolder = "241105-SMG4_SG1_cp_model0907"

# subFolders = ["241114-SMG1-BB94-1uM","241114-SMG2-BB94-1uM","241114-SMG3-BB94-1uM","241114-SMG4-BB94-1uM","241114-SMG5-BB94-1uM"]#,
# Nframes_array = [240,181,120,40,80]

# subFolders = ["241104-SMG2", "241104-SMG4", "241104-SMG5", "241105-SMG1", "241105-SMG2", "241105-SMG3",  "241105-SMG4_SG1", "241105-SMG4_SG2"] 
# Nframes_array = [65, 225,272,216,85,216,125,216]

# subFolders = ["250401-SMG1-BB94-5uM","250401-SMG2-BB94-5uM","250401-SMG3-BB94-5uM","250401-SMG4-BB94-5uM","250401-SMG5-BB94-5uM","250401-SMG6-BB94-5uM","250401-SMG7-BB94-5uM","250401-SMG8-BB94-5uM"]#,
# Nframes_array = [197,197,185,135,197,197,197,190]

# subFolders = ["250311-SMG1-FOV1-BB94-washout","250311-SMG1-FOV2-BB94-washout","250311-SMG2-FOV1-BB94-washout","250311-SMG2-FOV2-BB94-washout","250311-SMG3-FOV2-BB94-washout"]#,
# Nframes_array = [283,120,220,200,155]


# subFolders = ["250402-SMG2-BB94-washout","250402-SMG3-BB94-washout","250402-SMG4-BB94-washout","250402-SMG5-BB94-washout","250402-SMG6-BB94-washout","250402-SMG7-BB94-washout","250402-SMG8-BB94-washout"]#,
# Nframes_array = [100,125,50,110,100,150,20]

subFolders = ["250916-SMG1-BB94-washout","250916-SMG2-BB94-washout","250916-SMG3-BB94-washout","250916-SMG4-BB94-washout","250916-SMG5-BB94-washout","250916-SMG6-BB94-washout","250916-SMG7-BB94-washout","250916-SMG8-BB94-washout"]#,
Nframes_array = [218,160,200,218,218,218,218,218]

for j, subFolder in enumerate(subFolders):

#	subFolder = f"{subFolder}"
#	subFolder = f"{subFolder}_cp"
	subFolder = f"{subFolder}_cp_model0907"

	Nframes = Nframes_array[j] #125
	results = []

	for i in range(Nframes):
	#	fileName = f"./{Folder}/{subFolder}/mesh_epithelial_{i:03d}_v2.ply"
		fileName = f"./{Folder}/{subFolder}/mesh_epithelial_{i:03d}.ply"	
		if os.path.exists(fileName):
			surfaces = trimesh.load(fileName, force='mesh')
			meshes = surfaces.split(only_watertight=False)
			if len(meshes) == 1:
				mesh = meshes[0]
			else:
				mesh = max(meshes, key=lambda m: m.volume if m.is_watertight else m.area)
			
			if mesh.is_watertight:
				volume = mesh.volume
			else:
				volume = np.nan  # or use a repair/estimate function
			area = mesh.area

			time = i*5

			results.append((i, time, volume, area))

	np.savetxt(f'{Folder}/{subFolder}/volume_area_{subFolder}.txt', results, fmt='%d\t%d\t%.6f\t%.6f', header='frame\ttime_min\tvolume_um3\tsurface_area_um2', comments='')

# - Run the analysis for all folders
# - Save the result in the result folders

# - 1 slide: compare volume based versus coarse mesh based
# - compare raw autosegment, CellPose 1, CellPose 2 (for a pretty succesfull case and maybe a bad case (so SMG2 and SMG4 for example))

# - For each gland make a plot with cellpose 1 and cellpose 2

# - Superimpose the Cellpose 2 data (maybe keep the area and volume seperate)

# Make list of followup:
#	- check BB94 concentration dependency (maybe already give a trial) and stress the comparison with the simulations on this level
#	- check washout result