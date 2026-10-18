import numpy as np
import tifffile


if __name__ == "__main__":

	voxel_size_x = 1.0 #155.72/94
	voxel_size_y = 0.8545552# 85.15/512
	voxel_size_z = 0.8545552 #385.15/512

	# 1.6 um steps -> z


	original_spacing = np.array([voxel_size_x, voxel_size_y, voxel_size_z])
#	spacing = np.array([1.0, 1.0, 1.0])
	normalized_spacing = original_spacing / 1.0 #voxel_size_y
	
	# 1 = cells
	# 2 = membrane
	
	Nframes = 1#218
			
	start = 0
	stop = Nframes
	count = start
	step = 1
	rotate_mesh = False

#	image = np.zeros([zdim, 2*xdim, 2*ydim])

	dataPath = "/mnt/d/PD2_BuddingMorphogenesis/PaperData/"

	folder = "241104-SMG2_cp_model0907"
	# For some reason the spacing seems of: check in the metadata using fiji

	for frame in range(start,stop+1,step):

		print("Loading File")
		try:
			FileName = f"{dataPath}/Results/{folder}/thresholded_epithelial_{frame:03d}.npz"
			epithelialCells = np.load(FileName)['data']
		except FileNotFoundError:
			print(f"Could not find frame {frame}")
			continue

		OutName = f"{dataPath}/Results/{folder}/thresholded_epithelial_{frame:03d}.tif"

		int_array =  epithelialCells.astype("uint8") * 255
		tifffile.imwrite(OutName,
		                 int_array, 
		                 imagej=True,
		                 resolution=(1/voxel_size_y, 1/voxel_size_z),
						metadata={
						'spacing': 1.0,
						'unit': 'um'
						})  # often convenient for biological image stacks
							

		print(frame)

		count +=1

