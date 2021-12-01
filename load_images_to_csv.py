import cv2
import os
import glob
import numpy as np
from pandas import Series, DataFrame


'''
0 - Chanheum Park
1 - Feelgyun
2 - Taeksang

3 - 
4 -




'''

path_dataset = "./dataset"
path_labels  = [ "0", "1", "2" ]
path_resolu  = [ "64", "128" ]
label_align  = { "non_align": 0, "align" : 1 }

# ./dataset/0/64/non_align/sss.png
# ./dataset/0/64/align/sss.png

def save_csv_first(output_filename, nparray):
	data = DataFrame(nparray)
	data = data.drop_duplicates()
	data.to_csv(output_filename, index=False)
	print("{} saved after duplication from {}".format(len(data), len(nparray)))
	return len(data)

def save_csv_append(output_filename, nparray):
	if (len(nparray) == 0):
		return 0
	if (len(nparray.shape) == 1):
		nparray = nparray.reshape((1, len(nparray)))
		#print("reshaped : {}".format(nparray.shape))

	
	data = DataFrame(nparray)
	data = data.drop_duplicates()
	csv = data.to_csv(index=False, header=False)
	with open(output_filename, "a") as f:
		f.write(csv)
		f.close()
	print("{} saved after duplication from {}".format(len(data), len(nparray)))
	return len(data)


def load_folder(basefolder, resolution, output_filename):
	total_count = 0
	total_array = np. array([])
	saved = False
	
	for label in path_labels:
		for align in label_align:
			path = basefolder + "/" + label + "/" + resolution + "/" + align + "/"
			targets = path + '*.png'
			#print(targets)
			
			for filename in glob.iglob(targets, recursive=True):
				print("{} : {}({}) {}".format(filename, align, int(label_align[align]), int(label)))
				grayImg = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
				grayImg = grayImg.flatten()
				
				grayImg = np.append(grayImg, int(label_align[align]))				
				grayImg = np.append(grayImg, int(label))
				
				if (len(total_array) == 0):
					total_array = grayImg
				else:
					total_array = np.vstack((total_array, grayImg))

			if(saved == False):
				total_count += save_csv_first(output_filename, total_array)
				saved = True
			else:
				total_count += save_csv_append(output_filename, total_array)					
			total_array = None
			total_array = np. array([])

	return total_count

resol = "64"
load_folder(path_dataset, resol, "new_output.csv")


