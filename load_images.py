import cv2
import os
import glob
import numpy as np
from pandas import Series, DataFrame

path_dataset = "./dataset"
path_labels  = [ "0", "1", "2" ]
path_resolu  = [ "64", "128" ]
label_align  = { "non_align": 0, "align" : 1 }

# ./dataset/0/64/non_align/sss.png
# ./dataset/0/64/align/sss.png

def load_folder(basefolder, resolution):
	total_array = np. array([])
	
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
	return total_array

resol = "64"
raw_data = load_folder(path_dataset, resol)
print(raw_data)
data = DataFrame(raw_data)
data.to_csv("output.csv", index=False)

