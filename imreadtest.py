import numpy as np
import cv2
import glob

i=0
for fname in glob.glob('image1*.jpg'):
	print fname
	img = cv2.imread(fname)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	cv2.imshow('img',img)
	i=i+1
print(str(i))
