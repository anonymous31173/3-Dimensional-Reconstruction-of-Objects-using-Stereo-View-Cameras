import cv2

import numpy as np

cameraMatrix1=[]

cameraMatrix2=[]

cameramatrix1 = np.zeros((3,3), dtype=np.float32)
cameramatrix1[0,0] = 1.0
distcoeffs1 = np.zeros((8,1), np.float32)
cameramatrix2 = np.zeros((3,3), dtype=np.float32)
cameramatrix2[0,0] = 1.0
distcoeffs2 = np.zeros((8,1), np.float32)
R=[8]
T=[8]
E=[]
F=[]
R1=[]
R2=[]
P1=[]
P2=[]
distCoeffs1=[]
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
imgpoints1 = []
corners11= []
objpoints = [] 
imgpoints2 = []
corners21= []
i=0
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
cam1=cv2.VideoCapture(1)
cam2=cv2.VideoCapture(2)
while(cam1.isOpened() or cam2.isOpened()):
	ret1,img1=cam1.read()
	ret2,img2=cam2.read()
	
	gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
	gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
	ret1=cv2.imwrite("image1"+str(i)+".jpg",img1)
	ret2=cv2.imwrite("image2"+str(i)+".jpg",img2)
	ret1,corners1=cv2.findChessboardCorners(gray1,(8,6),None)
	print str(ret1)	
	ret2,corners2=cv2.findChessboardCorners(gray2,(8,6),None)
	

	if ret1==True:
		objpoints.append(objp)
		print np.size(objpoints)
		corners11=cv2.cornerSubPix(gray1,corners1,(11,11),(-1,-1),criteria)
		imgpoints1.append(corners11)
		print np.size(imgpoints1)

	if ret2==True:
		#objpoints.append(objp)
		corners21=cv2.cornerSubPix(gray2,corners2,(11,11),(-1,-1),criteria)
		imgpoints2.append(corners21)
		print np.size(imgpoints2)

	img1=cv2.drawChessboardCorners(img1,(8,6),corners1,ret1)
	img2=cv2.drawChessboardCorners(img2,(8,6),corners2,ret2)
	cv2.imshow('Camera 1',img1)
	cv2.imshow('Camera 2',img2)
		

	i=i+1
	k=cv2.waitKey(10)
	if k==27:
		break
	

#objpoints = [np.asarray(x) for x in objpoints]
#imgpoints1 = [np.asarray(x) for x in imgpoints1]
#imgpoints2 = [np.asarray(x) for x in imgpoints2]
retval, cameraMatrix1, distCoeffs1,R, T= cv2.calibrateCamera(objpoints, imgpoints1,(320,240),None,None)
print R
print cameraMatrix1
#cv.StereoRectify(cameraMatrix1, cameraMatrix2, distCoeffs1, distCoeffs2,(img1.width,img1.height), R, T, R1, R2, P1, P2, Q, CV_CALIB_ZERO_DISPARITY, -1, (0, 0))
#print  Q
#np.savetxt('Q_mat.txt',Q)

cam1.release()
cam2.release()
cv2.destroyAllWindows()
