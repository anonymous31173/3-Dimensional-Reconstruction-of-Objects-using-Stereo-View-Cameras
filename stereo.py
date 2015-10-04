import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt



#cameramatrix1=np.matrix(' 720.5535592 0 227.06678576;0 691.72691907 249.74921404;0 0 1')

#cameramatrix2=np.matrix(' 298.98505814 0 50.60641107;0 936.20963932 188.29212135;0 0 1')

#distcoeffs1 = ('-0.44954109  2.82864811  0.00702934  0.01597158 -5.69974952')
#distcoeffs2 = ('0.08469182  0.03184524 -0.0362362   0.06559929 -0.01019292')
cameraMartix1=[]
cameraMatrix2=[]
distCoeffs1=[]
distCoeffs2=[]
rvecs1=[]
rvecs2=[]
tvecs1=[]
tvecs2=[]
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
#objpoints= [] 
imgpoints2 = []
corners21= []
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
i=0
for fname in glob.glob('image24*.jpg'):
	print fname	
	print(np.size(objpoints))
	print(np.size(imgpoints1))
	img1 = cv2.imread(fname)
	gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
	cv2.imshow('image',img1)
	
    # Find the chess board corners
	ret, corners11 = cv2.findChessboardCorners(gray1,(8,6),None)
	print str(ret)
    # If found, add object points, image points (after refining them)
	if ret == True: 
		objpoints.append(objp)
		corners12 = cv2.cornerSubPix(gray1,corners11,(11,11),(-1,-1),criteria)
		imgpoints1.append(corners12)
        # Draw and display the corners
		img1 = cv2.drawChessboardCorners(img1, (8,6), corners12,ret)
		cv2.imshow('img',img1)
		cv2.waitKey(500)
	i=i+1
i=0
for fname in glob.glob('image14*.jpg'):
	print fname	
	print(np.size(objpoints))
	print(np.size(imgpoints2))
	img2 = cv2.imread(fname)
	gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
	cv2.imshow('image2',img2)
	
    # Find the chess board corners
	ret, corners21 = cv2.findChessboardCorners(gray2,(8,6),None)
	print str(ret)
    # If found, add object points, image points (after refining them)
	if ret == True: 
		corners22 = cv2.cornerSubPix(gray2,corners21,(11,11),(-1,-1),criteria)
		imgpoints2.append(corners22)
        # Draw and display the corners
		img2 = cv2.drawChessboardCorners(img2, (8,6), corners22,ret)
		cv2.imshow('img',img2)
		cv2.waitKey(500)
	i=i+1
cv2.destroyAllWindows()	

ret, cameramatrix2, distcoeffs2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints2, (320,240),None,None)
ret, cameramatrix1, distcoeffs1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints1, (320,240),None,None)
R1,Jacobian1=cv2.Rodrigues(rvecs1[1])
R2,Jacobian2=cv2.Rodrigues(rvecs2[1])
ret,cameraMatrix1,distCoeffs1,cameraMatrix2,distCoeffs2,R,T,E,F=cv2.stereoCalibrate(objpoints,imgpoints1,imgpoints2,(320,240),None,None,None,None)#cameramatrix1,distcoeffs1,cameramatrix2,distcoeffs2)
#cv2.StereoRectify(cameramatrix1,distcoeffs1,cameramatrix2,distcoeffs2,(320,240))#,R,T, R1, R2, P1, P2, Q, CV_CALIB_ZERO_DISPARITY, -1, (0, 0))
print str('Camera Matrix 1:') + str(cameramatrix1)
print str('Camera Matrix 2:')+str(cameramatrix2)
print str('Distortion Coefficients 1:')+str(distcoeffs1)
print str('Distortion Coefficients 2:')+str(distcoeffs2)
print str('Relative Rotational Matrix:')+str(R)
print str('Relative Translational Matrix:')+str(T)

#print Q


#stereo = cv2.createStereoBM(numDisparities=16, blockSize=15)
#disparity = stereo.compute(img1,img2)
#imshow('disparity',disparity)
#cv2.ReprojectImageto3D(disparity,_3DImage,Q,handleMissingValues=0)
#np.savetxt('Q_mat.txt',Q)

plt.im
