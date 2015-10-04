import numpy as np
import cv2 as cv
import cv2.cv as cv1
from VideoCapture import Device
import os


def caliLeftCam():    
args, img_mask = getopt.getopt(sys.argv[1:], '', ['save=', 'debug=',       'square_size='])
args = dict(args)
try: img_mask = img_mask[0]
except: img_mask = '../cpp/img*.jpg'
img_names = glob(img_mask)
debug_dir = args.get('--debug')
square_size = float(args.get('--square_size', 1.0))

pattern_size = (8, 6)
pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 )
pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size

obj_points = []
img_pointsL = []
h, w = 0, 0
for fn in img_names:
    print "processing %s..." % fn,
    imgL = cv.imread(fn, 0)
    h, w = imgL.shape[:2]
    found, corners = cv.findChessboardCorners(imgL, pattern_size)
	    if found:
	        term = ( cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1 )
	        cv.cornerSubPix(imgL, corners, (5, 5), (-1, -1), term)
	    if debug_dir:
	        vis = cv.cvtColor(imgL, cv.COLOR_GRAY2BGR)
	        cv.drawChessboardCorners(vis, pattern_size, corners, found)
	        path, name, ext = splitfn(fn)
	        cv.imwrite('%s/%s_chess.bmp' % (debug_dir, name), vis)
	    if not found:
	        print "chessboard not found"
	        continue
	    img_pointsL.append(corners.reshape(-1, 2))
	    obj_points.append(pattern_points)

	    print 'ok'

rmsL, cameraL_matrix, dist_coefsL, rvecsL, tvecsL = cv.calibrateCamera(obj_points, img_pointsL, (w, h))
print "RMSL:", rmsL
print "Left camera matrix:\n", cameraL_matrix
print "distortion coefficients: ", dist_coefsL.ravel()

newcameramtxL, roi=cv.getOptimalNewCameraMatrix(cameraL_matrix,dist_coefsL,(w,h),1,(w,h))
#undistort
mapxL,mapyL = cv.initUndistortRectifyMap(cameraL_matrix,dist_coefsL,None,newcameramtxL,(w,h),5)
dstL = cv.remap(imgL,mapxL,mapyL,cv.INTER_LINEAR)
return img_pointsL, cameraL_matrix, dist_coefsL
def caliRightCam():

args, img_mask = getopt.getopt(sys.argv[1:], '', ['save=', 'debug=', 'square_size='])
args = dict(args)
try: img_mask = img_mask[0]
except: img_mask = '../cpp/ph*.jpg'
img_names = glob(img_mask)
debug_dir = args.get('--debug')
square_size = float(args.get('--square_size', 1.0))

pattern_size = (7, 5)
pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 )
pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size

obj_points = []
img_pointsR = []
h, w = 0, 0
for fn in img_names:
    print "processing %s..." % fn,
    imgR = cv.imread(fn, 0)
    h, w = imgR.shape[:2]
    found, corners = cv.findChessboardCorners(imgR, pattern_size)
    if found:
        term = ( cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1 )
        cv.cornerSubPix(imgR, corners, (5, 5), (-1, -1), term)
    if debug_dir:
        vis = cv.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv.drawChessboardCorners(vis, pattern_size, corners, found)
        path, name, ext = splitfn(fn)
        cv.imwrite('%s/%s_chess.bmp' % (debug_dir, name), vis)

    if not found:
        print "chessboard not found"
        continue
    img_pointsR.append(corners.reshape(-1, 2))
    obj_points.append(pattern_points)

    print 'ok'

rmsR, cameraR_matrix, dist_coefsR, rvecsR, tvecsR = cv.calibrateCamera(obj_points, img_pointsR, (w, h))
print "RMSR:", rmsR
print "Right camera matrix:\n", cameraR_matrix
print "distortion coefficients: ", dist_coefsR.ravel()
newcameramtxR, roi=cv.getOptimalNewCameraMatrix(cameraR_matrix,dist_coefsR,(w,h),1,(w,h))
# undistort
mapxR,mapyR = cv.initUndistortRectifyMap(cameraR_matrix,dist_coefsR,None,newcameramtxR,(w,h),5)
dstR = cv.remap(imgR,mapxR,mapyR,cv.INTER_LINEAR)
return img_pointsR,obj_points,  cameraR_matrix, dist_coefsR
def Pics():
vc = cv.VideoCapture(2)
retVal, frame = vc.read();
while True :
    if frame is not None:
        imgray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        ret,thresh = cv.threshold(imgray,127,255,1)
        contours, hierarchy = cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        cv.namedWindow("threshold")
        cv.namedWindow("Camera")     
        #cv2.drawContours(frame, contours, -1, (0,255,0), 2)
        cv.imshow("Camera", frame)
        cv.imshow("threshold", thresh)
    rval, frame = vc.read()
    if cv.waitKey(1) & 0xFF == 27:
        break
cv1.DestroyAllWindows()
def LeftCap():
cam = Device(2)
cam.saveSnapshot('imageL.jpg')
fn = 'C:\opencv2.4.8\sources\samples\python2\imageL.jpg'
return fn
def RightCap():
cam = Device(0)
cam.saveSnapshot('imageR.jpg')
fn = 'C:\opencv2.4.8\sources\samples\python2\imageR.jpg'
return fn
def Calculate(Li, Ri, Mat):
img_L = cv.pyrDown( cv.imread(Li) )
img_R = cv.pyrDown( cv.imread(Ri) )
window_size = 3
min_disp = 16
num_disp = 112-min_disp
stereo = cv.StereoSGBM(minDisparity = min_disp,
    numDisparities = num_disp,
    SADWindowSize = window_size,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32,
    disp12MaxDiff = 1,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2,
    fullDP = False
)
print "computing disparity..."
disp = stereo.compute(img_L, img_R).astype(np.float32) / 16.0

print "generating 3d point cloud..."
h, w = img_L.shape[:2]
f = 0.8*w                          # guess for focal length
points = cv.reprojectImageTo3D(disp, Mat)
colors = cv.cvtColor(img_L, cv.COLOR_BGR2RGB)
mask = disp > disp.min()

cv.imshow('left', img_L)
cv.imshow('disparity', (disp-min_disp)/num_disp)

b=6.50
D=b*f/disp
print "The Distance =", D
cv.waitKey()
cv1.DestroyAllWindows()

if __name__ == '__main__':
import sys, getopt
from glob import glob

Img_pointsL, Cam_MatL, DisL = caliLeftCam()
Img_pointsR,obj_points,  Cam_MatR, DisR = caliRightCam()

print "Running stereo calibration..."
retval, Cam_MatL, DisL, Cam_MatR, DisR, R, T, E, F= cv.stereoCalibrate(obj_points, Img_pointsL, Img_pointsR,(384,288))
print"running rectifation..."
RL, Rr, PL, PR, Q, validRoiL, validRoiR = cv.stereoRectify(Cam_MatL, DisL, Cam_MatR, DisR,(384,288), R, T)
Pics()
Li = LeftCap()
Ri = RightCap()
Calculate(Li, Ri, Q)
