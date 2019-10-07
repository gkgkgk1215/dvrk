import numpy as np
import cv2

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# checkerboard dimension
row = 4
col = 4

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(row,col,0)
objp = np.zeros((row*col,3), np.float32)
objp[:,:2] = np.mgrid[0:row,0:col].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
cnt = 0

try:
    print ("camera ON")
    cap = cv2.VideoCapture(0)
except:
    print ("camera failed")

while True:
    ret, img = cap.read()
    if not ret:
        print ("video reading error")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (row,col), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners,(11,11),(-1,-1),criteria)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (row,col), corners2,ret)

    cv2.imshow('video', img)
    key = cv2.waitKey(1) & 0xFF
    if key == 13:   # ENTER
        if ret == True:
            cnt += 1
            objpoints.append(objp)
            imgpoints.append(corners2)
            print ("Corner captured: %d trials" %(cnt))
        else:
            print ("Corner not captured, try again")
    elif key == 27: # ESD
        cap.release()
        cv2.destroyAllWindows()
        break

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
np.savez('calib.npz', ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
print "calibration data has been saved"
