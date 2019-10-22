import numpy as np
import cv2

def drawCube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    cv2.drawContours(img, [imgpts[:4]], -1, (255, 0, 0), -3)

    for i,j in zip(range(4), range(4, 8)):
        cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (0, 255, 0), 2)

    cv2.drawContours(img, [imgpts[4:]], -1, (0, 255, 0), 2)

    return img

def poseEstimation(row, col):
    with np.load('calib_kinect.npz') as X:
        ret, mtx, dist, _, _ = [X[i] for i in ('ret', 'mtx', 'dist', 'rvecs', 'tvecs')]

    termination = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((row*col, 3), np.float32)
    objp[:,:2] = np.mgrid[0:row, 0:col].T.reshape(-1,2)
    axis = np.float32([[0,0,0], [0,row-1,0], [row-1,row-1,0], [row-1,0,0], [0,0,-row+1], [0,row-1,-row+1], [row-1,row-1,-row+1], [row-1,0,-row+1]])

    objpoints = []
    imgpoints = []

    try:
        print ("camera ON")
        cap = cv2.VideoCapture(0)
    except:
        print ("camera failed")

    while True:
        ret, frame = cap.read()
        if not ret:
            print ("video reading error")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (row,col), None)

        if ret == True:
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), termination)
            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, mtx, dist)
            print tvecs
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
            frame = drawCube(frame, corners, imgpts)

        cv2.imshow('AR', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

row = 13
col = 9
poseEstimation(row, col)