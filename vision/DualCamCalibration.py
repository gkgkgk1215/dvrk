import cv2
import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import threading
import numpy as np
import pyrealsense2 as rs

# Find Chessboard corners simultaneously from dual camera and calculate transformation matrix between those two.
class DualCamCalibration():
    def __init__(self, checkerboard_row, checkerboard_col, drawing_box_height, cam_type, loadfilename, savefilename):
        # data members
        self.__row = checkerboard_row
        self.__col = checkerboard_col
        self.__height = drawing_box_height
        self.__cam_type = cam_type
        self.__loadfilename = loadfilename
        self.__savefilename = savefilename
        self.__bridge = CvBridge()
        self.__img_raw_cam = [[],[]]

        # threading
        self.img_thr = threading.Thread(target=self.__img_raw_cam_thr)  # img receiving thread
        self.img_thr.daemon = True
        self.__stop_flag = False

        # initialize camera
        for type in self.__cam_type:
            self.__img_raw_cam[self.__cam_type.index(type)] = []
            if type == 'USB':
                # USB camera initialize
                try:
                    print ("camera ON")
                    self.cap = cv2.VideoCapture(0)
                except Exception as e:
                    print ("camera failed: ", e)
            elif type == 'REALSENSE':
                # Realsense configuring depth and color streams
                self.pipeline = rs.pipeline()
                self.config = rs.config()
                self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

                # Realsense streaming start
                self.pipeline.start(self.config)
            elif type == 'ROS_TOPIC':
                # create ROS node
                if not rospy.get_node_uri():
                    rospy.init_node('Image_pipeline_node', anonymous=True, log_level=rospy.WARN)
                    print ("ROS node initialized\n")
                else:
                    rospy.logdebug(rospy.get_caller_id() + ' -> ROS already initialized')
                # ROS subscriber
                rospy.Subscriber('/kinect2/qhd/image_color', Image, self.__img_raw_cam_cb)

        # start threading
        self.img_thr.start()
        self.main()

    def main(self):
        print ("Main loop started\n")
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(row,col,0)
        objp = np.zeros((self.__row * self.__col, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.__row, 0:self.__col].T.reshape(-1, 2)
        axis = np.float32([[0, 0, 0], [0, self.__col - 1, 0], [self.__row - 1, self.__col - 1, 0], [self.__row - 1, 0, 0], [0, 0, -self.__height + 1],
                           [0, self.__col - 1, -self.__height + 1], [self.__row - 1, self.__col - 1, -self.__height + 1], [self.__row - 1, 0, -self.__height + 1]])

        # Arrays to store object points and image points from all the images.
        # objpoints = []  # 3d point in real world space
        # imgpoints1 = []  # 2d points in image plane.
        # imgpoints2 = []  # 2d points in image plane.
        # cnt = 0

        mtx = [[],[]]
        dist = [[],[]]
        rvecs = [[],[]]
        tvecs = [[],[]]
        for i in range(len(self.__loadfilename)):
            with np.load(self.__loadfilename[i]) as X:
                _, mtx[i], dist[i], _, _ = [X[n] for n in ('ret', 'mtx', 'dist', 'rvecs', 'tvecs')]

        try:
            while True:
                img1 = self.__img_raw_cam[0]
                img2 = self.__img_raw_cam[1]
                if img1 != [] and img2 != []:
                    key = cv2.waitKey(1) & 0xFF
                    gray1 = cv2.cvtColor(self.__img_raw_cam[0], cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(self.__img_raw_cam[1], cv2.COLOR_BGR2GRAY)

                    # Find the chess board corners
                    ret1, corners1 = cv2.findChessboardCorners(gray1, (self.__row, self.__col), None)
                    ret2, corners2 = cv2.findChessboardCorners(gray2, (self.__row, self.__col), None)

                    if ret1 == True and ret2 == True:
                        # If found, add object points, image points (after refining them)
                        corners1_ = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
                        corners2_ = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

                        _, rvecs[0], tvecs[0], _ = cv2.solvePnPRansac(objp, corners1_, mtx[0], dist[0])
                        _, rvecs[1], tvecs[1], _ = cv2.solvePnPRansac(objp, corners2_, mtx[1], dist[1])

                        tc1m = np.array(tvecs[0]) * 10  # (mm)
                        tc2m = np.array(tvecs[1]) * 10  # (mm)
                        Rc1m = cv2.Rodrigues(rvecs[0])[0]
                        Rc2m = cv2.Rodrigues(rvecs[1])[0]
                        Tc1m = np.vstack((np.hstack((Rc1m, tc1m)), [0, 0, 0, 1]))
                        Tc2m = np.vstack((np.hstack((Rc2m, tc2m)), [0, 0, 0, 1]))

                        # Transformation matrix from cam 1 to cam 2
                        Tc1c2 = np.matrix(Tc1m) * np.linalg.inv(np.matrix(Tc2m))
                        Rc1c2 = Tc1c2[0:3,0:3]
                        rvecsc1c2 = np.array(cv2.Rodrigues(Rc1c2))
                        tvecsc1c2 = Tc1c2[0:3,3]

                        # Draw cube
                        imgpts1, jac1 = cv2.projectPoints(axis, rvecs[0], tvecs[0], mtx[0], dist[0])
                        img1 = self.drawCube(img1, imgpts1)
                        imgpts2, jac2 = cv2.projectPoints(axis, rvecs[1], tvecs[1], mtx[1], dist[1])
                        img2 = self.drawCube(img2, imgpts2)

                        # Put text presenting the distance
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        pos_str1 = "%0.1f, %0.1f, %0.1f" % (tvecs[0][0], tvecs[0][1], tvecs[0][2])
                        rot_str1 = "%0.1f, %0.1f, %0.1f" % (rvecs[0][0]*180/3.14, rvecs[0][1]*180/3.14, rvecs[0][2]*180/3.14)
                        pos_str3 = "%0.1f, %0.1f, %0.1f" % (tvecsc1c2[0], tvecsc1c2[1], tvecsc1c2[2])
                        rot_str4 = "%0.1f, %0.1f, %0.1f" % (rvecsc1c2[0][0]*180/3.14, rvecsc1c2[0][1]*180/3.14, rvecsc1c2[0][2]*180/3.14)
                        cv2.putText(img1, pos_str1, (20, 50), font, 1, (0, 255, 0), 3)
                        cv2.putText(img1, rot_str1, (20, 80), font, 1, (0, 255, 0), 3)
                        cv2.putText(img1, pos_str3, (300, 430), font, 1, (0, 255, 0), 3)
                        cv2.putText(img1, rot_str4, (300, 460), font, 1, (0, 255, 0), 3)

                        pos_str2 = "%0.1f, %0.1f, %0.1f" % (tvecs[1][0], tvecs[1][1], tvecs[1][2])
                        rot_str2 = "%0.1f, %0.1f, %0.1f" % (rvecs[1][0] * 180 / 3.14, rvecs[1][1] * 180 / 3.14, rvecs[1][2] * 180 / 3.14)
                        cv2.putText(img2, pos_str2, (20, 50), font, 1, (0, 255, 0), 3)
                        cv2.putText(img2, rot_str2, (20, 80), font, 1, (0, 255, 0), 3)


                        if key == ord('\r'):  # ENTER
                            np.savez(self.__savefilename, cam_transform=Tc1c2)
                            print "camera transformation mtx has been saved"

                    if key == ord('q'):  # ESD
                        self.__stop_flag = True
                        break
                    cv2.namedWindow('Camera 1')
                    cv2.moveWindow('Camera 1', 100, 50)
                    cv2.imshow('Camera 1', img1)
                    cv2.namedWindow('Camera 2')
                    cv2.moveWindow('Camera 2', 740, 50)
                    cv2.imshow('Camera 2', img2)
        finally:
            if self.__cam_type == 'USB':
                self.cap.release()
                cv2.destroyAllWindows()
            elif self.__cam_type == 'REALSENSE':
                # Stop streaming
                self.pipeline.stop()
            # if objpoints != [] and imgpoints1 != [] and imgpoints2 != []:
            #     np.savez('Dual_cam_calib.npz', imgpoints1=imgpoints1, imgpoints2=imgpoints2)
            #     print ("Calibration data has been saved to 'Dual_cam_calib.npz'")
            # else:
            #     print "Calibration data is empty"

    def __img_raw_cam_thr(self):
        try:
            print ("Camera thread started\n")
            while True:
                for i in range(len(self.__cam_type)):
                    if self.__cam_type[i] == 'USB':
                        ret, self.__img_raw_cam[i] = self.cap.read()
                    elif self.__cam_type[i] == 'REALSENSE':
                        # Wait for a coherent pair of frames: depth and color
                        frames = self.pipeline.wait_for_frames()
                        color_frame = frames.get_color_frame()

                        # Convert images to numpy arrays
                        self.__img_raw_cam[i] = np.asanyarray(color_frame.get_data())
                if self.__stop_flag == True:
                    break
        except Exception as e:
            print e

    def __img_raw_cam_cb(self, data):
        try:
            if type(data).__name__ == 'CompressedImage':
                self.__img_raw_cam[self.__cam_type.index('ROS_TOPIC')] = self.__compressedimg2cv2(data)
            elif type(data).__name__ == 'Image':
                self.__img_raw_cam[self.__cam_type.index('ROS_TOPIC')] = self.__bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def drawCube(self, img, imgpts):
        imgpts = np.int32(imgpts).reshape(-1,2)
        cv2.drawContours(img, [imgpts[:4]], -1, (255, 0, 0), -3)

        for i,j in zip(range(4), range(4, 8)):
            cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (0, 255, 0), 2)

        cv2.drawContours(img, [imgpts[4:]], -1, (0, 255, 0), 2)
        return img

    def __compressedimg2cv2(self, comp_data):
        np_arr = np.fromstring(comp_data.data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

if __name__ == '__main__':
    dc = DualCamCalibration(checkerboard_row=13, checkerboard_col=9, drawing_box_height=6,
                            cam_type=('REALSENSE', 'ROS_TOPIC'),
                            loadfilename=('calibration_files/calib_realsense.npz', 'calibration_files/calib_kinect_qhd.npz'),
                            savefilename='calibration_files/calib_dualcam.npz')
