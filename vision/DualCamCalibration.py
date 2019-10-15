import cv2
import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import threading
import numpy as np
import pyrealsense2 as rs

class DualCamCalibration():
    def __init__(self, checkerboard_row, checkerboard_col, cam_type=('USB', 'ROS_TOPIC'), filename='calib.npz'):
        # data members
        self.__row = checkerboard_row
        self.__col = checkerboard_col
        self.__cam_type = cam_type
        self.__filename = filename
        self.__bridge = CvBridge()
        self.__img_raw_cam = dict()

        # threading
        self.img_thr = threading.Thread(target=self.__img_raw_cam_thr)  # img receiving thread
        self.img_thr.daemon = True
        self.__stop_flag = False

        # initialize camera
        for type in self.__cam_type:
            self.__img_raw_cam[type] = []
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
                    print ("ROS node initialized")
                else:
                    rospy.logdebug(rospy.get_caller_id() + ' -> ROS already initialized')
                # ROS subscriber
                rospy.Subscriber('/kinect2/qhd/image_color', Image, self.__img_raw_cam_cb)

        # start threading
        self.img_thr.start()
        self.main()

    def main(self):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(row,col,0)
        objp = np.zeros((self.__row * self.__col, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.__row, 0:self.__col].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        cnt = 0

        try:
            while True:
                for type in self.__cam_type:
                    if self.__img_raw_cam[type] != []:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('\r'):  # ENTER
                            gray = cv2.cvtColor(self.__img_raw_cam, cv2.COLOR_BGR2GRAY)

                            # Find the chess board corners
                            ret, corners = cv2.findChessboardCorners(gray, (self.__row, self.__col), None)
                            if ret == True:
                                # If found, add object points, image points (after refining them)
                                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                                # Draw and display the corners
                                self.__img_raw_cam = cv2.drawChessboardCorners(self.__img_raw_cam, (self.__row, self.__col),
                                                                               corners2, ret)
                                cnt += 1
                                objpoints.append(objp)
                                imgpoints.append(corners2)
                                print ("Corner captured: %d trials" % (cnt))
                            else:
                                print ("Corner not captured, try again")
                        elif key == ord('q'):  # ESD
                            self.__stop_flag = True
                            break
                        cv2.imshow(type, self.__img_raw_cam[type])
        finally:
            if self.__cam_type == 'USB':
                self.cap.release()
                cv2.destroyAllWindows()
            elif self.__cam_type == 'REALSENSE':
                # Stop streaming
                self.pipeline.stop()

            if objpoints != [] and imgpoints != []:
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
                np.savez(self.__filename, ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
                print "Calibration data has been saved to", self.__filename
                print "mtx", mtx
            else:
                print "Calibration data is empty"

    def __img_raw_cam_thr(self):
        try:
            while True:
                for type in self.__cam_type:
                    if type == 'USB':
                        ret, self.__img_raw_cam[type] = self.cap.read()
                    elif type == 'REALSENSE':
                        # Wait for a coherent pair of frames: depth and color
                        frames = self.pipeline.wait_for_frames()
                        color_frame = frames.get_color_frame()

                        # Convert images to numpy arrays
                        self.__img_raw_cam[type] = np.asanyarray(color_frame.get_data())
                if self.__stop_flag == True:
                    break
        except Exception as e:
            print e

    def __img_raw_cam_cb(self, data):
        try:
            if type(data).__name__ == 'CompressedImage':
                self.__img_raw_cam['ROS_TOPIC'] = self.__compressedimg2cv2(data)
            elif type(data).__name__ == 'Image':
                self.__img_raw_cam['ROS_TOPIC'] = self.__bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def __compressedimg2cv2(self, comp_data):
        np_arr = np.fromstring(comp_data.data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

if __name__ == '__main__':
    dc = DualCamCalibration(checkerboard_row=13, checkerboard_col=9, cam_type=('USB', 'ROS_TOPIC'), filename='calib.npz')
