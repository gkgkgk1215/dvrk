import cv2
import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import threading
import numpy as np
import pyrealsense2 as rs

class CameraCalibration():
    def __init__(self, cube_row, cube_col, cube_height, cam_type='USB', filename='calib.npz'):
        # data members
        self.__row = cube_row
        self.__col = cube_col
        self.__height = cube_height
        self.__cam_type = cam_type
        self.__filename = filename
        self.__img_raw_cam = []
        self.__bridge = CvBridge()

        if self.__cam_type == 'USB':
            # USB camera initialize
            try:
                print ("camera ON")
                self.cap = cv2.VideoCapture(0)
            except Exception as e:
                print ("camera failed: ", e)
        elif self.__cam_type == 'REALSENSE':
            # Realsense configuring depth and color streams
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

            # Realsense streaming start
            self.pipeline.start(self.config)
        elif self.__cam_type == 'ROS_TOPIC':
            # ROS subscriber
            self.__sub = rospy.Subscriber('/kinect2/qhd/image_color/compressed', CompressedImage,
                                          self.__img_raw_cam_cb)
            # create ROS node
            if not rospy.get_node_uri():
                rospy.init_node('Image_pipeline_node', anonymous=True, log_level=rospy.WARN)
                print ("ROS node initialized")
            else:
                rospy.logdebug(rospy.get_caller_id() + ' -> ROS already initialized')

        self.main()

    def drawCube(self, img, imgpts):
        imgpts = np.int32(imgpts).reshape(-1,2)
        cv2.drawContours(img, [imgpts[:4]], -1, (255, 0, 0), -3)

        for i,j in zip(range(4), range(4, 8)):
            cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (0, 255, 0), 2)

        cv2.drawContours(img, [imgpts[4:]], -1, (0, 255, 0), 2)
        return img

    def main(self):
        with np.load(self.__filename) as X:
            ret, mtx, dist, _, _ = [X[i] for i in ('ret', 'mtx', 'dist', 'rvecs', 'tvecs')]

        termination = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        row = self.__row
        col = self.__col
        height = self.__height
        objp = np.zeros((row * col, 3), np.float32)
        objp[:, :2] = np.mgrid[0:row, 0:col].T.reshape(-1, 2)
        axis = np.float32([[0, 0, 0], [0, col - 1, 0], [row - 1, col - 1, 0], [row - 1, 0, 0], [0, 0, -height + 1],
                           [0, col - 1, -height + 1], [row - 1, col - 1, -height + 1], [row - 1, 0, -height + 1]])

        try:
            while True:
                if self.__cam_type == 'USB':
                    ret, self.__img_raw_cam = self.cap.read()
                    if not ret:
                        print ("video reading error")
                        break
                elif self.__cam_type == 'REALSENSE':
                    # Wait for a coherent pair of frames: depth and color
                    frames = self.pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()

                    # Convert images to numpy arrays
                    self.__img_raw_cam = np.asanyarray(color_frame.get_data())

                if self.__img_raw_cam != []:
                    img = self.__img_raw_cam
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    # Find the chess board corners
                    ret, corners = cv2.findChessboardCorners(gray, (self.__row, self.__col), None)
                    if ret == True:
                        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), termination)
                        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, mtx, dist)
                        # print rvecs[0], rvecs[1], rvecs[2]

                        # Rc1m = np.array(cv2.Rodrigues(rvecs)[0])    # 3x3 rotation matrix
                        # tc1m = np.array(tvecs)  # translational vector
                        # Tc1m = np.vstack((np.hstack((Rc1m, tc1m)), [0, 0, 0, 1])) # 4x4 homogeneous transformation matrix
                        # print Tc1m

                        # Put text presenting the distance
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        pos_str = "%0.1f, %0.1f, %0.1f" % (tvecs[0], tvecs[1], tvecs[2])
                        rot_str = "%0.1f, %0.1f, %0.1f" % (rvecs[0]*180/3.14, rvecs[1]*180/3.14, rvecs[2]*180/3.14)
                        cv2.putText(img, pos_str, (20,50), font, 1, (0, 255, 0), 3)
                        cv2.putText(img, rot_str, (20,80), font, 1, (0, 255, 0), 3)
                        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
                        img = self.drawCube(img, imgpts)

                    cv2.imshow("AR", img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):  # ENTER
                        break

        finally:
            if self.__cam_type == 'USB':
                self.cap.release()
                cv2.destroyAllWindows()
            elif self.__cam_type == 'REALSENSE':
                # Stop streaming
                self.pipeline.stop()

    def __img_raw_cam_cb(self, data):
        try:
            if type(data).__name__ == 'CompressedImage':
                self.__img_raw_cam = self.__compressedimg2cv2(data)
            elif type(data).__name__ == 'Image':
                self.__img_raw_cam = self.__bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def __compressedimg2cv2(self, comp_data):
        np_arr = np.fromstring(comp_data.data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

if __name__ == '__main__':
    cc = CameraCalibration(cube_row=13, cube_col=9, cube_height=6, cam_type='ROS_TOPIC', filename='calib_kinect_qhd.npz')
