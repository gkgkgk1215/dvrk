import sys
for p in sys.path:
    if p == '/opt/ros/kinetic/lib/python2.7/dist-packages':
        sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
# import rospy
# from sensor_msgs.msg import Image, CompressedImage
# from cv_bridge import CvBridge, CvBridgeError
import numpy as np
# import pyrealsense2 as rs

class CameraCalibration():
    def __init__(self, checkerboard_row, checkerboard_col, cam_type='USB', savefilename='calib.npz'):
        # data members
        self.__row = checkerboard_row
        self.__col = checkerboard_col
        self.__cam_type = cam_type
        self.__filename = savefilename
        # self.__bridge = CvBridge()
        self.__img_raw_cam = []

        # initialize camera
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
            rospy.Subscriber('/zivid_camera/color/image_color/compressed', CompressedImage, self.__img_raw_cam_cb)
            # create ROS node
            if not rospy.get_node_uri():
                rospy.init_node('Image_pipeline_node', anonymous=True, log_level=rospy.WARN)
                print ("ROS node initialized")
            else:
                rospy.logdebug(rospy.get_caller_id() + ' -> ROS already initialized')

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
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('\r'):  # ENTER
                        gray = cv2.cvtColor(self.__img_raw_cam, cv2.COLOR_BGR2GRAY)

                        # Find the chess board corners
                        ret, corners = cv2.findChessboardCorners(gray, (self.__col, self.__row), None)
                        if ret == True:
                            # If found, add object points, image points (after refining them)
                            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                            # Draw and display the corners
                            self.__img_raw_cam = cv2.drawChessboardCorners(self.__img_raw_cam, (self.__col, self.__row),
                                                                           corners2, ret)
                            cnt += 1
                            objpoints.append(objp)
                            imgpoints.append(corners2)
                            np.save("corners_8x6", imgpoints)
                            print (imgpoints)
                            print ("Corner captured: %d trials" % (cnt))
                        else:
                            print ("Corner not captured, try again")
                    elif key == ord('q'):  # ESD
                        break

                    cv2.imshow("Image", self.__img_raw_cam)
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
                print ("Calibration data has been saved to", self.__filename)
                print ("mtx", mtx)
            else:
                print ("Calibration data is empty")

    def __img_raw_cam_cb(self, data):
        try:
            if type(data).__name__ == 'CompressedImage':
                img_raw = self.__compressedimg2cv2(data)
            elif type(data).__name__ == 'Image':
                img_raw = self.__bridge.imgmsg_to_cv2(data, "bgr8")
            self.__img_raw_cam = self.__img_crop(img_raw)
        except CvBridgeError as e:
            print(e)

    def __img_crop(self, img):
        # Image cropping
        x = 710; w = 520
        y = 450; h = 400
        cropped = img[y:y + h, x:x + w]
        return cropped

    def __compressedimg2cv2(self, comp_data):
        np_arr = np.fromstring(comp_data.data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

if __name__ == '__main__':
    cc = CameraCalibration(checkerboard_row=3, checkerboard_col=4, cam_type='USB', savefilename='calib_laptop.npz')
