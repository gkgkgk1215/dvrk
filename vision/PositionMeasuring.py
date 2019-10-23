import cv2
import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import threading
import numpy as np
import pyrealsense2 as rs

class PositionMeasuring():
    def __init__(self, cam_type, loadfilename):
        # data members
        self.__cam_type = cam_type
        self.__loadfilename = loadfilename
        self.__bridge = CvBridge()
        self.__img_raw_cam = [[], []]

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

        mtx = [[], []]
        dist = [[], []]
        rvecs = [[], []]
        tvecs = [[], []]

        # Load camera intrinsic matrix
        for i in range(len(self.__loadfilename)-1):
            with np.load(self.__loadfilename[i]) as X:
                _, mtx[i], dist[i], _, _ = [X[n] for n in ('ret', 'mtx', 'dist', 'rvecs', 'tvecs')]

        # Load transformation matrix between two cameras
        i = len(self.__loadfilename)-1
        with np.load(self.__loadfilename[i]) as X:
            Tc1c2 = X['cam_transform']

        # Intrinsic parameters for camera 1
        fc1x = mtx[0][0][0]
        fc1y = mtx[0][1][1]
        cx1 = mtx[0][0][2]
        cy1 = mtx[0][1][2]

        # Intrinsic parameters for camera 2
        fc2x = mtx[1][0][0]
        fc2y = mtx[1][1][1]
        cx2 = mtx[1][0][2]
        cy2 = mtx[1][1][2]

        # Definition of variables to calculate
        r11=Tc1c2[0][0]; r12=Tc1c2[0][1]; r13=Tc1c2[0][2]; tx=Tc1c2[0][3]
        r21=Tc1c2[1][0]; r22=Tc1c2[1][1]; r23=Tc1c2[1][2]; ty=Tc1c2[1][3]
        r31=Tc1c2[2][0]; r32=Tc1c2[2][1]; r33=Tc1c2[2][2]; tz=Tc1c2[2][3]

        try:
            while True:
                img1 = self.__img_raw_cam[0]
                img2 = self.__img_raw_cam[1]
                if img1 != [] and img2 != []:
                    key = cv2.waitKey(1) & 0xFF
                    gray1 = cv2.cvtColor(self.__img_raw_cam[0], cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(self.__img_raw_cam[1], cv2.COLOR_BGR2GRAY)

                    # Image processing to detect an object
                    blur1 = cv2.medianBlur(img1, val)
                    gray1 = cv2.cvtColor(blur1, cv2.COLOR_BGR2GRAY)

                    if the object is found in the image:
                        # Least square approximation to get s1, s2
                        alp1 = (x1 - cx1) / fc1x
                        bet1 = (y1 - cy1) / fc1y
                        alp2 = (x2 - cx2) / fc2x
                        bet2 = (y2 - cy2) / fc2y
                        A = [[alp1, -r11*alp2-r12*bet2-r13],
                             [bet1, -r21*alp2-r22*bet2-r23],
                             [1,-r31*alp2-r32*bet2-r33]]
                        b = [[tx],[ty],[tz]]

                        # Calculate world coordinate of an object
                        xc1 = (x1 - cx1) / fc1x * s1
                        yc1 = (y1 - cy1) / fc1y * s1
                        zc1 = s1
                        xc2 = (x2 - cx2) / fc2x * s2
                        yc2 = (y2 - cy2) / fc2y * s2
                        zc2 = s2

                        # Put text presenting the distance
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        pos_str = "%0.1f, %0.1f, %0.1f" % (xc1, yc1, zc1)
                        cv2.putText(img1, pos_str, (20, 50), font, 1, (0, 255, 0), 3)

                        pos_str2 = "%0.1f, %0.1f, %0.1f" % (xc2, yc2, zc2)
                        cv2.putText(img2, pos_str2, (20, 50), font, 1, (0, 255, 0), 3)

                    if key == ord('q'):  # ESD
                        self.__stop_flag = True
                        break
                    cv2.imshow('img1', img1)
                    cv2.imshow('img2', img2)
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
        imgpts = np.int32(imgpts).reshape(-1, 2)
        cv2.drawContours(img, [imgpts[:4]], -1, (255, 0, 0), -3)

        for i, j in zip(range(4), range(4, 8)):
            cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (0, 255, 0), 2)

        cv2.drawContours(img, [imgpts[4:]], -1, (0, 255, 0), 2)
        return img

    def __compressedimg2cv2(self, comp_data):
        np_arr = np.fromstring(comp_data.data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

if __name__ == '__main__':
    pm = PositionMeasuring(cam_type=('REALSENSE', 'ROS_TOPIC'), loadfilename=('calib_realsense.npz', 'calib_kinect_qhd.npz', 'calib_dualcam.npz'))
