import cv2
import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import threading
import numpy as np
import pyrealsense2 as rs

class DualCamCalibration():
    def __init__(self, cam_type=('REALSENSE','KINECT')):

        # data members
        self.__cam_type = cam_type
        self.__bridge = CvBridge()

        self.__img_raw_cam1 = []
        self.__img_raw_cam2 = []

        # threading
        t1 = threading.Thread(target=self.run, args=(lambda: self.__stop_flag,))  # mainthread
        t2 = threading.Thread(target=self.__img_raw_cam1_th)    # img receiving thread
        t1.daemon = True
        t2.daemon = True
        self.__interval_ms = 10
        self.__stop_flag = False

        for type in self.__cam_type:
            if type == 'REALSENSE':
                # Realsense configuring depth and color streams
                self.pipeline = rs.pipeline()
                self.config = rs.config()
                self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

                # Realsense streaming start
                self.pipeline.start(self.config)
            elif type == 'USB':
                # Laptop camera initialize
                try:
                    print ("camera ON")
                    self.cap = cv2.VideoCapture(0)
                except Exception as e:
                    print ("camera failed: ", e)
            elif type == 'KINECT':
                # create ROS node
                if not rospy.get_node_uri():
                    rospy.init_node('Image_pipeline_node', anonymous=True, log_level=rospy.WARN)
                    self.rate = rospy.Rate(1000.0 / self.__interval_ms)
                    print ("ROS node initialized")
                else:
                    rospy.logdebug(rospy.get_caller_id() + ' -> ROS already initialized')
                # ROS subscriber
                self.__sub = rospy.Subscriber('/kinect2/hd/image_color/compressed', CompressedImage,
                                              self.__img_raw_cam2_cb)

        # start threading
        t1.start()
        t2.start()
        rospy.spin()

    def run(self, stop):
        print ("Main thread started")
        while True:
            if self.__img_raw_cam1==[] or self.__img_raw_cam2==[]:
                pass
            else:
                # Resizing images
                # img1 = cv2.resize(self.__img_raw_cam1, (640, 480))  # MainCam (Realsense)
                # img2 = cv2.resize(self.__img_raw_cam2, (640, 480))  # AuxCam (Kinect2)

                # Stack both images horizontally
                # images = np.hstack((img1, img2))

                cv2.imshow("Main cam", self.__img_raw_cam1)
                cv2.imshow("Aux cam", self.__img_raw_cam2)

                self.rate.sleep()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

    def __img_raw_cam1_th(self):
        try:
            while True:
                for type in self.__cam_type:
                    if type == 'REALSENSE':
                        # Wait for a coherent pair of frames: depth and color
                        frames = self.pipeline.wait_for_frames()
                        color_frame = frames.get_color_frame()

                        # Convert images to numpy arrays
                        self.__img_raw_cam1 = np.asanyarray(color_frame.get_data())
                    elif type == 'USB':
                        ret, self.__img_raw_cam1 = self.cap.read()
        finally:
            for type in self.__cam_type:
                if type == 'REALSENSE':
                    # Stop streaming
                    self.pipeline.stop()
                elif type == 'USB':
                    self.cap.release()

    def __img_raw_cam2_cb(self, data):
        try:
            if type(data).__name__ == 'CompressedImage':
                self.__img_raw_cam2 = self.__compressedimg2cv2(data)
            elif type(data).__name__ == 'Image':
                self.__img_raw_cam2 = self.__bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def __compressedimg2cv2(self, comp_data):
        np_arr = np.fromstring(comp_data.data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

if __name__ == '__main__':
    dc = DualCamCalibration(cam_type=('USB','KINECT'))   # Choose either ('REALSENSE','KINECT') or ('USB','KINECT')