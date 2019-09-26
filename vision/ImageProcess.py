import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import threading

class ImageConverter(threading.Thread):
    def __init__(self):
        # data members
        self.__bridge = CvBridge()
        self.__image_cv_left = []
        self.__image_cv_right = []

        # threading
        threading.Thread.__init__(self)
        self.__interval_ms = 10
        self.__stop_flag = False

        # subscriber
        self.__sub_list = [rospy.Subscriber('/endoscope/left/image_raw', Image, self.__image_raw_left_cb),
                           rospy.Subscriber('/endoscope/right/image_raw', Image, self.__image_raw_right_cb)]

        # create node
        if not rospy.get_node_uri():
            rospy.init_node('ImageConverter_node', anonymous=True, log_level=rospy.WARN)
            self.rate = rospy.Rate(1000.0 / self.__interval_ms)
        else:
            rospy.logdebug(rospy.get_caller_id() + ' -> ROS already initialized')

        self.start()
        rospy.spin()

    def start(self):
        self.__stop_flag = False
        self.__thread = threading.Thread(target=self.run, args=(lambda: self.__stop_flag,))
        self.__thread.daemon = True
        self.__thread.start()

    def stop(self):
        self.__stop_flag = True

    def run(self, stop):
        while True:
            # To do
            cv2.imshow("Image_raw_left", self.__image_cv_left)
            cv2.imshow("Image_raw_right", self.__image_cv_right)
            self.rate.sleep()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                stop()
                break

    def __image_raw_left_cb(self, data):
        try:
            self.__image_cv_left = self.__bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def __image_raw_right_cb(self, data):
        try:
            self.__image_cv_right = self.__bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)


if __name__ == '__main__':
    ic = ImageConverter()