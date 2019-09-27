import cv2
import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import threading
import numpy as np

class ImageConverter(threading.Thread):
    def __init__(self):
        # data members
        self.__bridge = CvBridge()
        self.__img_raw_left = []
        self.__img_raw_right = []

        # threading
        threading.Thread.__init__(self)
        self.__interval_ms = 10
        self.__stop_flag = False

        # subscriber
        self.__sub_list = [rospy.Subscriber('/endoscope/left/image_raw/compressed', CompressedImage, self.__img_raw_left_cb),
                           rospy.Subscriber('/endoscope/right/image_raw/compressed', CompressedImage, self.__img_raw_right_cb)]

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
            # Resizing images
            img1 = cv2.resize(self.__img_raw_left, (640, 360))
            img2 = cv2.resize(self.__img_raw_right, (640, 360))

            # Blurring
            val = 5
            # blur1 = img1
            # blur1 = cv2.blur(img1, (val, val))
            # blur1 = cv2.GaussianBlur(img1, (val, val), 0)
            blur1 = cv2.medianBlur(img1, val)
            # blur1 = cv2.bilateralFilter(img1, 15, 20, 20)

            gray1 = cv2.cvtColor(blur1, cv2.COLOR_BGR2GRAY)

            # CLAHE
            clahe_obj1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            histeq1 = clahe_obj1.apply(gray1)
            # histeq1 = cv2.equalizeHist(gray1)

            # Canny Edge detection
            edge1 = cv2.Canny(histeq1, 50, 200)

            cv2.imshow("original1", img1)
            cv2.imshow("blur1", blur1)
            cv2.imshow("gray1", gray1)
            cv2.imshow("clahe1", histeq1)
            cv2.imshow("edge1", edge1)
            # cv2.imshow("Image_raw_right", img2)
            self.rate.sleep()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                stop()
                break

    def __img_raw_left_cb(self, data):
        try:
            if type(data).__name__ == 'CompressedImage':
                self.__img_raw_left = self.__compressedimg2cv2(data)
            elif type(data).__name__ == 'Image':
                self.__img_raw_left = self.__bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def __img_raw_right_cb(self, data):
        try:
            if type(data).__name__ == 'CompressedImage':
                self.__img_raw_right = self.__compressedimg2cv2(data)
            elif type(data).__name__ == 'Image':
                self.__img_raw_right = self.__bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def __compressedimg2cv2(self, comp_data):
        np_arr = np.fromstring(comp_data.data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

if __name__ == '__main__':
    ic = ImageConverter()