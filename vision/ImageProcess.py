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
        self.__sub_list = [rospy.Subscriber('/kinect2/hd/image_color/compressed', CompressedImage, self.__img_raw_left_cb),
                           rospy.Subscriber('/kinect2/hd/image_color/compressed', CompressedImage, self.__img_raw_right_cb)]

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
            if self.__img_raw_left == [] or self.__img_raw_right == []:
                pass
            else:
                # Resizing images
                img1 = cv2.resize(self.__img_raw_left, (640, 360))
                img2 = cv2.resize(self.__img_raw_right, (640, 360))

                # # Color masking
                # hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
                # lower_green = np.array([30, 50, 30])
                # upper_green = np.array([90, 255, 255])
                # lower_red = np.array([-20, 50, 50])
                # upper_red = np.array([20, 255, 255])
                #
                # mask_green = cv2.inRange(hsv, lower_green, upper_green)
                # mask_red = cv2.inRange(hsv, lower_red, upper_red)
                #
                # mask_red_morph = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, np.ones((10, 10), np.uint8))
                # mask_red_morph = cv2.morphologyEx(mask_red_morph, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))
                #
                # res_green = cv2.bitwise_and(img1, img1, mask=mask_green)
                # res_red = cv2.bitwise_and(img1, img1, mask=mask_red_morph)

                # Blurring
                val = 5
                # blur1 = img1
                # blur1 = cv2.blur(img1, (val, val))
                # blur1 = cv2.GaussianBlur(img1, (val, val), 0)
                blur1 = cv2.medianBlur(img1, val)
                # blur1 = cv2.bilateralFilter(img1, 15, 100, 100)
                gray1 = cv2.cvtColor(blur1, cv2.COLOR_BGR2GRAY)

                # CLAHE
                clahe_obj1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                histeq1 = clahe_obj1.apply(gray1)
                # histeq1 = cv2.equalizeHist(gray1)

                # Canny Edge detection
                edge1 = cv2.Canny(blur1, 100, 200)
                _, contours, _ = cv2.findContours(edge1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # if len(cnts) == 2:  cnts = cnts[0]
                # elif len(cnts) == 3:    cnts = cnts[1]

                # Contour sorting & Thresholding by area
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
                # for i in range(len(contours)):
                #     if cv2.contourArea(contours[i]) < 10.0:
                #         contours = contours[:i]
                #         break

                # for i in range(len(contours)):
                #     print cv2.contourArea(contours[i])

                # epsilon = 0.008 * cv2.arcLength(contours[0], True)
                # approx = cv2.approxPolyDP(contours[0], epsilon, True)
                # hull = cv2.convexHull(contours[0])

                # cv2.drawContours(img1, [hull], -1, (0, 255, 0), 1)
                # cv2.drawContours(img1, contours[2], -1, (0, 0, 255), 3)

                cv2.imshow("original1", img1)
                cv2.imshow("blurred", blur1)
                # cv2.imshow("mask_green", res_green)
                # cv2.imshow("mask_red", res_red)
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