import cv2
import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class BlockDetectionZivid():
    def __init__(self):
        # data members
        self.__bridge = CvBridge()
        self.__img_color = []
        self.__img_depth = []

        # load calibration data
        loadfilename = ('calibration_files/calib_zivid.npz')
        with np.load(loadfilename) as X:
            _, self.__mtx, self.__dist, _, _ = [X[n] for n in ('ret', 'mtx', 'dist', 'rvecs', 'tvecs')]

        # ROS subscriber
        rospy.Subscriber('/zivid_camera/color/image_color', Image, self.__img_color_cb)
        rospy.Subscriber('/zivid_camera/depth/image_raw', Image, self.__img_depth_cb)

        # create ROS node
        if not rospy.get_node_uri():
            rospy.init_node('Image_pipeline_node', anonymous=True, log_level=rospy.WARN)
            print ("ROS node initialized")
        else:
            rospy.logdebug(rospy.get_caller_id() + ' -> ROS already initialized')

        self.main()

    def __img_color_cb(self, data):
        try:
            if type(data).__name__ == 'CompressedImage':
                self.__img_color = self.__compressedimg2cv2(data)
            elif type(data).__name__ == 'Image':
                self.__img_color = self.__bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def __img_depth_cb(self, data):
        try:
            if type(data).__name__ == 'CompressedImage':
                self.__img_depth = self.__compressedimg2cv2(data)
            elif type(data).__name__ == 'Image':
                self.__img_depth = self.__bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def __compressedimg2cv2(self, comp_data):
        np_arr = np.fromstring(comp_data.data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def depth_to_3ch(self, d_img, cutoff_min, cutoff_max):
        """Process depth images the same as in the ISRR 2019 paper.
        Only applies if we're using depth images.
        EDIT: actually we're going to add a min cutoff!
        """
        w, h = d_img.shape
        n_img = np.zeros([w, h, 3])
        d_img = d_img.flatten()

        # Instead of this:
        # d_img[d_img>cutoff] = 0.0
        # Do this? The cutoff_max means beyond the cutoff, pixels become white.
        # d_img[ d_img>cutoff_max ] = 0.0
        d_img[d_img > cutoff_max] = cutoff_max
        d_img[d_img < cutoff_min] = cutoff_min
        print('max/min depth after cutoff: {:.3f} {:.3f}'.format(np.max(d_img), np.min(d_img)))

        d_img = d_img.reshape([w, h])
        for i in range(3):
            n_img[:, :, i] = d_img
        return n_img

    def main(self):
        try:
            while True:
                if self.__img_color == [] or self.__img_depth == []:
                    pass
                else:
                    img_color = self.__img_color
                    img_depth = self.__img_depth

                    # Image crop
                    x=440; w=400
                    y=270; h=300
                    img_color = img_color[y:y + h, x:x + w]
                    img_depth = img_depth[y:y + h, x:x + w]

                    print img_depth[160,160]

                    # Thresholding by height
                    cutoff_min = 0.8611 # close to the camera
                    cutoff_max = 0.872  # farther from the camera
                    img_depth[img_depth > cutoff_max] = 0
                    img_depth[img_depth < cutoff_min] = 0

                    # print img_depth

                    # Color masking
                    # hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
                    # lower_green = np.array([30, 80, 80])
                    # upper_green = np.array([90, 255, 255])
                    # lower_red = np.array([-20, 40, 40])
                    # upper_red = np.array([20, 255, 255])

                    # mask_green = cv2.inRange(hsv, lower_green, upper_green)
                    # mask_red = cv2.inRange(hsv, lower_red, upper_red)
                    # mask_red_morph = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))
                    # mask_red_morph = cv2.morphologyEx(mask_red_morph, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8))

                    # img_color = cv2.bitwise_and(img_color, img_color, mask=mask_green)
                    # img_color = cv2.bitwise_and(img_color, img_color, mask=mask_red_morph)



                    # Blurring
                    val = 5
                    # blur = img
                    # blur = cv2.blur(img, (val, val))
                    # blur = cv2.GaussianBlur(img, (val, val), 0)
                    blur = cv2.medianBlur(img_color, val)
                    # blur = cv2.bilateralFilter(img, 15, 100, 100)
                    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

                    # CLAHE
                    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    histeq = clahe_obj.apply(gray)
                    # histeq = cv2.equalizeHist(gray)

                    # Canny Edge detection
                    edge = cv2.Canny(blur, 100, 200)
                    _, contours, _ = cv2.findContours(edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

                    cv2.imshow("color", img_color)
                    cv2.imshow("depth", img_depth)
                    # cv2.imshow("blurred", blur)
                    # cv2.imshow("mask_green", res_green)
                    # cv2.imshow("mask_red", res_red)
                    # cv2.imshow("gray1", gray)
                    # cv2.imshow("clahe1", histeq)
                    # cv2.imshow("edge1", edge)
                    # cv2.imshow("Image_raw_right", img2)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

        finally:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    bdz = BlockDetectionZivid()
