import cv2
from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import Image, CompressedImage, PointCloud2
from sensor_msgs import point_cloud2 as pc2
import ros_numpy
import numpy as np

class BlockDetectionZivid():
    def __init__(self):
        # data members
        self.__bridge = CvBridge()
        self.__img_color = []
        self.__img_depth = []
        self.__points_list = []
        self.__points_ros_msg = PointCloud2()

        # load calibration data
        loadfilename = ('calibration_files/calib_zivid.npz')
        with np.load(loadfilename) as X:
            _, self.__mtx, self.__dist, _, _ = [X[n] for n in ('ret', 'mtx', 'dist', 'rvecs', 'tvecs')]

        # ROS subscriber
        rospy.Subscriber('/zivid_camera/color/image_color/compressed', CompressedImage, self.__img_color_cb)
        rospy.Subscriber('/zivid_camera/depth/image_raw', Image, self.__img_depth_cb)
        # rospy.Subscriber('/zivid_camera/points', PointCloud2, self.__pcl_cb)  # not used in this time

        # create ROS node
        if not rospy.get_node_uri():
            rospy.init_node('Image_pipeline_node', anonymous=True, log_level=rospy.WARN)
            print ("ROS node initialized")
        else:
            rospy.logdebug(rospy.get_caller_id() + ' -> ROS already initialized')

        self.interval_ms = 30
        self.rate = rospy.Rate(1000.0 / self.interval_ms)
        self.main()

    def __img_color_cb(self, data):
        try:
            if type(data).__name__ == 'CompressedImage':
                self.__img_color = self.__compressedimg2cv2(data)
            elif type(data).__name__ == 'Image':
                self.__img_color = self.__bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def __compressedimg2cv2(self, comp_data):
        np_arr = np.fromstring(comp_data.data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def __img_depth_cb(self, data):
        try:
            if type(data).__name__ == 'CompressedImage':
                self.__img_depth = self.__compressedimg2cv2(data)
            elif type(data).__name__ == 'Image':
                self.__img_depth = self.__bridge.imgmsg_to_cv2(data, "32FC1")
        except CvBridgeError as e:
            print(e)

    def __pcl_cb(self, data):
        pc = ros_numpy.numpify(data)
        points = np.zeros((pc.shape[0], pc.shape[1], 3))
        points[:, :, 0] = pc['x']
        points[:, :, 1] = pc['y']
        points[:, :, 2] = pc['z']
        self.__points_list = points

    def main(self):
        try:
            while True:
                if self.__img_color == [] or self.__img_depth == []:
                    pass
                else:
                    # Image cropping
                    x=700; w=400
                    y=150; h=300
                    img_color = self.__img_color[y:y + h, x:x + w]
                    img_depth = self.__img_depth[y:y + h, x:x + w]

                    # Depth masking
                    depth_min = np.array([0.868])
                    depth_max = np.array([0.882])
                    mask_block = cv2.inRange(img_depth, depth_min, depth_max)
                    mask_block_morph = cv2.morphologyEx(mask_block, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))
                    mask_block_morph = cv2.morphologyEx(mask_block_morph, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8))
                    img_color = cv2.bitwise_and(img_color, img_color, mask=mask_block)

                    # Blurring
                    val = 5
                    blur = cv2.medianBlur(img_color, val)
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
                    # cv2.imshow("depth", img_depth)
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

                self.rate.sleep()
        finally:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    bdz = BlockDetectionZivid()
