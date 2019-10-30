import cv2
from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import Image, CompressedImage, PointCloud2
from sensor_msgs import point_cloud2 as pc2
import ros_numpy
import numpy as np
import math

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
                    depth_min_block = np.array([0.868])
                    depth_max_block = np.array([0.882])
                    mask_block = cv2.inRange(img_depth, depth_min_block, depth_max_block)
                    depth_min_peg = np.array([0.86])
                    depth_max_peg = np.array([0.87])
                    mask_peg = cv2.inRange(img_depth, depth_min_peg, depth_max_peg)

                    # Blurring
                    val = 3
                    # blur = cv2.blur(mask_block, (val, val))
                    blur_block = cv2.medianBlur(mask_block, val)
                    blur_peg = cv2.medianBlur(mask_peg, val)

                    # Erosion peg image
                    kernel = np.ones((2,2), np.uint8)
                    erosion_peg = cv2.erode(blur_peg, kernel, iterations=2)

                    # Corner detection
                    corners = cv2.goodFeaturesToTrack(erosion_peg, 12, 0.1, 20)
                    corners = np.int0(corners)
                    p = np.array([i.ravel() for i in corners])
                    for i in corners:
                        x,y = i.ravel()
                        cv2.circle(img_color, (x,y), 3, (0, 0, 255), -1)

                    # RANSAC
                    # import sample block
                    block_sample = cv2.imread('../img/block_sample.png', cv2.IMREAD_GRAYSCALE)
                    block_sample = cv2.bitwise_not(block_sample)
                    dx = 60
                    dy = 60
                    block_sample = cv2.resize(block_sample, dsize=(dx, dy))

                    # segmentation
                    dx = 32; dy = 32
                    # for i in p:
                    block = [blur_block[c[1]-dy:c[1]+dy,c[0]-dx:c[0]+dx] for c in p]

                        # cv2.circle(img_color, (p[i][0], p[i][1]), 25, (0, 0, 225), 1)

                    # p2 =

                    # # circular mask
                    # radius = 26
                    # mask_circle = np.ones((dx*2,dy*2), np.uint8)*255
                    # cv2.circle(mask_circle, (dx, dy), radius, (0,0,0), -1)




                    # # block1 = cv2.bitwise_and(block1, block1, mask=mask_circle)
                    # p2 = np.argwhere(block1 == 255)  # extracting coordinate of the block1
                    # p2 = np.array([[x,y] for y,x in p2])-[dx,dy]
                    # p2 = p2.astype(float)
                    # p3 = np.argwhere(block_sample == 255)     # extracting coordinate of the sample block
                    # p3 = np.array([[x,y] for y,x in p3])-[dx,dy]
                    # p3 = p3.astype(float)

                    # n = 100 # number of repeat
                    # for i in range(1):
                    #     # randomly pick two points and calculate the inclinded angle
                    #     rand = np.random.randint(p2.shape[0], size=2)
                    #     p2_rand1 = p2[rand[0]]
                    #     p2_rand2 = p2[rand[1]]
                    #     dist_p2_rand = np.linalg.norm(p2_rand1-p2_rand2)
                    #     if dist_p2_rand > 40:
                    #         p2_rand_diff = p2_rand1 - p2_rand2
                    #         if p2_rand_diff[0] == 0:
                    #             theta = 90
                    #         else:
                    #             theta = math.atan2(-p2_rand_diff[1], p2_rand_diff[0])*180/np.pi
                    #         R = cv2.getRotationMatrix2D((0,0), theta, 1)[:,:2]
                    #         T = np.matmul(R,[25,15])-p2_rand1
                    #         print p2_rand1, p2_rand2, np.matmul(R,[25,15]), theta, T
                    #         M_rot = cv2.getRotationMatrix2D((30,30), theta, 1)
                    #         M_tran = np.float32([[1, 0, T[0]], [0, 1, T[1]]])
                    #         block_sample = cv2.warpAffine(block_sample, M_rot, (60, 60))
                    #         block_sample = cv2.warpAffine(block_sample, M_tran, (60, 60))

                    # Block sample rotation
                    # M_rot = cv2.getRotationMatrix2D((30, 30), 100, 1)
                    # tx = 10
                    # ty = 10
                    # M_tran = np.float32([[1, 0, tx], [0, 1, ty]])
                    # block_sample = cv2.warpAffine(block_sample, M_rot,(60,60))
                    # block_sample = cv2.warpAffine(block_sample, M_tran, (60, 60))


                    # dist = np.array([np.sqrt(x*x+y*y) for x,y in p2])
                    # indices = np.argwhere(dist > 20)
                    # p2[indices]



                    # Edge detection
                    # _, contours, _ = cv2.findContours(mask_block, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    # cnt = contours[0]
                    # epsilon = 0.1 * cv2.arcLength(cnt, True)
                    # epsilon = 2
                    # approx = cv2.approxPolyDP(cnt, epsilon, True)
                    # cv2.drawContours(mask_block, [approx], 0, (0, 255, 0), 3)

                    # Canny Edge detection
                    # edge = cv2.Canny(blur, 170, 200)
                    # _, contours, _ = cv2.findContours(edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    # if len(cnts) == 2:  cnts = cnts[0]
                    # elif len(cnts) == 3:    cnts = cnts[1]

                    # Contour sorting & Thresholding by area
                    # contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
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
                    cv2.imshow("blurred", blur_block)
                    cv2.imshow("block_sample", block_sample)
                    cv2.imshow("block1", block[0])
                    # cv2.imshow("mask_green", dst)
                    # cv2.imshow("mask_circle", background)
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
