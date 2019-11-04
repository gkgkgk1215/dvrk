import cv2
from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import Image, CompressedImage, PointCloud2
from sensor_msgs import point_cloud2 as pc2
import ros_numpy
import numpy as np
import math
import imageio, imutils
from scipy import signal

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

        self.interval_ms = 300
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

    def img_transform(self, img, angle_deg, tx, ty):
        M_rot = cv2.getRotationMatrix2D((img.shape[0] / 2, img.shape[1] / 2), angle_deg, 1)
        M_tran = np.float32([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, M_rot, (img.shape[0], img.shape[1]))
        rotated = cv2.warpAffine(img, M_tran, (img.shape[0], img.shape[1]))
        return rotated

    def pnt_transform(self, pnts, angle_deg, tx, ty):
        R = cv2.getRotationMatrix2D((0, 0), -angle_deg, 1)[:,:2]
        T = np.array([tx, ty])
        return np.array([np.array(np.matmul(R, p) + T) for p in pnts])

    def cnt_transform(self, cnt, angle_deg, tx, ty):
        coords = np.array([[p[0][0], p[0][1]] for p in cnt])
        coords_transformed = self.pnt_transform(coords, angle_deg, tx, ty)
        coords_transformed = coords_transformed.astype(int)
        return np.reshape(coords_transformed, (coords_transformed.shape[0],1,2))\

    def overlayContour(self, img, img_template, angle_deg, pos_x, pos_y, color, thickness):
        img_transform = self.img_transform(img_template, angle_deg, 0,0)
        edge = cv2.Canny(img_transform, img_template.shape[0], img_template.shape[1])
        _, contours, _ = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours[0] += [pos_x-img_template.shape[0]/2, pos_y-img_template.shape[1]/2]
        cv2.drawContours(img, contours, -1, color, thickness)

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

                    # Depth masking: thresholding by depth to find blocks & pegs
                    img_blocks = cv2.inRange(img_depth, 0.868, 0.882)
                    pegs_masked = cv2.inRange(img_depth, 0.86, 0.87)

                    ## PEG DETECTION
                    val = 3
                    blur_peg = cv2.medianBlur(pegs_masked, val)     # blurring peg image
                    kernel = np.ones((2,2), np.uint8)
                    erosion_peg = cv2.erode(blur_peg, kernel, iterations=2)     # erosion peg image
                    corners = cv2.goodFeaturesToTrack(erosion_peg, 12, 0.1, 20)     # corner detection
                    corners = np.int0(corners)
                    pp = np.array([i.ravel() for i in corners])  # positions of pegs
                    # pp[0] = [124,65]
                    # pp[0] = 125,226
                    for i in corners:       # draw red dots above pegs
                        x,y = i.ravel()
                        cv2.circle(img_color, (x,y), 3, (255, 255, 255), -1)

                    # Segmenting each block around peg
                    dx = 70; dy = 70
                    block = [img_blocks[c[1]-dy/2:c[1]+dy/2,c[0]-dx/2:c[0]+dx/2] for c in pp]

                    # Importing a sample block
                    block_template = cv2.imread('../img/block_sample.png', cv2.IMREAD_GRAYSCALE)
                    # block_template = cv2.bitwise_not(block_template)
                    block_template = cv2.resize(block_template, dsize=(dx, dy))

                    result = [None]*np.shape(block)[0]
                    x = np.r_[-20:20:5]
                    y = np.r_[-20:20:5]
                    theta = np.r_[0:120:10]
                    for n, b in enumerate(block):
                        n_max = 0
                        for i,tx in enumerate(x):
                            for j,ty in enumerate(y):
                                for k,ang in enumerate(theta):
                                    block_transform = self.img_transform(block_template,ang,tx,ty)
                                    block_crossed = cv2.bitwise_and(b, b, mask=block_transform)
                                    n_cross = np.shape(np.argwhere(block_crossed == 255))[0]
                                    if n_cross > n_max:
                                        n_max = n_cross
                                        theta_final = ang
                                        x_final = tx
                                        y_final = ty

                        result[n] = [theta_final, x_final, y_final, n_max]

                    print result
                    block_final = self.img_transform(block_template, theta_final, x_final, y_final)

                    # Overlay contour
                    for i,res in enumerate(result):
                        theta, x, y, _ = res
                        img_transform = self.img_transform(block_template, theta, 0, 0)
                        edge = cv2.Canny(img_transform, block_template.shape[0], block_template.shape[1])
                        _, contours, _ = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        contours[0] += [pp[i][0]+x-block_template.shape[0]/2, pp[i][1]+y-block_template.shape[1]/2]
                        cv2.drawContours(img_color, contours, -1, (0,255,0), 2)

                    # self.overlayContour(img_color, block_template, theta_final, pp[0][0]+x_final, pp[0][1]+y_final, (0,255,0), 2)
                    cv2.imshow("img_color", img_color)
                    cv2.imshow("img_blocks", img_blocks)
                    cv2.imshow("block_segmented", block[0])
                    cv2.imshow("block_template", block_template)
                    cv2.imshow("block_final", block_final)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

                self.rate.sleep()
        finally:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    bdz = BlockDetectionZivid()
