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

        self.__mask = []
        self.__contour = []
        self.__grasping_points = []
        self.__dx = 70
        self.__dy = 70

        # load mask
        filename = '../img/block_sample_drawing2.png'
        self.__mask = self.load_mask(filename, self.__dx, self.__dy)
        # self.__contour = self.load_contour(filename)
        # self.__grasping_points = self.load_grasping_points(filename)

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

    def load_mask(self, filename, dx, dy):
        img = cv2.imread(filename)
        mask = cv2.resize(img, dsize=(dx, dy))
        ret, mask_inv = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_not(mask_inv)
        mask = mask[:, :, 0]
        return mask

    def overlayContour(self, img, img_template, angle_deg, pos_x, pos_y, color, thickness):
        img_transform = self.img_transform(img_template, angle_deg, 0,0)
        edge = cv2.Canny(img_transform, img_template.shape[0], img_template.shape[1])
        _, contours, _ = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours[0] += [pos_x-img_template.shape[0]/2, pos_y-img_template.shape[1]/2]
        cv2.drawContours(img, contours, -1, color, thickness)

    def pegs_detection(self, masked_img, number_of_pegs):
        val = 3
        # blur_peg = cv2.medianBlur(masked_img, val)  # blurring peg image
        # kernel = np.ones((3, 3), np.uint8)
        # erosion_peg = cv2.erode(masked_img, kernel, iterations=1)  # erosion peg image
        corners = cv2.goodFeaturesToTrack(masked_img, 12, 0.1, 20)  # corner detection
        corners = np.int0(corners)
        peg_points = np.array([i.ravel() for i in corners])  # positions of pegs
        peg_points = self.sort_position(peg_points)
        return peg_points

    def sort_position(self, points):
        arg_x = np.argsort(points[:, 0])

        g1 = points[arg_x][:3]
        g2 = points[arg_x][3:6]
        g3 = points[arg_x][6:8]
        g4 = points[arg_x][8:10]
        g5 = points[arg_x][10:12]

        g1 = g1[np.argsort(g1[:,1])]
        g2 = g2[np.argsort(g2[:,1])]
        g3 = g3[np.argsort(g3[:,1])]
        g4 = g4[np.argsort(g4[:,1])]
        g5 = g5[np.argsort(g5[:,1])]

        p1=g1[0]; p2=g2[0]; p3=g1[1]; p4=g2[1]; p5=g1[2]; p6=g2[2]
        p7=g4[0]; p8=g3[0]; p9=g5[0]; p10=g3[1]; p11=g5[1]; p12=g4[1]

        sorted = np.array([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12])
        return sorted

    def overlay_pegs(self, img, peg_points):
        for i in peg_points:  # draw red dots above pegs
            x, y = i.ravel()
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

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
        return np.reshape(coords_transformed, (coords_transformed.shape[0],1,2))

    # def find_blocks(self, img, number_of_blocks):

    def change_color(self, img, color):
        colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        args = np.argwhere(colored)
        for n in args:
            colored[n[0]][n[1]] = list(color)
        return colored

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
                    blocks_masked = cv2.inRange(img_depth, 0.868, 0.882)
                    pegs_masked = cv2.inRange(img_depth, 0.85, 0.870)

                    # Pegs detection & overlay
                    peg_points = self.pegs_detection(pegs_masked, 12)
                    self.overlay_pegs(img_color, peg_points)

                    # Segmenting block around each peg
                    block = np.array([blocks_masked[c[1]-self.__dy/2:c[1]+self.__dy/2, c[0]-self.__dx/2:c[0]+self.__dx/2] for c in peg_points])
                    result = [None] * np.shape(block)[0]
                    x = np.r_[-20:20:5]
                    y = np.r_[-20:20:5]
                    theta = np.r_[0:120:10]
                    for n, b in enumerate(block):
                        n_max = 0
                        for i, tx in enumerate(x):
                            for j, ty in enumerate(y):
                                for k, ang in enumerate(theta):
                                    block_transform = self.img_transform(self.__mask, ang, tx, ty)
                                    block_crossed = cv2.bitwise_and(b, b, mask=block_transform)
                                    n_cross = np.shape(np.argwhere(block_crossed == 255))[0]
                                    if n_cross > n_max:
                                        n_max = n_cross
                                        theta_final = ang
                                        x_final = tx
                                        y_final = ty

                        result[n] = [theta_final, x_final, y_final, n_max]

                    # print result
                    block_final = self.img_transform(self.__mask, theta_final, x_final, y_final)
                    blocks_masked_colored = self.change_color(blocks_masked, (0,255,255))

                    # Overlay contour
                    for i,res in enumerate(result):
                        theta, x, y, _ = res
                        img_transform = self.img_transform(self.__mask, theta, 0, 0)
                        edge = cv2.Canny(img_transform, self.__mask.shape[0], self.__mask.shape[1])
                        _, contours, _ = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        contours[0] += [peg_points[i][0]+x-self.__mask.shape[0]/2, peg_points[i][1]+y-self.__mask.shape[1]/2]
                        cv2.drawContours(blocks_masked_colored, contours, -1, (0,255,0), 2)

                        grasping_points = np.array([[21,27],[49,28],[37,49]])
                        grasping_points_rotated = self.pnt_transform(grasping_points, theta, x, y)
                        self.__grasping_points = np.array([peg_points[i][0]+grasping_points_rotated[:,0], peg_points[i][1]+grasping_points_rotated[:,1]]).astype(int)
                        for p in self.__grasping_points:
                            cv2.circle(blocks_masked_colored, (p[0], p[1]), 3, (0, 0, 255), -1)

                    # self.overlayContour(img_color, block_template, theta_final, pp[0][0]+x_final, pp[0][1]+y_final, (0,255,0), 2)
                    cv2.imshow("img_color", img_color)
                    cv2.imshow("masked_blocks", blocks_masked_colored)
                    cv2.imshow("masked_pegs", pegs_masked)
                    # cv2.imshow("block_template", block_template)
                    # cv2.imshow("block_final", block_final)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

                self.rate.sleep()
        finally:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    bdz = BlockDetectionZivid()
