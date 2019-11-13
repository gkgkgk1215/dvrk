import cv2
from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import Image, CompressedImage, PointCloud2
# from sensor_msgs import point_cloud2 as pc2
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
        self.__mask_transformed = []
        self.__contour = []
        self.__grasping_points = np.array([[22, 30], [48, 30], [35, 50]])
        self.__dx = 70  # mask size
        self.__dy = 70
        self.__theta = np.r_[-60:60:4]
        self.__x = np.r_[-15:15:5]
        self.__y = np.r_[-15:15:5]

        # load mask
        filename = '../img/block_sample_drawing3.png'
        self.__mask = self.load_mask(filename, self.__dx, self.__dy)
        self.__mask_transformed = self.transform_mask(self.__mask)
        self.__contour = self.load_contour(self.__mask, 2)
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

    def load_mask(self, filename, size_x, size_y):
        img = cv2.imread(filename)
        mask = cv2.resize(img, dsize=(size_x, size_y))
        ret, mask_inv = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_not(mask_inv)
        mask = mask[:, :, 0]
        return mask

    def transform_mask(self, mask):
        transformed = np.zeros((len(self.__theta),len(self.__x),len(self.__y),np.shape(mask)[0],np.shape(mask)[1]), np.uint8)
        for n, ang in enumerate(self.__theta):
            for i, tx in enumerate(self.__x):
                for j, ty in enumerate(self.__y):
                    transformed[n][i][j] = self.img_transform(mask, ang, tx, ty)
        return transformed

    def load_contour(self, mask, linewidth):
        dx = mask.shape[0]
        dy = mask.shape[1]
        edge = cv2.Canny(mask, dx, dy)
        _, contours, _ = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_contour = np.zeros((dx, dy), np.uint8)
        cv2.drawContours(img_contour, contours, -1, (255,255,255), linewidth)
        ret, img_contour = cv2.threshold(img_contour, 10, 255, cv2.THRESH_BINARY)
        return img_contour

    def img_transform(self, img, angle_deg, tx, ty):
        """

        :param img:
        :param angle_deg: positive in counter clockwise
        :param tx: positive in rightward
        :param ty: positive in downward
        :return:
        """
        M_rot = cv2.getRotationMatrix2D((img.shape[0] / 2, img.shape[1] / 2), angle_deg, 1)
        M_tran = np.float32([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, M_rot, (img.shape[0], img.shape[1]))
        rotated = cv2.warpAffine(img, M_tran, (img.shape[0], img.shape[1]))
        return rotated

    def pnt_transform(self, pnts, size_img, angle_deg, tx, ty):
        shifted = [[p[0]-size_img[0]/2, p[1]-size_img[1]/2] for p in pnts]
        R = cv2.getRotationMatrix2D((0,0), angle_deg, 1)[:,:2]
        T = np.array([tx, ty])
        transformed = [np.array(np.matmul(R, p) + T) for p in shifted]
        new_pnts = [[p[0]+size_img[0]/2, p[1]+size_img[1]/2] for p in transformed]
        # import pdb; pdb.set_trace()
        return new_pnts

    def cnt_transform(self, cnt, angle_deg, tx, ty):
        raise NotImplementedError
        coords = np.array([[p[0][0], p[0][1]] for p in cnt])
        coords_transformed = self.pnt_transform(coords, angle_deg, tx, ty)
        coords_transformed = coords_transformed.astype(int)
        return np.reshape(coords_transformed, (coords_transformed.shape[0],1,2))

    def pegs_detection(self, masked_img, number_of_pegs):
        corners = cv2.goodFeaturesToTrack(masked_img, number_of_pegs, 0.1, 30)  # corner detection
        corners = np.int0(corners)
        corners = np.array([i.ravel() for i in corners])  # positions of pegs

        args = np.argwhere(masked_img > 10)
        dx = 10
        dy = 10
        peg_points = []
        for p in corners:
            args_y = np.argwhere((p[1]-dy<args[:,0]) & (args[:,0]<p[1]+dy))
            args_x = np.argwhere((p[0]-dx<args[:,1]) & (args[:,1]<p[0]+dx))
            common = np.intersect1d(args_x, args_y)
            average = np.average(args[common], axis=0)
            peg_points.append([average[1], average[0]])

        peg_points = np.array(peg_points).astype(int)
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
        p7=g3[0]; p8=g4[0]; p9=g5[0]; p10=g3[1]; p11=g4[1]; p12=g5[1]
        sorted = np.array([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12])
        return sorted

    def find_blocks(self, blocks_masked, peg_points):
        # Segmenting block around each peg
        block = np.array(
            [blocks_masked[c[1] - self.__dy / 2:c[1] + self.__dy / 2, c[0] - self.__dx / 2:c[0] + self.__dx / 2] for c
             in peg_points])
        result = [None] * np.shape(block)[0]
        for n, b in enumerate(block):
            n_max = 0
            for k, ang in enumerate(self.__theta):
                for i, tx in enumerate(self.__x):
                    for j, ty in enumerate(self.__y):
                        block_crossed = cv2.bitwise_and(b, b, mask=self.__mask_transformed[k][i][j])
                        n_cross = np.shape(np.argwhere(block_crossed == 255))[0]
                        if n_cross > n_max:
                            n_max = n_cross
                            theta_final = ang
                            x_final = tx
                            y_final = ty
            result[n] = [theta_final, x_final, y_final, n_max]
        return result

    def find_grasping_pose(self, result_global):
        pose = []
        for res in result_global:
            theta, x, y = res
            if theta > 0: grasping_angle_rotated = [-30+theta, 30+theta, -90+theta]
            else:         grasping_angle_rotated = [-30+theta, 30+theta, 90+theta]
            grasping_points_rotated = self.pnt_transform(self.__grasping_points, self.__mask.shape, theta, 0, 0)
            pose.append(np.array([[ga, gp[0]+x, gp[1]+y] for ga,gp in zip(grasping_angle_rotated, grasping_points_rotated)]))
        return pose

    def overlay_blocks(self, img, result_global):
        # Coloring yellow on blocks
        blocks_colored = self.change_color(img, (0, 255, 255))
        # Overlay contour
        for i, res in enumerate(result_global):
            theta, x, y = res
            dx = self.__contour.shape[1]
            dy = self.__contour.shape[0]
            roi = blocks_colored[y:y+dy, x:x+dx]
            transformed = self.img_transform(self.__contour, theta, 0, 0)
            transformed_inv = cv2.bitwise_not(transformed)
            bg = cv2.bitwise_and(roi, roi, mask=transformed_inv)
            transformed_colored = self.change_color(transformed, (0, 255, 0))   # green color overlayed
            dst = cv2.add(bg, transformed_colored)
            blocks_colored[y:y+dy, x:x+dx] = dst
        return blocks_colored

    def overlay_pegs(self, img, peg_points):
        # Coloring white on blocks
        pegs_colored = self.change_color(img, (255, 255, 255))
        count = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        for p in peg_points:
            cv2.circle(pegs_colored, (p[0], p[1]), 5, (0, 255, 0), 2, -1)   # green color overlayed
            text = "%d" % (count);
            cv2.putText(pegs_colored, text, (p[0]-10, p[1]-10), font, 0.5, (255, 255, 255), 1)
            count += 1
        return pegs_colored

    def overlay_grasping_pose(self, img, grasping_pose, put_text=False):
        overlayed = np.copy(img)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for gp in grasping_pose:
            gp = gp.astype(int)
            cv2.circle(overlayed, (gp[0][1], gp[0][2]), 3, (0, 0, 255), 2, -1)     # red color overlayed
            cv2.circle(overlayed, (gp[1][1], gp[1][2]), 3, (0, 0, 255), 2, -1)
            cv2.circle(overlayed, (gp[2][1], gp[2][2]), 3, (0, 0, 255), 2, -1)
            if put_text:
                text = "%d" % (gp[0][0]);
                cv2.putText(overlayed, text, (gp[0][1] - 20, gp[0][2]), font, 0.3, (255, 255, 255), 1)
                text = "%d" % (gp[1][0]);
                cv2.putText(overlayed, text, (gp[1][1] + 10, gp[1][2]), font, 0.3, (255, 255, 255), 1)
                text = "%d" % (gp[2][0]);
                cv2.putText(overlayed, text, (gp[2][1], gp[2][2] + 15), font, 0.3, (255, 255, 255), 1)
        return overlayed

    def overlay_numbering(self, img, grasping_pose):
        overlayed = np.copy(img)
        count = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        for gp in grasping_pose:
            gp = gp.astype(int)
            text = "%d" % (count); cv2.putText(overlayed, text, (gp[0][1] - 20, gp[0][2]), font, 0.3, (255, 255, 255), 1)
            count += 1
            text = "%d" % (count); cv2.putText(overlayed, text, (gp[1][1] + 10, gp[1][2]), font, 0.3, (255, 255, 255), 1)
            count += 1
            text = "%d" % (count); cv2.putText(overlayed, text, (gp[2][1], gp[2][2] + 15), font, 0.3, (255, 255, 255), 1)
            count += 1
        return overlayed

    def change_color(self, img, color):
        colored = np.copy(img)
        colored = cv2.cvtColor(colored, cv2.COLOR_GRAY2BGR)
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
                    pegs_overlayed = self.overlay_pegs(pegs_masked, peg_points)

                    # Segment images around peg_points & Find blocks
                    result = self.find_blocks(blocks_masked, peg_points)
                    result_global = [[res[0], pp[0]+res[1]-self.__mask.shape[0]/2, pp[1]+res[2]-self.__mask.shape[1]/2] for res, pp in zip(result, peg_points)]

                    # Coloring & Overlay
                    blocks_overlayed = self.overlay_blocks(blocks_masked, result_global)

                    # Find grasping pose & overlay
                    grasping_pose = self.find_grasping_pose(result_global)
                    blocks_overlayed = self.overlay_grasping_pose(blocks_overlayed, grasping_pose, True)
                    # blocks_overlayed = self.overlay_numbering(blocks_overlayed, grasping_pose)

                    cv2.imshow("img_color", img_color)
                    cv2.imshow("masked_pegs", pegs_overlayed)
                    cv2.imshow("masked_blocks", blocks_overlayed)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

                self.rate.sleep()
        finally:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    bdz = BlockDetectionZivid()