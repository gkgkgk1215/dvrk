import cv2
from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import Image, CompressedImage, PointCloud2
from sensor_msgs import point_cloud2 as pc2
import ros_numpy
import numpy as np
from scipy.signal import correlate2d
import imutils

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

    def load_mask(self, filename):
        img = cv2.imread(filename)
        ret, mask_inv = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_not(mask_inv)
        mask = mask[:, :, 0]
        return mask

    def load_contour(self, filename):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        ret, mask = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
        edge = cv2.Canny(img, mask.shape[0], mask.shape[1])
        _, cnts, _ = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = np.zeros_like(img)
        cv2.drawContours(contour, cnts, -1, (255, 255, 255), 2)
        return contour

    def load_grasping_points(self, filename):
        contour = self.load_contour(filename)
        grasping_points = cv2.cvtColor(contour, cv2.COLOR_GRAY2BGR)

        # red dots on grasping point
        grasping_points[25][15] = [0, 0, 255]
        grasping_points[25][40] = [0, 0, 255]
        grasping_points[45][27] = [0, 0, 255]
        return grasping_points

    def downsample_naive(self, img, downsample_factor):
        """
        Naively downsamples image without LPF.
        """
        new_img = img.copy()
        new_img = new_img[::downsample_factor]
        new_img = new_img[:, ::downsample_factor]
        return new_img

    def locate_block(self, img, mask=None, downsample_factor=4, correlated=None):
        if len(img.shape) == 3:
            img = img[:, :, 0]
        if mask is None:
            mask = np.load("mask.npy")
        nonzero = (img > 0).astype(float)
        nonzero = self.downsample_naive(nonzero, downsample_factor)
        downsampled_mask = self.downsample_naive(mask, downsample_factor)
        if correlated is None:
            correlated = correlate2d(nonzero, downsampled_mask, mode='same')
        best = np.array(np.unravel_index(correlated.argmax(), nonzero.shape)) * downsample_factor
        best_args = np.copy(best)
        best[0] -= mask.shape[0] // 2
        best[1] -= mask.shape[1] // 2
        return (best, mask, correlated.max(), correlated)

    def get_masked_image(self, img, mask, start):
        new_img = np.zeros_like(img)
        new_img[start[0]:start[0] + mask.shape[0], start[1]:start[1] + mask.shape[1]] = mask
        return np.multiply(new_img, img)

    def zero_correlated(self, correlated, start, mask, downsample_factor=4):
        start_downsampled = start // downsample_factor
        buffer_scale = 1.0
        downsampled_mask = self.downsample_naive(mask, downsample_factor)
        correlated[start_downsampled[0]:start_downsampled[0] + int(downsampled_mask.shape[0] * buffer_scale),
        start_downsampled[1]:start_downsampled[1] + int(downsampled_mask.shape[1] * buffer_scale)] = 0
        return correlated

    def rotate_mask(self, angle):
        rotated = imutils.rotate_bound(self.__mask, angle)
        rotated[rotated > 0] = 1
        return rotated

    def find_masks(self, img, num_triangles):
        masks = [self.rotate_mask(i) for i in np.r_[0:120:4]]
        ret = []
        angles = []
        best_args = []
        for mask in masks:
            ret.append(self.locate_block(img, mask))
        best_value = np.argmax([r[2] for r in ret])
        angle = np.r_[0:120:4][best_value]
        angles.append(angle)
        best_arg = ret[best_value][0]
        best_args.append(best_arg)

        for j in range(1, num_triangles):
            corr = [self.zero_correlated(r[3], ret[best_value][0], ret[best_value][1]) for r in ret]
            ret = []
            for i, mask in enumerate(masks):
                ret.append(self.locate_block(img, mask, correlated=corr[i]))
            best_value = np.argmax([r[2] for r in ret])
            angle = np.r_[0:120:4][best_value]
            angles.append(angle)
            best_arg = ret[best_value][0]
            best_args.append(best_arg)
        return angles, best_args

    def overlay_contour(self, img, x,y,angle, color):
        rotated = imutils.rotate_bound(self.__contour, angle)
        args_local = np.argwhere(rotated)
        args = np.array([[x+p[0], y+p[1]] for p in args_local])
        for n in args:
            if n[0] < img.shape[0] and n[1] < img.shape[1]:
                img[n[0]][n[1]] = list(color)

    def change_color(self, img, color):
        colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        args = np.argwhere(colored)
        for n in args:
            colored[n[0]][n[1]] = list(color)
        return colored

    def main(self):
        try:
            filename = '../img/block_sample_drawing.png'
            self.__mask = self.load_mask(filename)
            self.__contour = self.load_contour(filename)
            self.__grasping_points = self.load_grasping_points(filename)
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
                    blocks_masked = cv2.inRange(img_depth, 0.872, 0.882)
                    pegs_masked = cv2.inRange(img_depth, 0.86, 0.87)

                    angles, args = self.find_masks(blocks_masked, 12)
                    blocks_masked_colored = self.change_color(blocks_masked, (0,255,255))   # yellow color on blocks

                    for p,theta in zip(args, angles):
                        self.overlay_contour(blocks_masked_colored, p[0], p[1], theta, (0, 255, 0))

                    cv2.imshow("original", img_color)
                    cv2.imshow("blocks", blocks_masked)
                    cv2.imshow("pegs", pegs_masked)
                    cv2.imshow("blocks_contour", blocks_masked_colored)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

                self.rate.sleep()
        finally:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    bdz = BlockDetectionZivid()
