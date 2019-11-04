import cv2
from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import Image, CompressedImage, PointCloud2
from sensor_msgs import point_cloud2 as pc2
import ros_numpy
import numpy as np
import imageio, imutils
from scipy.signal import correlate2d

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
        return np.reshape(coords_transformed, (coords_transformed.shape[0],1,2))

    def overlayContour(self, img, img_template, angle_deg, pos_x, pos_y, color, thickness):
        img_transform = self.img_transform(img_template, angle_deg, 0,0)
        edge = cv2.Canny(img_transform, img_template.shape[0], img_template.shape[1])
        _, contours, _ = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours[0] += [pos_x-img_template.shape[0]/2, pos_y-img_template.shape[1]/2]
        cv2.drawContours(img, contours, -1, color, thickness)

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

        # TODO: downsample mask too
        nonzero = (img > 0).astype(float)
        nonzero = self.downsample_naive(nonzero, downsample_factor)
        downsampled_mask = self.downsample_naive(mask, downsample_factor)
        if correlated is None:
            correlated = correlate2d(nonzero, downsampled_mask, mode='same')
        best = np.array(np.unravel_index(correlated.argmax(), nonzero.shape)) * downsample_factor
        best[0] -= mask.shape[0] // 2
        best[1] -= mask.shape[1] // 2

        new_img = np.zeros_like(img)
        new_img[best[0]:best[0] + mask.shape[0], best[1]:best[1] + mask.shape[1]] = mask
        return (best, correlated.max())

    def find_N_largest(self, array, N, reference_column):
        array_copy = array[:]
        result = []
        for i in range(N):
            max = max(array_copy[:,reference_column])
            result.append(array[max])
            array_copy.remove(array[max])
        return result

    def get_masked_image(self, img, mask, start):
        new_img = np.zeros_like(img)
        new_img[start[0]:start[0] + mask.shape[0], start[1]:start[1] + mask.shape[1]] = mask
        return np.multiply(new_img, img)

    def rotate_mask(self, angle):
        mask = np.load("mask.npy")
        rotated = imutils.rotate_bound(mask, angle)
        rotated[rotated > 0] = 1
        return rotated

    def find_masks(self, img, num_triangles):
        masks = [self.rotate_mask(i) for i in np.r_[0:120:4]]
        ret = []
        angles = []
        for mask in masks:
            ret.append(self.locate_block(img, mask))
        best_value = np.argmax([r[2] for r in ret])
        angle = np.r_[0:120:4][best_value]
        angles.append(angle)
        print angles
        # return angles,

        # mask_im = self.get_masked_image(img, ret[best_value][1], ret[best_value][0])

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

                    self.find_masks(img_blocks, 12)

                    # # Overlay contour
                    # for i,res in enumerate(result):
                    #     theta, x, y, _ = res
                    #     img_transform = self.img_transform(block_template, theta, 0, 0)
                    #     edge = cv2.Canny(img_transform, block_template.shape[0], block_template.shape[1])
                    #     _, contours, _ = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    #     contours[0] += [pp[i][0]+x-block_template.shape[0]/2, pp[i][1]+y-block_template.shape[1]/2]
                    #     cv2.drawContours(img_color, contours, -1, (0,255,0), 2)

                    # self.overlayContour(img_color, block_template, theta_final, pp[0][0]+x_final, pp[0][1]+y_final, (0,255,0), 2)
                    cv2.imshow("img_color", img_color)
                    cv2.imshow("img_blocks", img_blocks)
                    # cv2.imshow("block_segmented", block[0])
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