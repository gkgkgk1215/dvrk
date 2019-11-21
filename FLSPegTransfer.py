import numpy as np
import cv2
import ros_numpy
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage, PointCloud2

from dvrk.vision.BlockDetection import BlockDetection
from dvrk.vision.MappingC2R import MappingC2R
from dvrk.motion.dvrkArm import dvrkArm

class FLSPegTransfer():
    def __init__(self):
        # import other modules
        filename = '../calibration_files/mapping_table_PSM1'
        self.__mapping1 = MappingC2R(filename)
        filename = '../calibration_files/mapping_table_PSM2'
        self.__mapping2 = MappingC2R(filename)
        self.__dvrk1 = dvrkArm('/PSM1')
        self.__dvrk2 = dvrkArm('/PSM2')
        self.__block_detection = BlockDetection()

        # data members
        self.__bridge = CvBridge()
        self.__img_color = []
        self.__img_depth = []
        self.__points_list = []
        self.__points_ros_msg = PointCloud2()

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
                img_raw = self.__compressedimg2cv2(data)
            elif type(data).__name__ == 'Image':
                img_raw = self.__bridge.imgmsg_to_cv2(data, "bgr8")
            self.__img_color = self.__img_crop(img_raw)
        except CvBridgeError as e:
            print(e)

    def __compressedimg2cv2(self, comp_data):
        np_arr = np.fromstring(comp_data.data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def __img_depth_cb(self, data):
        try:
            if type(data).__name__ == 'CompressedImage':
                img_raw = self.__compressedimg2cv2(data)
            elif type(data).__name__ == 'Image':
                img_raw = self.__bridge.imgmsg_to_cv2(data, "32FC1")
            self.__img_depth = self.__img_crop(img_raw)
        except CvBridgeError as e:
            print(e)

    def __pcl_cb(self, data):
        pc = ros_numpy.numpify(data)
        points = np.zeros((pc.shape[0], pc.shape[1], 3))
        points[:, :, 0] = pc['x']
        points[:, :, 1] = pc['y']
        points[:, :, 2] = pc['z']
        self.__points_list = points

    def __img_crop(self, img):
        # Image cropping
        x = 650; w = 520
        y = 100; h = 400
        cropped = img[y:y + h, x:x + w]
        return cropped

    def main(self):
        try:
            while True:
                if self.__img_color == [] or self.__img_depth == []:
                    pass
                else:
                    peg_points, final_grasping_pose, pegs_overlayed, blocks_overlayed = self.__block_detection.FLSPerception(self.__img_depth)

                    # for gp in final_grasping_pose:
                    #     picking_point = gp[1:]
                    #     pts_robot = self.__mapping.transform_pixel2robot(picking_point)

                    pts_robot1 = np.array([self.__mapping1.transform_pixel2robot(gp[1:]) for gp in final_grasping_pose])
                    pts_robot2 = np.array([self.__mapping2.transform_pixel2robot(gp[1:]) for gp in final_grasping_pose])
                    print pts_robot1, pts_robot2

                    # pos_des = [0.0, 0.0, -0.14]  # Position (m)
                    rot_des = [0, 0, 0]  # Euler angle ZYX (or roll-pitch-yaw)
                    # jaw_des = [0]
                    # p.set_pose(pos_des, rot_des, 'deg')
                    for pr in pts_robot1:
                        pos_des1 = pr
                        self.__dvrk1.set_pose(pos_des1, rot_des, 'deg')

                    for pr in pts_robot2:
                        pos_des2 = pr
                        self.__dvrk2.set_pose(pos_des2, rot_des, 'deg')

                    cv2.imshow("img_color", self.__img_color)
                    cv2.imshow("masked_pegs", pegs_overlayed)
                    cv2.imshow("masked_blocks", blocks_overlayed)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

                self.rate.sleep()
        finally:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    FLSPegTransfer()