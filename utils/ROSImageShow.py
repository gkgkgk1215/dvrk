import cv2
from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import Image, CompressedImage
import numpy as np

class ROSImageShow():
    def __init__(self, topic):
        # data members
        self.__bridge = CvBridge()
        self.__img_color = []
        self.topic = topic

        # ROS subscriber
        rospy.Subscriber(topic, CompressedImage, self.__img_color_cb)

        # create ROS node
        if not rospy.get_node_uri():
            rospy.init_node('ROS_Image_Show', anonymous=True, log_level=rospy.WARN)
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

    def __compressedimg2cv2(self, comp_data):
        np_arr = np.fromstring(comp_data.data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def main(self):
        while True:
            if self.__img_color == []:
                pass
            else:
                cv2.imshow("original", self.__img_color)
                cv2.waitKey(1)

if __name__ == '__main__':
    ros_image = ROSImageShow(topic="/endoscope/left/image_color/compressed")