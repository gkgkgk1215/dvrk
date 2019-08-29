import numpy as np
import rospy
from sensor_msgs.msg import JointState


class RecordingROSData():
    def __init__(self):
        self.__topic_cnt = 0
        self.__sub_list = []
        self.__dict_list = {}
        self.__data_recorded = []

        # self.__sub_list = [rospy.Subscriber(self.__full_ros_namespace + '/goal_reached',
        #                                     Bool, self.__goal_reached_cb),
        #                    rospy.Subscriber(self.__full_ros_namespace + '/position_cartesian_current',
        #                                     PoseStamped, self.__position_cartesian_current_cb),
        #                    rospy.Subscriber(self.__full_ros_namespace + '/io/joint_position',
        #                                     JointState, self.__position_joint_current_cb),
        #                    rospy.Subscriber(self.__full_ros_namespace + '/state_jaw_current',
        #                                     JointState, self.__position_jaw_current_cb)]

        # create node
        if not rospy.get_node_uri():
            rospy.init_node('Recording_ROS_Data_node', anonymous=True, log_level=rospy.WARN)
        else:
            rospy.logdebug(rospy.get_caller_id() + ' -> ROS already initialized')


    def add_topic(self, topic_name, msg_type, ):
        self.__sub_list.append(rospy.Subscriber(topic_name, msg_type, self.__callback, self.__topic_cnt))
        self.__dict_list[self.topic_cnt] = {topic_name, msg_type}
        self.topic_cnt += 1

    def __callback(self, data, arg):

        __dict_list[arg]

        // if msg_type is JointState




    def switch(self, x):


    def __actuator_current_measured_cb(data):
        global __position_joint_current
        global jaw_current_record
        __position_joint_current.resize(len(data.position))
        __position_joint_current.flat[:] = data.position
        jaw_current_record = np.vstack((jaw_current_record,[__position_joint_current[5], __position_joint_current[6]]))

if __name__ == '__main__':
    rospy.init_node('Record_ROS_Topic', anonymous=True, log_level=rospy.WARN)
    rate = rospy.Rate(30)
    rospy.Subscriber('/dvrk_python/PSM2/io/actuator_current_measured', JointState, __actuator_current_measured_cb)

    __position_joint_current = np.array(0, dtype=np.float)
    jaw_current_record = np.array([0,0], dtype=np.float)

    p = dvrkArm('/PSM2')
    pos = [0., 0., -0.11]
    rot = [0., 0., 0.]
    p.set_pose(pos, rot)
    p.set_jaw(50, 'deg')
    time.sleep(1)

    p.set_jaw(-5, 'deg')
    time.sleep(0.5)
    p.set_jaw(50, 'deg')
    time.sleep(0.5)

    print jaw_current_record
    np.savetxt('jaw_current.txt', jaw_current_record)  # X is an array