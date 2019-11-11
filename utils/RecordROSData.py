import threading, time
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped
from tf_conversions import posemath

class RecordROSData(threading.Thread):
    """
    This class is for recording ROS topic data to a file.
    """
    def __init__(self, interval_ms):
        """

        :param interval_ms: sampling interval in ms
        """
        self.__topic_cnt = 0
        self.__sub_list = []
        self.__topic_list = []
        self.__msg_type_list = []
        self.__data_recorded = []
        self.__data_received = []
        self.__file_name = []

        # thread
        self.__interval_ms = interval_ms
        self.__stop_flag = False
        self.__data_receive_event = threading.Event()

        # create node
        if not rospy.get_node_uri():
            rospy.init_node('Recording_ROS_Data_node', anonymous=True, log_level=rospy.WARN)
            self.rate = rospy.Rate(1000.0 / self.__interval_ms)
            self.start()
        else:
            rospy.logdebug(rospy.get_caller_id() + ' -> ROS already initialized')

    def start(self):
        self.__stop_flag = False
        self.thread = threading.Thread(target=self.run, args=(lambda: self.__stop_flag,))
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.__stop_flag = True

    def run(self, stop):
        while True:
            self.__data_receive_event.clear()
            self.__data_receive_event.wait(20)  # 1 minute at most
            if self.__data_receive_event:
                for i in range(self.__topic_cnt):
                    if self.__data_received[i]:
                        if self.__data_recorded[i]:
                            self.__data_recorded[i].append(self.__data_received[i])
                        else:
                            self.__data_recorded[i] = [self.__data_received[i]]
            self.rate.sleep()
            if stop():
                break

    def add_topic(self, topic_name, msg_type, file_name):
        """
        This method adds topic for data recording
        :param topic_name: name of ROS topic
        :param msg_type:  type of ROS msg
        :param file_name: name of file in which ROS data is recorded
        :return:
        """
        self.__topic_list.append(topic_name)
        self.__msg_type_list.append(type(msg_type()).__name__)
        self.__data_recorded.append([])
        self.__data_received.append([])
        self.__file_name.append(file_name)
        self.__sub_list.append(rospy.Subscriber(topic_name, msg_type, self.__callback, self.__topic_cnt))
        self.__topic_cnt += 1

    def __callback(self, data, arg):
        """
        This method calls back
        :param data:
        :param arg:
        :return:
        """
        self.__data_receive_event.set()
        data_type = self.__msg_type_list[arg]
        self.__data_received[arg] = self.__switch(data_type, data)

    def __switch(self, data_type, data):
        self.data_type = "case_" + data_type
        self.case = getattr(self, self.data_type, lambda: "default")
        return self.case(data)

    def case_PoseStamped(self, data):
        frame = posemath.fromMsg(data.pose)
        pos = np.array([frame.p[0], frame.p[1], frame.p[2]])
        rot = frame.M.GetQuaternion()
        return list(pos) + list(rot)

    def case_JointState(self, data):
        joint = np.array(0, dtype=np.float)
        joint.resize(len(data.position))
        joint.flat[:] = data.position
        return list(joint)

    def write_file(self):
        for i in range(self.__topic_cnt):
            np.savetxt('{}.txt'.format(self.__file_name[i]), self.__data_recorded[i])  # X is an array

    def __actuator_current_measured_cb(data):
        global __position_joint_current
        global jaw_current_record
        __position_joint_current.resize(len(data.position))
        __position_joint_current.flat[:] = data.position
        jaw_current_record = np.vstack((jaw_current_record,[__position_joint_current[5], __position_joint_current[6]]))


if __name__ == '__main__':
    p = RecordROSData(10)
    # p.add_topic('/dvrk/PSM2/position_cartesian_current', PoseStamped, 'pose_record')
    # p.add_topic('/dvrk/PSM2/joint_states', JointState, 'joint_record')
    p.add_topic()
    p.add_topic('/dvrk/PSM1/io/actuator_current_measured', JointState, 'actual_current_peg_transfer_PSM1')
    p.add_topic('/dvrk/PSM2/io/actuator_current_measured', JointState, 'actual_current_peg_transfer_PSM2')
    rospy.spin()
    if rospy.ROSException:
        p.write_file()