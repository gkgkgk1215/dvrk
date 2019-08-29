import time

import numpy as np
import rospy
from sensor_msgs.msg import JointState

from reference.dvrk_python.dvrkArm import dvrkArm


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

    p.set_jaw(-5, 'deg')
    time.sleep(0.5)
    p.set_jaw(50, 'deg')
    time.sleep(0.5)

    p.set_jaw(-5, 'deg')
    time.sleep(0.5)
    p.set_jaw(50, 'deg')
    time.sleep(0.5)

    p.set_jaw(-5, 'deg')
    time.sleep(0.5)
    p.set_jaw(50, 'deg')
    time.sleep(0.5)

    p.set_jaw(-5, 'deg')
    time.sleep(0.5)
    p.set_jaw(50, 'deg')
    time.sleep(0.5)

    p.set_jaw(-5, 'deg')
    time.sleep(0.5)
    p.set_jaw(50, 'deg')
    time.sleep(0.5)

    p.set_jaw(-5, 'deg')
    time.sleep(0.5)
    p.set_jaw(50, 'deg')
    time.sleep(0.5)

    p.set_jaw(-5, 'deg')
    time.sleep(0.5)
    p.set_jaw(50, 'deg')
    time.sleep(0.5)

    p.set_jaw(-5, 'deg')
    time.sleep(0.5)
    p.set_jaw(50, 'deg')
    time.sleep(0.5)

    p.set_jaw(-5, 'deg')
    time.sleep(0.5)
    p.set_jaw(50, 'deg')
    time.sleep(0.5)

    p.set_jaw(-5, 'deg')
    time.sleep(0.5)
    p.set_jaw(50, 'deg')
    time.sleep(0.5)

    p.set_jaw(-5, 'deg')
    time.sleep(0.5)
    p.set_jaw(50, 'deg')
    time.sleep(0.5)

    p.set_jaw(-5, 'deg')
    time.sleep(0.5)
    p.set_jaw(50, 'deg')
    time.sleep(0.5)

    p.set_jaw(-5, 'deg')
    time.sleep(0.5)
    p.set_jaw(50, 'deg')
    time.sleep(0.5)

    p.set_jaw(-5, 'deg')
    time.sleep(0.5)
    p.set_jaw(50, 'deg')
    time.sleep(0.5)

    print jaw_current_record
    np.savetxt('jaw_current.txt', jaw_current_record)  # X is an array