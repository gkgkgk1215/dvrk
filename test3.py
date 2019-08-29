import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

import numpy as np
import rospy
from sensor_msgs.msg import JointState

def plot(msg):
    global counter
    if counter % 10 == 0:
        stamp = msg.header.stamp
        time = stamp.secs + stamp.nsecs * 1e-9
        print time, msg.position[6]
        plt.plot(time, msg.position[6], '*')
        plt.draw()
        plt.pause(0.00000000001)
    counter += 1

if __name__ == '__main__':
    counter = 0
    rospy.init_node("plotter")
    rospy.Subscriber("/dvrk_python/PSM1/io/actuator_current_measured", JointState, plot)
    plt.ion()
    # plt.show()
    # x = [1, 2, 3, 4]
    # y = [5, 6, 7, 8]
    # plt.plot(x, y)
    # plt.show()

    rospy.spin()