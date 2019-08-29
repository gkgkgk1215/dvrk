import time

import rospy
from sensor_msgs.msg import JointState

from reference.utils.pyLivePlot import pyLivePlot

counter = 0

def get_data(msg):
    global counter, y1, y2
    if counter % 10 == 0:
        stamp = msg.header.stamp
        time = stamp.secs + stamp.nsecs * 1e-9
        y1 = msg.position[5]
        y2 = msg.position[6]
    counter += 1

if __name__ == '__main__':
    couner = 0
    y1 = 0
    y2 = 0
    p = pyLivePlot(1000)
    rospy.init_node("actual_current_plot")
    rospy.Subscriber("/dvrk_python/PSM1/io/actuator_current_measured", JointState, get_data)
    while True:
        p.live_plotter(y1,y2)
        time.sleep(0.01)