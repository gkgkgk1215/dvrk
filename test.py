# import sys
# sys.path.append('/home/hwangmh/pycharmprojects/dvrk_python/')
# from dvrkArm import dvrkArm
#
# if __name__ == "__main__":
#     m = dvrkArm('/MTMR')
#     # print m.get_current_pose('deg')
#     pos_des = [0.0, -0.16, -0.129]  # Position (m)
#     rot_des = [0, 0, 0]  # Euler angle ZYX (or roll-pitch-yaw)
#     m.set_pose(pos_des, rot_des, 'deg')
#

# import numpy as np
#
# a = [1,2,3,4]
# b = np.array([k/3.0 for k in a])
# c = np.array(a) /3.0
#
# print b
# print c

# def normalize(v):
#     norm=np.linalg.norm(v, ord=2)
#     if norm==0:
#         norm=np.finfo(v.dtype).eps
#     return v/norm
#
# import numpy as np
#
# a = [1,2,3,4]
# b = [1,2,3,4]
# if a==b:
#     print "a"
# else:
#     print "b"
# print np.array(a) / np.array(b)
#
# cc = [0,0,0,0,0]
# if np.all(cc) == 0:
#     print "yes"
# else:
#     print "no"
#
# aa = [-1,-2,-3,-4]
# bb = [-1,-2,-3,-4]
# cc = np.array([-1.,-1,-1,-1])
# print cc.dtype
# print np.abs(aa)
# print np.array(bb)/np.array(aa)
# print np.linalg.norm(aa,2)
# print np.sqrt(np.sum([i*i for i in aa]))
#
# nn = normalize(aa)
# print nn, np.linalg.norm(nn)
#
# print np.array_equal(aa,bb)

# import numpy as np
#
# a = np.array([[1,2],[3,4],[5,6]])
# print a
# print np.concatenate((a,[[7,8]]), axis=0)
# print np.vstack((a,[7,8]))
# print a

# import rospy
#
# def callback(data, arg):
#     print "ok"
#
# rospy.init_node('test_node', anonymous=True, log_level=rospy.WARN)
# dict_1 = "a"
# dict_2 = "b"
# callback_lambda = lambda x: callback(x,dict_1,dict_2)
# # sub = rospy.Subscriber("text", float, callback, dict_1, dict_2)
# sub = rospy.Subscriber("text", float, callback_lambda)

# dict = {
#     1:"one",
#     2:"two",
#     3:"three",
#     4:"four",
#     5:"five"
# }
#
# print dict.get(3)
# print dict.keys()

# import rospy
# from sensor_msgs.msg import JointState
#
# a = JointState()
# print type(a).__name__

# a = {}
# a['a'] = 'b'
# print a

# class pySwitch:
#     def switch(self, arg):
#         self.case_name = "case_" + str(arg)
#         self.case = getattr(self, self.case_name, lambda:"default")
#         return self.case()
#
#     def case_0(self):
#         print "zero"
#
#     def case_1(self):
#         print "one"
#
#     def case_2(self):
#         print "two"
#
#     def case_default(self):
#         print "default"
#
# p = pySwitch()
# p.switch(2)

# import numpy as np
#
# a = np.array(0, dtype=np.float)
# a = []
# a = np.array([0,1,2])
#
# print np.size(a)

# a = {}
# a[0] = {1,'one'}
# a[1] = {2,'two'}
# a[2] = {3,'three'}
#
# ss = list(a[0])
# print ss[1]


# a = []
# a.append(19)
# print a

# import keyword
# print keyword.kwlist

# import numpy as np
# a = []
# a.append([])
# a.append([])
# print (a)
# print (a[0])
# a[0] = [[0,0,0]]
# a[0].append(np.array([1,2,3]))
# a[0].append([4,5,6])
# a[1].append(np.array([1,2,3,4]))
# a[1].append([5,6,7,8])
# print (np.shape(a))
# if a:
#     print ("yes")
# else:
#     print ("no")

# print a[0]
# print a[1]
# b = [[7,8,9]]
# print np.shape(a)
# print np.shape(b)

# import numpy as np
# print np.pi

# # import matplotlib.pyplot as plt
# # We prepare the plot
# fig = plt.figure(1)
# # We define a fake subplot that is in fact only the plot.
# ax = fig.add_subplot(111)
#
# # We change the fontsize of minor ticks label
# ax.tick_params(axis='both', which='major', labelsize=10)
# ax.tick_params(axis='both', which='minor', labelsize=8)
#
# fig.show()

# a = [10,9,8,7,6,5,4,3,2,1]
# for i in range(len(a)):
#     if a[i] < 5:
#         a = a[:i]
#         break
#
# print a

# import numpy as np
# a = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]]
# print a[:2]
# a = np.array(a)
# print a.shape, a.shape[:2]
# b = [1,2,3,4,5,6]
# print b[:3]

# import numpy as np
# a = np.mgrid[0:5,0:3]
# print np.shape(a)
# print a
# print a[:,3,1]

# import numpy as np
# a = np.load('vision/calib_191010.npz')
# print a['mtx']

# a = ('aa', 'bb')
# print type(a)
# for i in a:
#     print i

# import numpy as np
# a = np.matrix([[1, 2, 3, 4], [5, 6, 7, 8]])
# b = np.reshape(a, -1)
# print a
# print a.shape
# print b
# print b.shape

# import numpy as np
# row = 7
# col = 5
# objp = np.zeros((row*col, 3), np.float32)
# print objp
# objp[:,:2] = np.mgrid[0:row, 0:col].T.reshape(-1,2)
# print objp
# a = np.mgrid[0:7, 0:5].T.reshape(-1,2)
# print np.mgrid[0:7, 0:5].T
# print np.mgrid[0:7, 0:5].T.reshape(-1,2)
# print np.mgrid[0:6, 0:6].T.reshape(-1,3)
# axis = np.float32([[0,0,0], [0,row-1,0], [row-1,row-1,0], [row-1,0,0], [0,0,-row+1], [0,row-1,-row+1], [row-1,row-1,-row+1], [row-1,0,-row+1]])
# print axis

# import cv2

# stop_flag = False
# while not stop_flag:
#     user_input = raw_input("you want to quit?(y or n)")
#     if user_input == 'y':
#         stop_flag = True
#         print "aa"
#         break

# import numpy as np
# with np.load('vision/calib_kinect.npz') as X:
#     ret, mtx, dist, _, _ = [X[i] for i in ('ret', 'mtx', 'dist', 'rvecs', 'tvecs')]
#     print ret, mtx
#
# with np.load('vision/calib_kinect_44.npz') as X:
#     ret, mtx, dist, _, _ = [X[i] for i in ('ret', 'mtx', 'dist', 'rvecs', 'tvecs')]
#     print ret, mtx

# import numpy as np
# a = (100,50)
# print type(a)
# b = tuple([100, 50])
# print type(b)
# print b
# c = tuple(np.array([500,50]))
# print c
# print type(c)

# a = dict()
# __cam_type = ('USB', 'ROS_TOPIC')
# for type in range(len(__cam_type)):
#     print type
#
# print __cam_type.index('ROS_TOPIC')

import numpy as np
# a = [[],[]]
# a[0] = [1,2,3]
# a[1] = [4,5,6]

# import numpy as np
# a = [1,2,3]
# b = [4,5,6]
# c = np.vstack((a,b))
# print c
# np.savetxt('testsave', c)

# import numpy as np
#
# with np.load('vision/Dual_cam_calib.npz') as X:
#     imgpoints1, imgpoints2 = [X[i] for i in ('imgpoints1', 'imgpoints2')]
#     print imgpoints1
#     print imgpoints2

# import numpy as np
# a = np.array([[1,2,3],[4,5,6],[7,8,9]])
# t = np.array([11,12,13])[np.newaxis]
# Tmtx = np.vstack((np.hstack((a,t.T)), [0,0,0,1]))
# print Tmtx

a = 55.5555
b = 44.4444
c = 33.3333
str = "%0.1f\n%0.1f\n%0.1f\n" % (a,b,c)
print str
