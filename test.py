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

import numpy as np
a = []
a.append([])
a.append([])
print (a)
print (a[0])
# a[0] = [[0,0,0]]
# a[0].append(np.array([1,2,3]))
# a[0].append([4,5,6])
# a[1].append(np.array([1,2,3,4]))
# a[1].append([5,6,7,8])
print (np.shape(a))
if a:
    print ("yes")
else:
    print ("no")

# print a[0]
# print a[1]
# b = [[7,8,9]]
# print np.shape(a)
# print np.shape(b)
