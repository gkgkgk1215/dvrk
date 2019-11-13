import numpy as np

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


# a = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]]
# print a[:2]
# a = np.array(a)
# print a.shape, a.shape[:2]
# b = [1,2,3,4,5,6]
# print b[:3]


# a = np.mgrid[0:5,0:3]
# print np.shape(a)
# print a
# print a[:,3,1]


# a = np.load('vision/calib_191010.npz')
# print a['mtx']

# a = ('aa', 'bb')
# print type(a)
# for i in a:
#     print i


# a = np.matrix([[1, 2, 3, 4], [5, 6, 7, 8]])
# b = np.reshape(a, -1)
# print a
# print a.shape
# print b
# print b.shape


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


# with np.load('vision/calib_kinect.npz') as X:
#     ret, mtx, dist, _, _ = [X[i] for i in ('ret', 'mtx', 'dist', 'rvecs', 'tvecs')]
#     print ret, mtx
#
# with np.load('vision/calib_kinect_44.npz') as X:
#     ret, mtx, dist, _, _ = [X[i] for i in ('ret', 'mtx', 'dist', 'rvecs', 'tvecs')]
#     print ret, mtx


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


# a = [[],[]]
# a[0] = [1,2,3]
# a[1] = [4,5,6]


# a = [1,2,3]
# b = [4,5,6]
# c = np.vstack((a,b))
# print c
# np.savetxt('testsave', c)


#
# with np.load('vision/Dual_cam_calib.npz') as X:
#     imgpoints1, imgpoints2 = [X[i] for i in ('imgpoints1', 'imgpoints2')]
#     print imgpoints1
#     print imgpoints2


# a = np.array([[1,2,3],[4,5,6],[7,8,9]])
# t = np.array([11,12,13])[np.newaxis]
# Tmtx = np.vstack((np.hstack((a,t.T)), [0,0,0,1]))
# print Tmtx

# a = 55.5555
# b = 44.4444
# c = 33.3333
# str = "%0.1f\n%0.1f\n%0.1f\n" % (a,b,c)
# print str


# a = np.matrix([[1,2],[3,4]])
# b = np.matrix([[-1,0.5],[1,1]])
# print a*b


# Tc1c2 = np.matrix([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
# Rc1c2 = Tc1c2[0:3,0:3]
# rvecsc1c2 = cv2.Rodrigues(Rc1c2)
# tvecsc1c2 = Tc1c2[0:3,3]
# print tvecsc1c2


# import cv2

# mtx = [[], []]
# dist = [[], []]
# rvecs = [[], []]
# tvecs = [[], []]
# loadfilename = ('vision/calibration_files/calib_realsense.npz', 'vision/calibration_files/calib_kinect_qhd.npz', 'vision/calibration_files/calib_dualcam.npz')
# for i in range(len(loadfilename)):
#     with np.load(loadfilename[i]) as X:
#         _, mtx[i], dist[i], _, _ = [X[n] for n in ('ret', 'mtx', 'dist', 'rvecs', 'tvecs')]
#
# print mtx[0]
# fc1x = mtx[0][0][0]
# fc1y = mtx[0][1][1]
# cx1 = mtx[0][0][2]
# cy1 = mtx[0][1][2]
# print fc1x, fc1y, cx1, cy1


# loadfilename = ('vision/calibration_files/calib_zivid.npz')
# with np.load(loadfilename) as X:
#     _, mtx, dist, _, _ = [X[n] for n in ('ret', 'mtx', 'dist', 'rvecs', 'tvecs')]
#
# print mtx
# print dist

# D: [-0.2826650142669678, 0.42553916573524475, -0.0005135679966770113, -0.000839113024994731, -0.5215581655502319]
# K: [2776.604248046875, 0.0, 952.436279296875, 0.0, 2776.226318359375, 597.9248046875, 0.0, 0.0, 1.0]
# R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
# P: [2776.604248046875, 0.0, 952.436279296875, 0.0, 0.0, 2776.226318359375, 597.9248046875, 0.0, 0.0, 0.0, 1.0, 0.0]
#
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# np.savez(self.__filename, ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

# ret = []
# rvecs = []
# tvecs = []
# mtx = np.array([[2776.604248046875, 0.0, 952.436279296875], [0.0, 2776.226318359375, 597.9248046875], [0.0, 0.0, 1.0]])
# dist = np.array([-0.2826650142669678, 0.42553916573524475, -0.0005135679966770113, -0.000839113024994731, -0.5215581655502319])
# np.savez('calib_zivid.npz', ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)


# a = np.array([[5],[1],[2],[3],[10],[11]])
# b = np.array([[2],[1],[2],[3],[1],[1]])
# a[b>=2] = -1
# print a

# a = [1,2,3,4,5]
# b = [1,3,5,7,9]
# c = [1,1,1,1,1]
# d = [x if c else y for c,x,y in zip(a,b,c)]
# print d


# import cv2
# img = cv2.imread('img/hand.jpg', cv2.IMREAD_GRAYSCALE)
# cv2.imshow("img", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# coordlist = np.argwhere(img<150)
# print coordlist


# a = np.array([[3,4],[1,2],[5,6]])
# b = np.zeros((a.shape[0],1))
# a = np.hstack((a,b))
# print a

# for i, name in enumerate(['body', 'foo', 'bar']):
#     print (i,name)

# from sklearn.neighbors import NearestNeighbors
# X = np.array([[0, 0], [1, 1], [2, 2], [3,3]])
# nbrs = NearestNeighbors(n_neighbors=3, algorithm='brute').fit(X)
# distances, indices = nbrs.kneighbors(X)
# print indices
# print distances

# a = [1,2]
# b = np.array([-1,-2])
# print a-b
# print np.linalg.norm(a-b)

# a = np.array([[1,2],[3,4],[5,6]])
# b = [np.sqrt(x*x+y*y) for x,y in a]
# print b

# a = np.array([1,2,3,4,5,6,7,8,9,10])
# b = np.argwhere(a>3)
# print b

# a = [-1,-1]
# print np.arctan(a[1]/a[0])*180/np.pi
# print np.cos(45*np.pi/180)

# a = [2,5]
# b = np.array([[2,3],[4,5]])
# print np.matmul(b,a)

# p1 = np.array([[1,2], [3,4]])
# p2 = [[x,y] for y,x in p1]
# print p2

# print np.random.randint(10)

# p1 = np.array([1,2])
# p2 = np.array([3,4])
# print np.linalg.norm(p1-p2)
# print np.sqrt(4+4)

# b = [2,1]
# print np.arctan(1)*180/np.pi
# print np.arctan(0.5)*180/np.pi

# import math
# print math.atan2(1,3)*180/np.pi

# print np.r_[0:120:4]

# a = np.array([[2,3,4], [5,9,10]])
# b = np.zeros_like(a)
# print a
# print b

# np.unravel_index(correlated.argmax(), nonzero.shape)

# print 10//3, 10%3

# a = np.array([[1,2,3],[4,5,6],[10,8,9]])
# print a, np.argmax(a)

# pnt = [1,2]
# a = [p for p in pnt]
# print a

# a = np.array([[1,2],[3,4],[5,6],[7,8]])
# b = np.reshape(a, (4,1,2))
# print b.shape, b

# a = np.array([[[0,1,2],[1,2,2],[4,6,8]],[[5,3,3],[8,4,4],[10,9,10]],[[9,5,5],[10,6,6],[11,11,21]]])
# print a.shape, a
# print a[:,:,0]
# print a[::2]
# print a[:,::2]


# a = np.array([[1,2,3], [4,5,6]])
# b = (a<3).astype(float)
# print b

# angle = np.r_[0:120:4][10]
# print angle

# print range(1,12)

# a = [[1,2,3,4], [5,6,7,8]]
# b = [[9,2,3,4], [5,6,7,11]]
# a.append(b)
# print np.shape(a), a[2]

# def find_N_largest_row(array, N, ref_column):
#     copied = np.copy(array)
#     result = []
#     for i in range(N):
#         extract = copied[:, ref_column]
#         max_index = extract.argmax()
#         result.append(copied[max_index])
#         copied = np.delete(copied, max_index, axis=0)
#     return result
#
# a = np.array([[1,10,3],[5,1,2],[19,20,1],[20,3,10]])
# print np.array(a)
# print np.array(find_N_largest_row(a, 3, 2))

# in_arr = np.array([ 2, 0,  1, 5, 4, 1, 9])
# out_arr = np.argsort(in_arr)
# in_arr_sorted = in_arr[out_arr]
# print in_arr
# print out_arr
# print in_arr_sorted

# a = np.array([[1,10,3],[5,1,2],[19,20,1],[20,3,10]])
# print a
# ind = np.argsort(a, axis=0)  # sorts along first axis (down)
# print ind
# print np.take_along_axis(a, ind, axis=0)  # same as np.sort(x, axis=0)

# input array
# in_arr = np.array([21, 100, 112, 51, 4, 8, 10,9,11])
# print in_arr
# out_arr = np.partition(in_arr, 5)
# print out_arr
# index_arr = np.argpartition(in_arr,5)
# print index_arr

# a = np.array([[1,10,3],[5,1,2],[19,20,1],[20,3,10],[44,45,100]])
#
# def find_N_largest(array, N):
#     array_ravel = array.ravel()
#     result = -np.partition(-array_ravel, N)[:N]
#
#     result_args = np.argpartition(-array_ravel, N)[:N]
#     result_args = np.transpose(np.unravel_index(result_args, array.shape))
#     return (result, result_args)
#
# print a
# print find_N_largest(a, 6)

# def find_N_largest_row(array, N, ref_column):
#     copied = np.copy(array)
#     result = []
#     for i in range(N):
#         extract = copied[:, ref_column]
#         max_index = extract.argmax()
#         result.append(copied[max_index])
#         copied = np.delete(copied, max_index, axis=0)
#     return result

# best_value = [0,1,2,3]
# angle = np.r_[0:120:4][best_value]
# print angle

# a = [1,2,3,4,5]
# b = [6,7,8,9,10]
# c = [i+k for i,k in zip(a,b)]

# a = 10
# b = np.r_[-a:a:2]
# print b

# a = np.array([[1,10,3],[5,1,2],[19,20,1],[20,3,10],[44,45,100]])
# args = np.argwhere(a==3)
# print np.where(a[:,0]>3, 100, 0)

# a = np.array([[1,2],[3,4],[5,6],[7,8]])
# arg_x = np.argwhere(a[:,0]>3)
# arg_y = np.argwhere(a[:,1]<7)
# common = np.intersect1d(arg_x, arg_y)
#
# print a
# print arg_x
# print arg_y
# print common
# print a[common].ravel()

# b = []
# a = np.array([[1,2],[3,4],[5,6],[7,8]])
# print np.average(a, axis=0)
# b.append(a[0])
# b.append(a[1])
# print np.array(b)

# a = np.array([[1,2],[3,4],[5,6],[7,8]])
# print a - [1,1]