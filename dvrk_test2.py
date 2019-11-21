from dvrk.motion.dvrkArm import dvrkArm

psm2 = dvrkArm('/PSM2')
while True:
    pos_des = [-0.05, -0.05, -0.12]  # Position (m)
    rot_des = [0, 0, 0]  # Euler angle ZYX (or roll-pitch-yaw)
    psm2.set_pose(pos_des, rot_des, 'deg')

    pos_des = [0.05, 0.05, -0.15]  # Position (m)
    rot_des = [0, 0, 0]  # Euler angle ZYX (or roll-pitch-yaw)
    psm2.set_pose(pos_des, rot_des, 'deg')