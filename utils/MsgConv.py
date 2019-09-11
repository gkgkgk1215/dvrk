import PyKDL
import numpy as np


def PyKDLFrame_to_NumpyArray(frame):
    """

    :param frame: PYKDL Frame
    :return: x,y,z position & Z,Y,X euler angle
    """
    pos = np.array([frame.p[0], frame.p[1], frame.p[2]])
    rot = frame.M.GetEulerZYX()
    return pos, rot


def NumpyArraytoPyKDLFrame(pos, rot):
    """

    :param pos: x,y,z position
    :param rot: Z,Y,X euler angle
    :return:
    """
    px, py, pz = pos
    rz, ry, rx = rot
    return PyKDL.Frame(PyKDL.Rotation.EulerZYX(rz, ry, rx), PyKDL.Vector(px, py, pz))


def NumpyArraytoPyKDLFrame_quaternion(pos, rot):
    """

    :param pos: x,y,z position
    :param rot: Z,Y,X euler angle
    :return:
    """
    px, py, pz = pos
    rx, ry, rz, rw = rot
    return PyKDL.Frame(PyKDL.Rotation.Quaternion(rx, ry, rz, rw), PyKDL.Vector(px, py, pz))