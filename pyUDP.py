# UDP multicast examples, Hugo Vincent, 2005-05-14.
import socket
import struct
import numpy as np
import FLSpegtransfer.utils.CmnUtil as U

def send():
    UDP_IP = "127.0.0.1"
    UDP_PORT = 1215
    MESSAGE = "Hello, World!"

    print "UDP target IP:", UDP_IP
    print "UDP target port:", UDP_PORT
    print "message:", MESSAGE

    pos1 = [0.03, -0.03, -0.12]
    rot1 = np.array([0, 0, 0])*np.pi/180.
    jaw1 = [0*np.pi/180.]

    pos2 = [0.03, -0.03, -0.12]
    rot2 = np.array([0, 0, 0]) * np.pi / 180.
    jaw2 = [0 * np.pi / 180.]

    q1 = U.euler_to_quaternion(rot1)
    q2 = U.euler_to_quaternion(rot2)

    data = make_datagram(pos1, q1, jaw1)

    sock = socket.socket(socket.AF_INET, # Internet
                         socket.SOCK_DGRAM) # UDP
    sock.sendto(data1, (UDP_IP, UDP_PORT))

def make_datagram(pos1=[], rot1=[], jaw1=[], pos2=[], rot2=[], jaw2=[]):
    """

    :param pos: (m)
    :param rot: (quaternion)
    :param jaw: (rad)
    :return: struct
    """
    if pos1==[]:



    return struct.pack('ffffffffffffffff',
                       pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3], jaw[0],
                       pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3], jaw[0])

def recv():
    UDP_IP = "127.0.0.1"
    UDP_PORT = 1215

    sock = socket.socket(socket.AF_INET,  # Internet
                         socket.SOCK_DGRAM)  # UDP
    sock.bind((UDP_IP, UDP_PORT))

    while True:
        data, addr = sock.recvfrom(1024)  # buffer size is 1024 bytes
        print "received message:", data

if __name__ == "__main__":
    send()