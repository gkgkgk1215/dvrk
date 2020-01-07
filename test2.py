# UDP multicast examples, Hugo Vincent, 2005-05-14.
import socket
import struct
from FLSpegtransfer.motion.dvrkArm import dvrkArm

def send():
    UDP_IP = "127.0.0.1"
    UDP_PORT = 1215
    MESSAGE = "Hello, World!"

    print "UDP target IP:", UDP_IP
    print "UDP target port:", UDP_PORT
    print "message:", MESSAGE

    sock = socket.socket(socket.AF_INET, # Internet
                         socket.SOCK_DGRAM) # UDP
    sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))

def recv():
    UDP_IP = "127.0.0.1"
    UDP_PORT = 1215

    sock = socket.socket(socket.AF_INET,  # Internet
                         socket.SOCK_DGRAM)  # UDP
    sock.bind((UDP_IP, UDP_PORT))
    p = dvrkArm('/PSM1')

    while True:
        data, addr = sock.recvfrom(1024)  # buffer size is 1024 bytes
        pos1, pos2, pos3, qx, qy, qz, qw, jaw = struct.unpack('ffffffff', data)
        p.set_pose(pos=[pos1, pos2, pos3], rot=[qx, qy, qz, qw])
        p.set_jaw(jaw=[jaw])
        # print "received message:", [0]

if __name__ == "__main__":
    recv()