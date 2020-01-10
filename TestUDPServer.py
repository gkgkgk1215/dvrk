# UDP multicast examples, Hugo Vincent, 2005-05-14.
import socket
import struct
import numpy as np
import FLSpegtransfer.utils.CmnUtil as U

def server():
    print ()
    UDP_IP = "127.0.0.1"
    UDP_PORT_SERV = 1215
    UDP_PORT_CLNT = 1216

    sock = socket.socket(socket.AF_INET, # Internet
                         socket.SOCK_DGRAM) # UDP
    sock.bind((UDP_IP, UDP_PORT_SERV))

    while True:
        data = struct.pack('fff', 1, 2, 3)
        sock.sendto(data, (UDP_IP, UDP_PORT_CLNT))

        data_recv, addr = sock.recvfrom(1024)  # buffer size is 1024 bytes
        result = struct.unpack('f', data_recv)
        print (result)

if __name__ == "__main__":
    server()