# UDP multicast examples, Hugo Vincent, 2005-05-14.
import socket
import struct

def client():
    UDP_IP = "127.0.0.1"
    UDP_PORT_SERV = 1215
    UDP_PORT_CLNT = 1216

    sock = socket.socket(socket.AF_INET,  # Internet
                         socket.SOCK_DGRAM)  # UDP
    sock.bind((UDP_IP, UDP_PORT_CLNT))

    while True:
        aa = 0.13
        data_send = struct.pack('f', aa)
        sock.sendto(data_send, (UDP_IP, UDP_PORT_SERV))

        data_recv, addr = sock.recvfrom(1024)  # buffer size is 1024 bytes
        pos1, pos2, pos3 = struct.unpack('fff', data_recv)
        print (pos1, pos2, pos3)

if __name__ == "__main__":
    client()