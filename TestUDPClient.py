import socket
import struct

def client():
    UDP_IP = "127.0.0.1"
    UDP_PORT_SERV = 1215
    UDP_PORT_SERV2 = 1216
    UDP_PORT_CLNT = 1217
    UDP_PORT_CLNT2 = 1218

    sock = socket.socket(socket.AF_INET,  # Internet
                         socket.SOCK_DGRAM)  # UDP
    sock2 = socket.socket(socket.AF_INET,  # Internet
                         socket.SOCK_DGRAM)  # UDP
    sock.bind((UDP_IP, UDP_PORT_CLNT))
    sock2.bind((UDP_IP, UDP_PORT_CLNT2))

    while True:
        aa = 0.13
        data_send = struct.pack('f', aa)
        sock.sendto(data_send, (UDP_IP, UDP_PORT_SERV))

        data_send2 = struct.pack('ff', aa, aa)
        sock2.sendto(data_send2, (UDP_IP, UDP_PORT_SERV2))

        data_recv, addr = sock.recvfrom(1024)  # buffer size is 1024 bytes
        pose = struct.unpack('=6f', data_recv)
        print (addr, pose)

        data_recv, addr = sock2.recvfrom(1024)  # buffer size is 1024 bytes
        pos1, pos2, pos3 = struct.unpack('fff', data_recv)
        print(addr, pos1, pos2, pos3)

if __name__ == "__main__":
    client()