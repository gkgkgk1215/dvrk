import socket
import struct

def server():
    print ()
    UDP_IP = "127.0.0.1"
    UDP_PORT_SERV = 1215
    UDP_PORT_SERV2 = 1216
    UDP_PORT_CLNT = 1217
    UDP_PORT_CLNT2 = 1218

    sock = socket.socket(socket.AF_INET, # Internet
                         socket.SOCK_DGRAM) # UDP
    sock2 = socket.socket(socket.AF_INET,  # Internet
                         socket.SOCK_DGRAM)  # UDP

    # sock.setblocking(0)  # Non-blocking mode
    # sock2.setblocking(0)  # Non-blocking mode

    sock.bind((UDP_IP, UDP_PORT_SERV))
    sock2.bind((UDP_IP, UDP_PORT_SERV2))
    import numpy as np

    while True:
        X = np.array([1,2,3,4,5,6])
        data = struct.pack('=6f', *X)
        sock.sendto(data, (UDP_IP, UDP_PORT_CLNT))
        data = struct.pack('fff', 4, 5, 6)
        sock2.sendto(data, (UDP_IP, UDP_PORT_CLNT2))

        data_recv, addr = sock.recvfrom(1024)  # buffer size is 1024 bytes
        result = struct.unpack('f', data_recv)
        print(addr, result)

        data_recv, addr = sock2.recvfrom(1024)  # buffer size is 1024 bytes
        result = struct.unpack('ff', data_recv)
        print(addr, result)

if __name__ == "__main__":
    server()