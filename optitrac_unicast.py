import socket as socket
import struct as struct

NATNET_PING = 0

MAX_PACKETSIZE = 100000
SOCKET_BUFSIZE = 0x100000

client_address = '172.16.0.15'
server_address = '172.16.0.10'
command_port = 1510

def ConnectCommandSocket():
    print("Create a command socket.")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
    sock.bind((client_address, command_port))
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_BUFSIZE)
    sock.setblocking(0)
    return sock

commandSocket = ConnectCommandSocket()

## Send a ping command so that Tracker will begin streaming data
msg = struct.pack("I", NATNET_PING)
result = commandSocket.sendto(msg, (server_address, command_port))

while True:
    try:
        msg, address = commandSocket.recvfrom(MAX_PACKETSIZE + 4)
        print(msg.decode(), "\n")
    except socket.error:
        pass
