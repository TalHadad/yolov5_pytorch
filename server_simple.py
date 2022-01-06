import socket

listen_s = socket.socket()

port = 8001
max_connection = 5
#ip = socket.gethostname()
ip = '192.168.1.106'

listen_s.bind((ip,port))

listen_s.listen(max_connection)
print(f'server started at {ip} on port {port}')

(client_s, addresss) = listen_s.accept()
print('new connection made')
