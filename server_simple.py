import socket

listen_s = socket.socket()

port = 8000
max_connection = 5
ip = socket.gethostname()

listen_s.bind(('',port))

listen_s.listen(max_connection)
print(f'server started at {ip} on port {port}')

(client_s, addresss) = listen_s.accept()
print('new connection made')
