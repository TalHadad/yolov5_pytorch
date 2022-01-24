# client_handler.py
import socket
import pickle
from server_handler import MessageType, HEADERSIZE

class Client():

    def __init__(self, ip, port: int, controller):
        self.connect_socket(ip, port)
        self.controller = controller

    def process_message(self, msg):
        if msg == MessageType.WELCOME:
            print(msg)
        elif msg == MessageType.CLOSE:
            print(msg)
            self.controller.clean_exit()
        else:
            previous_x, previous_y, current_x, current_y = msg
            self.controller.move_car(previous_x, previous_y, current_x, current_y)
            #try:
            #    self.controller.move_car(previous_x, previous_y, current_x, current_y)
            #except:
            #    self.controller.clean_exit()

    def connect_socket(self, ip:str, port:int ):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.connect((ip, port))
            print (f'connection client to {ip}:{port}')
        except socket.error as err:
            print ("socket creation failed with error %s" %(err))

    def start(self):
        new_msg = True
        full_msg = b''
        while True:
            msg = self.server_socket.recv(16)
            if new_msg:
                print(f"new message length: {msg[:HEADERSIZE]}")
                msglen = int(msg[:HEADERSIZE])
                new_msg = False

            #full_msg += msg.decode("utf-8")
            full_msg += msg

            if len(full_msg)-HEADERSIZE == msglen:
                print("full msg recvd")
                #print(full_msg[HEADERSIZE:])

                message = pickle.loads(full_msg[HEADERSIZE:])
                #print(f'{message}')
                #print(message)
                self.process_message(message)

                image = self.controller.take_a_picture()
                msg = pickle.dumps(image)
                msg = bytes(f'{len(msg):<{HEADERSIZE}}', "utf-8") + msg
                self.server_socket.send(msg)

                new_msg = True
                full_msg = b''



if __name__ == '__main__':
    ip = '192.168.1.106'
    port = 8003
    client = Client(ip, port, controller=Controller())
    client.start()