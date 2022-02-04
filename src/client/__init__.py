# import zmq
# from utils_2.comunication import send
# detector_context = zmq.Context()
# detector_socket = detector_context.socket(zmq.PUB)
# detector_socket.connect(f"tcp://192.168.1.106:7001")
#
# i = 0
# while True:
#     i += 1
#     send(detector_socket, i)
