# import zmq
# import time
# from utils_2.comunication import receive
# detector_context = zmq.Context()
# detector_socket = detector_context.socket(zmq.SUB)
# detector_socket.bind(f"tcp://192.168.1.106:7001")
# detector_socket.subscribe("")
# while True:
#     msg = receive(detector_socket)
#     print(msg)
#     time.sleep(3)
