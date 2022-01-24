import socket
import RPi.GPIO as GPIO
import time

s = socket.socket()

ip = '192.168.1.106'
port = 8002
max_connections = 5

s.connect((ip, port))

forward_pin = 12
GPIO.setmode(GPIO.BOARD)
GPIO.setup(forward_pin, GPIO.OUT)

while True:
      x = input("Enter message: ")
      s.send(x.encode())

      GPIO.output(forward_pin, True)
      time.sleep(5)
      GPIO.output(forward_pin, False)