import socket
import RPi.GPIO as GPIO
import time

s = socket.socket()

ip = ''
port = 8001
max_connections = 5

s.connect((ip, port))

GPIO.setmode(GPIO.BOARD)
GPIO.setup(7, GPIO.OUT)

while True:
      x = input("Enter message: ")
      s.send(x.encode())

      GPIO.output(7, True)
      time.sleep(5)
      GPIO.output(7, False)