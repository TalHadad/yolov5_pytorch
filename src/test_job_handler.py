# test_job_handler.py

# How to start pytest?
# 1. def test_answer():
# 2. in test_sample.py
# 3. $ pytest
# 4. (only need assert exp, unlike unitest that you need to remember many assertTypes)

import logging
import time

from client.controller import JobHandler
from client.controller_raspi import ControllerRPi
from utils_2.config_parser import ConfigReader

def _get_controller():
    conf = ConfigReader().get_params()
    controller_raspi = ControllerRPi(conf=conf)
    return controller_raspi

def _get_action():
    action = 1
    return action

def test_job_handler():
    controller_raspi = _get_controller()
    action = _get_action()

    job_handler = JobHandler(controller_raspi, action)
    logging.info(f'job handler doning action: {action}')
    #job_handler = JobHandler(action)
    job_handler.start()
    time.sleep(5)
    job_handler.join()

def test_movement_multiprocessing():
    from multiprocessing import Process
    import os
    p = Process(target=movement, args=())
    #  class multiprocessing.Process(group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None) # making sure that the default of args is =().
    # p.run()/start()/join(timeout)/name/is_alive()/daemon/pid/terminate()/kill()/close()
    # exception multiprocessing.ProcessError (base/all errors)/TimeoutError (specific error)
    # run() Method representing the process's activity, should be override, default is object's __call__.
    # start() Start the process's activity. Call at most once. Invoke object's run() in separate process.
    # Can't use as task agent? Can't call start() then close() multipule times?
    # Solution is to create process object for each call?
    logging.info(f'multiprocess starting movement program.')
    _log_pid()
    p.start()
    p.join()

def _log_pid():
    logging.info(f'mudule name: {__name__}')
    logging.info(f'parent process: {os.getppid()}')
    logging.info(f'process id: {os.getpid()}')

def test_movement_subprocess():
    import subprocess
    logging.info(f'subprocess running controller_raspi.py main (do_action program).')
    # subprocess.run(args, *, stdin=None, input=None, stdout=None, stderr=None, capture_output=False, shell=False, cwd=None, timeout=None, check=False, encoding=None, errors=None, text=None, env=None, universal_newlines=None, **other_popen_kwargs)
    # subprocess.call(args, *, stdin=None, stdout=None, stderr=None, shell=False, cwd=None, timeout=None, **other_popen_kwargs)
    subprocess.run(['python3 client.controller_raspi.py'])
    # Older high-level API
    # subprocess.call(...)
    # Code needing to capture stdout or stderr should use run(...) instead.
    #subprocess.call([f'python -m core.{package_name}.{service_name} -e 1'], shell=True)
    logging.info(f'subprocess stoped controller_raspi.py main (do_action program).')

def movement():
    _log_pid()
    backward_pin = 11 # white_pin
    forward_pin = 12 # purple_pin
    right_pin = 13 # green_pin
    left_pin = 15 # blue_pin

    seconds = 2
    freq = 100
    duty_cycle = 100
    print(f'Enter pin number to turn on (11/12/13/15/-1 to exit): ')
    pin = int(input())
    try:
        while pin != -1:
            GPIO.setup(pin, GPIO.OUT)
            pwm = GPIO.PWM(pin, freq)
            pwm.start(duty_cycle)
            print(f'pin {pin} is now on freq {freq} and duty cycle {duty_cycle} and for the next {seconds} seconds')
            time.sleep(seconds)
            pwm.stop()
            print(f'pin {pin} stopped.')
            print(f'Enter pin number to turn on: ')
            pin = int(input())
    except:
        print('exiting.')
        GPIO.cleanup()
    finally:
        print('exiting.')
        GPIO.cleanup()

if __name__ == "__main__":
    #test_job_handler()
    test_movment_multiprocessing()
    #test_movement_subprocess()
