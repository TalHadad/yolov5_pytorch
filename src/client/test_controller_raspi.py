# raspberry_pi_controller.py

import RPi.GPIO as GPIO
import time
# define pins/pwms numbering scheme (physical (board) or gpio (bcm))
GPIO.setmode(GPIO.BOARD)
# (or) GPIO.setmode(GPIO.BCM)

def main():
    print('starting GPIO 11 and 13.')
    # define output pwms
    backward_pin = 11 # white_pin
    forward_pin = 12 # purple_pin
    right_pin = 13 # green_pin
    left_pin = 15 # blue_pin
    action_and_color_pwm = {backward_pin: "backward, white",
                        forward_pin: "forward, purple",
                        right_pin: "right, green",
                        left_pin: "left, blue"}
    #pwm_nums = [backward_pin, forward_pin, right_pin, left_pin]
    for num in action_and_color_pwm:
        GPIO.setup(num, GPIO.OUT)

    # send signal throght output pin
    #GPIO.output(11, True) # or 1 instead of True
    #GPIO.output(11, False) # or 0 instead of False

    # controll the signal voltage (dim)
    pwms = {}
    for num in action_and_color_pwm:
        # PWM(pin, freq) pin_num/GPIO_num, requency
        pwm = GPIO.PWM(num, 100)
        pwms[num] = pwm

    try:
        # release definition from pwms
        for pwm_num in pwms:
            # start(dutyCycle), dutyCycle = [0,100] continuous range
            pwms[pwm_num].start(19)
            #  ChangeFrequency(freq) freq = new frequency in Hertz
            #pwm.ChangeFrequency(freq)
            # ChangeDutyCycle(dutyCycle) dutyCycle = [0,100] continuous range
            #pwm.ChangeDutyCycle(100)

            # send UP signal for 60 seconds
            i=0
            seconds = 5
            while i<seconds:
                i+=1
                print(f'{action_and_color_pwm[pwm_num]} ({pwm_num}) sleep second {i}/{seconds}')
                time.sleep(1)

            pwms[pwm_num].stop()
    finally:
        print("clean exit.")
        for pwm_num in pwms:
            pwms[pwm_num].stop()

        GPIO.cleanup()

# results:
# freq = 100 (100 Hz), duty cycle = 50,19 => 1.7,0.7v
# freq = 1,000,000 (1 MHz), duty cycle = 19,7,0.01 => 1.63v
# freq = 2,000, duty cycle = 19,10 => 0.85,0.63v
# freq = 20,000, duty cycle = 10,2 => 1.3,1.22v
# conclusion: staying with freq = 100 and duty cycle = 10
def test_forward():
    forward_pin = 12 # purple_pin
    action_and_color_pwm = {forward_pin: "forward, purple"}
    for num in action_and_color_pwm:
        GPIO.setup(num, GPIO.OUT)

    pwms = {}
    for num in action_and_color_pwm:
        pwm = GPIO.PWM(num, 100)
        pwms[num] = pwm

    try:
        for pwm_num in pwms:
            pwms[pwm_num].start(100)

            i=0
            seconds = 50
            while i<seconds:
                i+=1
                print(f'{action_and_color_pwm[pwm_num]} ({pwm_num}) sleep second {i}/{seconds}')
                time.sleep(1)

            pwms[pwm_num].stop()
    finally:
        print("clean exit.")
        for pwm_num in pwms:
            pwms[pwm_num].stop()

        GPIO.cleanup()

def set_freq_and_duty_cycle():
    backward_pin = 11 # white_pin
    forward_pin = 12 # purple_pin
    right_pin = 13 # green_pin
    left_pin = 15 # blue_pin


    pin = right_pin
    seconds = 5
    freq = 100
    duty_cycle = 100
    try:
        GPIO.setup(pin, GPIO.OUT)
        pwm = GPIO.PWM(pin, freq)
        pwm.start(duty_cycle)
        print(f'pin {pin} is now on freq {freq} and duty cycle {duty_cycle} and for the next {seconds} seconds')
        time.sleep(seconds)
        msg = input().strip()
        while msg!='quit':
            freq, duty_cycle = tuple(map(int, msg.split()))
            print(f'pin {pin} is now on freq {freq} and duty cycle {duty_cycle} and for the next {seconds} seconds')
            pwm.ChangeFrequency(freq)
            pwm.ChangeDutyCycle(duty_cycle)
            time.sleep(seconds)
            msg = input().strip()

        print('exiting.')
        pwm.stop()
        GPIO.cleanup()
    except:
        print('exiting.')
        pwm.stop()
        GPIO.cleanup()

def test_movement():
    backward_pin = 11 # white_pin
    forward_pin = 12 # purple_pin
    right_pin = 13 # green_pin
    left_pin = 15 # blue_pin

    seconds = 2
    freq = 100
    duty_cycle = 100
    pin = int(input())
    try:
        while pin != -1:
            GPIO.setup(pin, GPIO.OUT)
            pwm = GPIO.PWM(pin, freq)
            pwm.start(duty_cycle)
            print(f'pin {pin} is now on freq {freq} and duty cycle {duty_cycle} and for the next {seconds} seconds')
            time.sleep(seconds)
            pwm.stop()
            GPIO.cleanup()
    except:
        print('exiting.')
        pwm.stop()
        GPIO.cleanup()
    finally:
        print('exiting.')
        pwm.stop()
        GPIO.cleanup()

def frequency_dutycycle_grid_search():
    pin = 12 # 12 = forward pin = purple pin
    seconds = 5
    freq = 100
    duty_cycle = 100
    try:
        GPIO.setup(pin, GPIO.OUT)
        pwm = GPIO.PWM(pin, freq)
        pwm.start(duty_cycle)
        print(f'pin {pin} is now on freq {freq} and duty cycle {duty_cycle} and for the next {seconds} seconds')
        time.sleep(seconds)
        freq_duty_cycle_map = [
            (100, 100),
            (100, 50),
            (100, 18.18),
            (100, 19),
            (1000000, 19),
            (1000000, 7),
            (1000000, 0.01),
            (1000000, 100),
            (2000,19),
            (2000,10),
            (20000,10),
            (20000,2),
        ]
        for freq, duty_cycle in freq_duty_cycle_map:
            print(f'pin {pin} is now on freq {freq} and duty cycle {duty_cycle} and for the next {seconds} seconds')
            pwm.ChangeFrequency(freq)
            pwm.ChangeDutyCycle(duty_cycle)
            time.sleep(seconds)

        print('exiting.')
        pwm.stop()
        GPIO.cleanup()
    except:
        print('exiting.')
        pwm.stop()
        GPIO.cleanup()
if __name__ == '__main__':
    #main()
    test_forward()
    #set_freq_and_duty_cycle()
    #frequency_dutycycle_grid_search()
