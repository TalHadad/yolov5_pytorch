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
        pwm = GPIO.PWM(num, 20000)
        pwms[num] = pwm

    try:
        for pwm_num in pwms:
            pwms[pwm_num].start(2)

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


if __name__ == '__main__':
    #main()
    test_forward()