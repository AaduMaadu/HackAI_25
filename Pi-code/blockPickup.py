# SPDX-FileCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT

"""Simple test for a standard servo on channel 0 and a continuous rotation servo on channel 1."""
import time
from adafruit_servokit import ServoKit

# Set channels to the number of servo channels on your kit.
# 8 for FeatherWing, 16 for Shield/HAT/Bonnet.
kit = ServoKit(channels=16)
base = 0
shoulder1 = 1
#shoulder2 = 2
elbow = 8
wrist = 9
claw = 10

while(1): #shoulder 2 and wrist is HIGH
    time.sleep(1)
    kit.servo[base].angle = 15
    kit.servo[shoulder1].angle = 0
    #kit.servo[shoulder2].angle = 135
    kit.servo[elbow].angle = 0
    kit.servo[wrist].angle = 180
    kit.servo[claw].angle = 40
    time.sleep(1)
    
    #turn, pick up block, turn, drop block
    kit.servo[base].angle = 45
    time.sleep(1)
    kit.servo[shoulder1].angle = 15
    #kit.servo[shoulder2].angle = 115
    time.sleep(1)
    kit.servo[elbow].angle = 5
    time.sleep(0.5)
    kit.servo[wrist].angle = 125
    time.sleep(1)
    kit.servo[shoulder1].angle = 25
    time.sleep(0.5)
    kit.servo[claw].angle = 7 #grab block
    time.sleep(1)
    kit.servo[shoulder1].angle = 15
    time.sleep(0.5)
    kit.servo[wrist].angle = 180
    time.sleep(0.5)
    kit.servo[elbow].angle = 0
    time.sleep(1)
    kit.servo[shoulder1].angle = 15
    #kit.servo[shoulder2].angle = 180
    kit.servo[base].angle = 0
    time.sleep(1)
    kit.servo[shoulder1].angle = 15
    #kit.servo[shoulder2].angle = 115
    time.sleep(1)
    kit.servo[elbow].angle = 5
    time.sleep(0.5)
    kit.servo[wrist].angle = 125
    time.sleep(1)
    kit.servo[shoulder1].angle = 25
    time.sleep(0.5)
    kit.servo[claw].angle = 40 #drop block
    kit.servo[shoulder1].angle = 15
    time.sleep(0.5)
    time.sleep(1)
    
    
    



    
