import os, tf, uos, gc
import sensor
import time
import math
from machine import Pin, UART

pin2 = Pin("P7", Pin.OUT)
ResetPin = Pin("P8", Pin.IN)
max_detection_val_right = 250
max_detection_val_left = 80
mid_detection_val_up = 60
mid_detection_val_down = 240-mid_detection_val_up
objposition = 0

reset_time = 0
reset_help = 0

variables_Color = variables_Letter = 0

thresholds = [
    (30, 100, 15, 127, 15, 127),    #generic_red_thresholds
    (33, 100, -51, -19, 4, 43),    #generic_green_thresholds
    (77, 99, -21, -4, 12, 68),    #generic_yellow_thresholds
]

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time=2000)
sensor.set_auto_gain(False)
sensor.set_auto_whitebal(False)
clock = time.clock()

CamTransmit = ""
CounterR = CounterG = CounterY = 0
CounterH = CounterS = CounterU = 0
FrameCounter = 0

variables = ""

uart = UART(1, 115200, timeout_char=200)

stop = True

stop_reset = False

net = None
labels = None
min_confidence = 0.6


try:
    net = tf.load("trained.tflite", load_to_fb=uos.stat('trained.tflite')[6] > (gc.mem_free() - (64*1024)))
except Exception as e:
    raise Exception('Failed to load "trained.tflite", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')
try:
    labels = [line.rstrip('\n') for line in open("labels.txt")]
except Exception as e:
    raise Exception('Failed to load "labels.txt", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

data = False

#Checkt den Reset Pin ab und wartet 3 Sekunden:
def checkRESET():

    global stop_reset
    global reset_time
    global reset_help

    if ResetPin.value() == True:
        print("Reset")
        pin2.value(0)
        #,reset_time = time.ticks_ms()
        reset_help = True
        



def checkUART():

    #if uart.any() == True:
    #print("PinderKenis")
    global data
    global stop
    data = uart.read()

    if data is not None:
        print(data)
        if "A" in data:
            print("Start")
            stop = False
            uart.write("X")

        elif "E" in data:
            print("End")
            stop = True
            uart.write("X")

while True:

    print("Pin2Value:", pin2.value())
    checkUART()
    #checkRESET()

    while stop == True:
        checkUART()
        time.sleep(0.1)
        clock.tick()
        img = sensor.snapshot()


    while stop_reset == True:
        checkRESET()
        time.sleep(0.1)
        #print(reset_time)
        if reset_help == True: 
            reset_time = time.ticks_ms()
        if time.ticks_diff(time.ticks_ms(), reset_time) > 3000:
            stop_reset = False
            reset_help = False
        clock.tick()
        img = sensor.snapshot()


    clock.tick()
    img = sensor.snapshot()

    img.draw_line(max_detection_val_right, 0 ,max_detection_val_right, 240 , 255, 1)
    img.draw_line(max_detection_val_left, 0 ,max_detection_val_left, 240 , 255, 1)
    #img.draw_line(0, mid_detection_val_up ,240 ,mid_detection_val_up , 255, 1)
    #img.draw_line(0, mid_detection_val_down ,240 ,mid_detection_val_down , 255, 1)

    detected = False  #flag to check if something was detected

    #Start of color detection:
    for i, threshold in enumerate(thresholds):
        for blob in img.find_blobs(
            [threshold],
            pixels_threshold=200,
            area_threshold=200,
            merge=True,
        ):
            detected = True  #at least one blob detected, so increase FrameCounter
            if i == 0:
                CounterR += 1
                #print("Counter R:", CounterR)
            elif i == 1:
                CounterG += 1
                #print("Counter G:", CounterG)
            elif i == 2:
                CounterY += 1
                #print("Counter Y:", CounterY)

            #graphical bounding of the blob
            [x, y, w, h] = blob.rect()
            center_x = math.floor(x + (w / 2))
            center_y = math.floor(y + (h / 2))
            objposition = center_x
            #print(center_x)
            if blob.elongation() > 0.5:
                img.draw_edges(blob.min_corners(), color=(255, 0, 0))
                img.draw_line(blob.major_axis_line(), color=(0, 255, 0))
                img.draw_line(blob.minor_axis_line(), color=(0, 0, 255))
            img.draw_rectangle(blob.rect())
            img.draw_cross(blob.cx(), blob.cy())
            img.draw_keypoints(
                [(blob.cx(), blob.cy(), int(math.degrees(blob.rotation())))], size=20

            )

    #Start of the letter detection:
    for i, detection_list in enumerate(net.detect(img, thresholds=[(math.ceil(min_confidence * 255), 255)])):
            if (i == 0): continue
            if (len(detection_list) == 0): continue
            for d in detection_list:
                detected = True
                [x, y, w, h] = d.rect()
                center_x = math.floor(x + (w / 2))
                center_y = math.floor(y + (h / 2))
                img.draw_circle((center_x, center_y, 12), color=(0, 0, 255), thickness=4)
                img.draw_rectangle(d.rect(), color=(0, 0, 255), thickness=2)
                img.draw_string(10, 10, labels[i], color=(0, 0, 255), scale=3)
                objposition = center_x
                #if center_x <= max_detection_val_right:
                if labels[i] == "H":
                    CounterH += 1
                    #print("Counter H:", CounterH)
                if labels[i] == "U":
                    CounterU += 1
                    #print("Counter U:", CounterU)
                if labels[i] == "S":
                    CounterS += 1
                    #print("Counter S:", CounterS)


    if detected:
        FrameCounter += 1

    if (FrameCounter >= 5):
        variables = {
        'R': CounterR,
        'G': CounterG,
        'Y': CounterY,
        'H': CounterH,
        'S': CounterS,
        'U': CounterU
        }

        CamTransmit = max(variables, key=variables.get)  #find the most detected color

        if (objposition <= max_detection_val_right and objposition >= max_detection_val_left):
            uart.write(CamTransmit.encode())  #send color via UART
            print(f"Detected Object: {CamTransmit}")
            stop_reset = True


        #reset counters
        CounterR = CounterG = CounterY = 0
        CounterH = CounterS = CounterU = 0
        FrameCounter = 0

    #set pin2 to high if more than 10 detections
    if FrameCounter >= 2:
        pin2.value(1)
        last_pin2_time = time.ticks_ms()

    # check if 5 seconds have passed since pin2 was set
    if pin2.value() == 1 and time.ticks_diff(time.ticks_ms(), last_pin2_time) > 3000:
        pin2.value(0)
        print("Pin2 zurückgesetzt")
        FrameCounter = 0

    #check for received data
    if uart.any():
        data = uart.read()
        if data and b"R" in data:  #check if "R" (reset) was received
            pin2.value(0)
            print("Reset")
            CamTransmit = "0"
            CounterR = CounterG = CounterY = 0
            CounterH = CounterS = CounterU = 0
            FrameCounter = 0  #reset FrameCounter
            stop_reset = 0
