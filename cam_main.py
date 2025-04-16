# modules
import os
import tf,
import uos,
import gc
import sensor
import time
import math
from machine import Pin, UART

# set the pins
pin_alert = Pin("P7", Pin.OUT)
pin_reset = Pin("P8", Pin.IN)

# define grid
max_detection_val_right = 250
max_detection_val_left = 80
mid_detection_val_up = 60
mid_detection_val_down = 240 - mid_detection_val_up

# LAB thresholds for color detection
thresholds = [
    (30, 100, 15, 127, 15, 127),    #generic_red_thresholds
    (33, 100, -51, -19, 4, 43),    #generic_green_thresholds
    (77, 99, -21, -4, 12, 68),    #generic_yellow_thresholds
]

# initialize the camera sensor
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time=2000)
sensor.set_auto_gain(False)
sensor.set_auto_whitebal(False)
clock = time.clock()

# variables
transmission_data = ""
counter_red = counter_green = counter_yellow = 0
counter_harmed = counter_safe = counter_unharmed = 0
frame_counter = 0
objposition = 0
detection_counter = {}
run = False
await_reset = False
obj_detected = False  #flag to check if something was detected
net = None
labels = None
min_confidence = 0.7

# config UART interface
uart = UART(1, 115200, timeout_char=200)

# load training model
try:
    net = tf.load("trained.tflite", load_to_fb=uos.stat('trained.tflite')[6] > (gc.mem_free() - (64*1024)))
except Exception as e:
    raise Exception('Failed to load "trained.tflite", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')
try:
    labels = [line.rstrip('\n') for line in open("labels.txt")]
except Exception as e:
    raise Exception('Failed to load "labels.txt", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

# function to check the reset pin and pause the detection for 3s:
def check_reset():
    if pin_reset.value() == True:
        pin_alert.value(0)
        print("Reset")
        time.sleep(3)
        await_reset = False
        #reset counters
        counter_red = counter_green = counter_yellow = 0
        counter_harmed = counter_safe = counter_unharmed = 0
        frame_counter = 0

# function to check start/end from UART
def check_start():
    data = uart.read()
    if data is not None:
        print(data)
        if "A" in data:
            print("Start")
            run = True
            uart.write("X")

        elif "E" in data:
            print("End")
            run = False
            uart.write("X")

while True:
    # Check start/stop
    check_start()

    # Check reset
    check_reset()

    # Run the detection
    if run and not await_reset:
        clock.tick()
        img = sensor.snapshot()
        
        # draw grid
        img.draw_line(max_detection_val_right, 0 ,max_detection_val_right, 240 , 255, 1)
        img.draw_line(max_detection_val_left, 0 ,max_detection_val_left, 240 , 255, 1)

        #Start of color detection:
        for color_type, threshold in enumerate(thresholds):
            for blob in img.find_blobs(
                [threshold],
                pixels_threshold=200,
                area_threshold=200,
                merge=True,
            ):
                obj_detected = True  #at least one blob detected, so increase frame_counter
                if color_type == 0:
                    counter_red += 1
                elif color_type == 1:
                    counter_green += 1
                elif color_type == 2:
                    counter_yellow += 1

                #graphical bounding of the blob
                [x, y, w, h] = blob.rect()
                center_x = math.floor(x + (w / 2))
                center_y = math.floor(y + (h / 2))
                objposition = center_x
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
        for victim_type, detection_list in enumerate(net.detect(img, thresholds=[(math.ceil(min_confidence * 255), 255)])):
                if (victim_type == 0): continue
                if (len(detection_list) == 0): continue
                for d in detection_list:
                    obj_detected = True
                    [x, y, w, h] = d.rect()
                    center_x = math.floor(x + (w / 2))
                    center_y = math.floor(y + (h / 2))
                    img.draw_circle((center_x, center_y, 12), color=(0, 0, 255), thickness=4)
                    img.draw_rectangle(d.rect(), color=(0, 0, 255), thickness=2)
                    img.draw_string(10, 10, labels[i], color=(0, 0, 255), scale=3)
                    objposition = center_x
                    
                    if labels[victim_type] == "H":
                        counter_harmed += 1
                    if labels[victim_type] == "U":
                        counter_unharmed += 1
                    if labels[victim_type] == "S":
                        counter_safe += 1
        # increase counter if an object is detected
        if obj_detected:
            frame_counter += 1

        # if more than 5 detections occured, pick the most occuring victim
        if (frame_counter >= 5):
            detection_counter = {
            'R': counter_red,
            'G': counter_green,
            'Y': counter_yellow,
            'H': counter_harmed,
            'S': counter_safe,
            'U': counter_unharmed
            }
            transmission_data = max(detection_counter, key=detection_counter.get)

            # check if object is within the bounds, output to the Arduino
            if (objposition <= max_detection_val_right and objposition >= max_detection_val_left):
                uart.write(transmission_data.encode())  #send color via UART
                print(f"Detected Object: {transmission_data}")
                await_reset = True

            #reset counters
            counter_red = counter_green = counter_yellow = 0
            counter_harmed = counter_safe = counter_unharmed = 0
            frame_counter = 0

        #set pin_alert to high if more than 10 detections
        if frame_counter >= 2:
            pin_alert.value(1)
            last_pin_alert_time = time.ticks_ms()

        # check if 2 seconds have passed since pin_alert was set, reset if so
        if pin_alert.value() == 1 and time.ticks_diff(time.ticks_ms(), last_pin_alert_time) > 2000:
            pin_alert.value(0)
            print("pin_alert zurückgesetzt")
            frame_counter = 0
    else:
        time.sleep(0.01)
