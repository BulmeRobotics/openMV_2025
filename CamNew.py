# modules
import os
import tf
import uos
import gc
import sensor
import time
import math
from machine import Pin, UART


#Set used Pins
alert_pin = Pin("P7", Pin.OUT)
reset_pin = Pin("P8", Pin.IN)
wall_pin = Pin("P9", Pin.IN) #Pin korrigieren

#Set Detection Grid
max_detection_val_right = 270
max_detection_val_left = 60
mid_detection_val_top = 30                           #Change to Set Top Boarder (Je niedrieger desto höher die Grenze)
mid_detection_val_bottom = 190
top_boarder_deactivated = False                      #Set to False, when Horizontal Boarder needed
if top_boarder_deactivated:
    mid_detection_val_bottom = 255

#Variables
center_positon_x = 0    #Position of the detectet Object on the X - Achses
center_positon_y = 0    #Position of the detectet Object on the Y - Achses
reset_time = 0
CamTransmit = ""        #Safe of the detected Object for Handover
Counter_Red = Counter_Green = Counter_Yellow = 0      #Letter Counter
Counter_Harmed = Counter_Safe = Counter_Unharmed = 0  #Colour Counter
frame_counter = 0                                     #Counting every Detection, till Object handover
last_alert_pin_time = 0
variables = ""          #Saves the Object - Counters
run = False             #Start | Stop Flag for Camera Detection
stop_reset = False      #Camera Stop, awaiting a reset
net = None
labels = None
data = False
min_confidence = 0.6
acknowledged = "X"
init_done = True
noWallFlag = False

cam_debug_mode = True   #Start debugging mode, skipping all resets


#Colour Thresholds
thresholds = [
    (16, 34, 10, 54, -11, 55),    #generic_red_thresholds
    (15, 48, -50, -14, -13, 35),     #generic_green_thresholds
    (58, 88, -20, 10, 40, 77),      #generic_yellow_thresholds
]

time.sleep(2)

#Camera initialize
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time=2000)
sensor.set_auto_gain(False)
sensor.set_auto_whitebal(True)
clock = time.clock()

#Initialize UART Interface
uart = UART(1, 115200, timeout_char=200)

#Load training model
try:
    net = tf.load("trained.tflite", load_to_fb=uos.stat('trained.tflite')[6] > (gc.mem_free() - (64*1024)))
except Exception as e:
    raise Exception('Failed to load "trained.tflite", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')
try:
    labels = [line.rstrip('\n') for line in open("labels.txt")]
except Exception as e:
    raise Exception('Failed to load "labels.txt", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

#Function to check the reset pin and pause the detection for 3s
def check_reset():
    global stop_reset
    global reset_time
    global frame_counter
    global run
    global noWallFlag
    if reset_pin.value():
        alert_pin.value(0)
        #time.sleep(1.5)
        print("Reset")
        stop_reset = False
        run = False
        frame_counter = 0
        noWallFlag = True
    elif reset_pin.value() == False & noWallFlag:
        run = True
        noWallFlag = False


#Function to check the start condition from UART and sending X to acknowlage
def check_start():
    global data
    global run
    global Counter_Red
    global Counter_Green
    global Counter_Yellow
    global Counter_Harmed
    global Counter_Safe
    global Counter_Unharmed
    global frame_counter

    data = uart.read()

    if data is not None:
        print(data)
        if "A" in data:
            print("Start")
            run = True

        elif "E" in data:
            print("End")
            run = False

            #Reset Counters
            Counter_Red = Counter_Green = Counter_Yellow = 0
            Counter_Harmed = Counter_Safe = Counter_Unharmed = 0
            frame_counter = 0


def detection_grid():
    #Drawing detection - grid
    img.draw_line(max_detection_val_right, 0 ,max_detection_val_right, 240 , 255, 1)
    img.draw_line(max_detection_val_left, 0 ,max_detection_val_left, 240 , 255, 1)
    if top_boarder_deactivated == False:
        img.draw_line(0, mid_detection_val_top ,255, mid_detection_val_top , 255, 1)
        img.draw_line(0, mid_detection_val_bottom ,255, mid_detection_val_bottom , 255, 1)


def cams_init():
    global run
    global init_done
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)
    sensor.skip_frames(time=2000)
    sensor.set_auto_gain(False)
    sensor.set_auto_whitebal(True)
    run = False


#Main Loop
while True:
    clock.tick()
    img = sensor.snapshot()

    #Cam initialised when Reset Pin high & Init unused
    if (reset_pin.value() & init_done):
        cams_init()

    #Check Start & Reset Pin
    check_start()
    check_reset()

    #Skipps all resets
    if cam_debug_mode:
        run = True
        stop_reset = False

    #Query to start and stop the detection after handover or at thee End
    if run == True and stop_reset == False:

        #Stops Initialisation
        init_done = False

        #Drawing a detection-grid
        detection_grid()

        #Start of color detection:
        for i, threshold in enumerate(thresholds):
            for blob in img.find_blobs(
                [threshold],
                pixels_threshold=200,
                area_threshold=200,
                merge=True,
            ):

                #graphical bounding of the blob
                [x, y, w, h] = blob.rect()
                center_x = math.floor(x + (w / 2))
                center_y = math.floor(y + (h / 2))
                center_positon_x = center_x
                center_positon_y = center_y

                #Top Border
                if (center_y >= mid_detection_val_top and center_positon_y <= mid_detection_val_bottom) or top_boarder_deactivated == True:

                    #Colour Counter increases when detected
                    frame_counter += 1

                    if i == 0 and center_positon_y <= mid_detection_val_bottom:
                        Counter_Red += 1
                    elif i == 1:
                        Counter_Green += 1
                    elif i == 2:
                        Counter_Yellow += 1

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

                    #graphical bounding of the blob
                    [x, y, w, h] = d.rect()
                    center_x = math.floor(x + (w / 2))
                    center_y = math.floor(y + (h / 2))
                    img.draw_circle((center_x, center_y, 12), color=(0, 0, 255), thickness=4)
                    img.draw_rectangle(d.rect(), color=(0, 0, 255), thickness=2)
                    img.draw_string(10, 10, labels[i], color=(0, 0, 255), scale=3)
                    center_positon_x = center_x
                    center_positon_y = center_y
                    #Top Border
                    if (center_y >= mid_detection_val_top and center_y <= mid_detection_val_bottom) or top_boarder_deactivated == True:

                        #Letter Counter increases when detected
                        frame_counter += 1
                        if labels[i] == "H":
                            Counter_Harmed += 1
                            #print(Counter_Harmed)
                        if labels[i] == "U":
                            Counter_Unharmed += 1
                            #print(Counter_Unharmed)
                        if labels[i] == "S":
                            Counter_Safe += 1
                            #print(Counter_Safe)
                    #print("X Achse: ",center_x, "Y Achse: ",center_y)
        #Evaluation and Handover of most detected Object
        if (frame_counter >= 5):
            variables = {
            'R': Counter_Red,
            'G': Counter_Green,
            'Y': Counter_Yellow,
            'H': Counter_Harmed,
            'S': Counter_Safe,
            'U': Counter_Unharmed
            }
            #Find the most detected Object
            CamTransmit = max(variables, key=variables.get)

            #Handover when position is correct
            if (center_positon_x <= max_detection_val_right and center_positon_x >= max_detection_val_left):
                uart.write(CamTransmit.encode())             #Send Object via UART
                print(f"Detected Object: {CamTransmit}")
                stop_reset = True                            #Stop dectection after Handover

            #Reset Counters
            Counter_Red = Counter_Green = Counter_Yellow = 0
            Counter_Harmed = Counter_Safe = Counter_Unharmed = 0
            frame_counter = 0

        #Set Alert Pin to high if more than 3 detections
        if frame_counter >= 2 and alert_pin.value() == False:
            alert_pin.value(1)
            last_alert_pin_time = time.ticks_ms()

        #Check if 3 seconds have passed since Alter Pin was set and reset Alter Pin
        if alert_pin.value() == 1 and time.ticks_diff(time.ticks_ms(), last_alert_pin_time) > 2000:
            alert_pin.value(0)
            print("alert_pin zurückgesetzt")
            frame_counter = 0

####################################################################################################

#Änderungen:

    #Allgemein:
        #Variablen Namen
        #Kommentare
        #Den Flag "Detected" entfernt, frame_counter direkt bei erkennung hochgezählt
        #Beide while Schleifen zum warten, bei Start und bei Erkennung mit If Abfrage ersetzt
    #Entfernt:
        #Abfrage nach UART Reset
        #Horizontale Linien setzung
        #reset_help
        #variables_Color
        #variables_Letter


#Entfernte Code Fragmente:

    #Beginnend Line 110:

        #############################################
            #Wait until starting variable is sent
            #while run == False:
                #check_start()
                #time.sleep(0.1)


            #while stop_reset == True:

                #check_reset()
                #print(stop_reset)
                #time.sleep(0.1)
                #if stop_reset == False: break
        #############################################

    #Beginnt Line 208:

        #############################################
        #check for received data
        #if uart.any():
            #data = uart.read()
            #if data and b"R" in data:  #check if "R" (reset) was received
                #alert_pin.value(0)
                #print("Reset")
                #CamTransmit = "0"
                #Counter_Red = Counter_Green = Counter_Yellow = 0
                #Counter_Harmed = Counter_Safe = Counter_Unharmed = 0
                #frame_counter = 0  #reset frame_counter
                #stop_reset = 0
        #############################################

    #Beginnt Line 115:

        #############################################
        #img.draw_line(0, mid_detection_val_top ,240 ,mid_detection_val_top , 255, 1)
        #img.draw_line(0, mid_detection_val_bottom ,240 ,mid_detection_val_bottom , 255, 1)
        #############################################

