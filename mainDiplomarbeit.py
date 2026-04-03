import cv2
import numpy as np
import sys
import os
import time
import serial
from gpiozero import DigitalOutputDevice


Counter_Harmed = Counter_Safe = Counter_Unharmed = 0
frame_counter = 0
MIN_CONFIDENCE = 0.4
MIN_CONFIDENCE_H = 0.7
MIN_CONFIDENCE_S = 0.95
MIN_CONFIDENCE_U = 0.3
MODEL_PATH = "trained.tflite"
LABEL_PATH = "labels.txt"
MIN_BLOB_AREA = 500  # Mindestgröße in Pixeln, damit es als "Blob" zählt
COM_PORT = '/dev/ttyAMA0'        
BAUD_RATE = 9600 
uart_aktiv = True
alert_pin = DigitalOutputDevice(17)     #Ändern auf jeweiligen IO Pin
initDone = False
CamStart = True
initError = False
found_label =""
found_color = ""
RunErkennung = False
try:
    import imp
except ImportError:
    import importlib.util
    class MockImp:
        @staticmethod
        def find_module(name, path=None):
            spec = importlib.util.find_spec(name, path)
            return spec if spec else None
        @staticmethod
        def load_module(name, file, pathname, description):
            return __import__(name)
    sys.modules['imp'] = MockImp()

sys.path.append('/usr/lib/python3/dist-packages')

try:
    import tensorflow.lite as tflite
    from picamera2 import Picamera2
except ImportError as e:
    print(f"Fehler: {e}")
    sys.exit()

if uart_aktiv == True:
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, parity = serial.PARITY_NONE, stopbits = serial.STOPBITS_ONE, bytesize = serial.EIGHTBITS, timeout=1)
        print(f"Erfolgreich mit {COM_PORT} verbunden!")
    except Exception as e:
        exit()




color_ranges = {
    "ROT":  [(np.array([70, 70, 70]), np.array([135,255,255]))],
    "GELB": [(np.array([20, 100, 100]), np.array([35, 255, 255]))], #oaschlecken
    "GRUEN": [(np.array([35, 50, 50]), np.array([90, 255, 255]))]
}

def load_labels(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return {i: line.strip() for i, line in enumerate(f.readlines())}
    return {0: "background", 1: "H", 2: "S", 3: "U"}

def CamVorverarbeitung():
    global input_data, prep_img, frame_display, im_h, im_w
    frame_display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    im_h, im_w, _ = frame_display.shape

    prep_img = cv2.resize(frame_rgb, (ki_width, ki_height))
    if is_grayscale:
        prep_img = cv2.cvtColor(prep_img, cv2.COLOR_RGB2GRAY)
        prep_img = np.expand_dims(prep_img, axis=-1)
    
    input_data = np.expand_dims(prep_img, axis=0)
    if is_int8:
        input_data = (input_data.astype(np.float32) - 128).astype(np.int8)
    else:
        input_data = (input_data / 255.0).astype(np.float32)

def AiAusführung():
    global output_data, grid_h, grid_w, num_classes
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    grid_h, grid_w, num_classes = output_data.shape

# --- NEU: FARBERKENNUNG FUNKTION ---
def Farberkennung(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    found_colors = []

    for color_name, ranges in color_ranges.items():
        mask = None
        for (lower, upper) in ranges:
            target_mask = cv2.inRange(hsv, lower, upper)
            mask = target_mask if mask is None else cv2.bitwise_or(mask, target_mask)
        
        # Blobs finden
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > MIN_BLOB_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame_display, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv2.putText(frame_display, color_name, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                found_colors.append(color_name)
    return found_colors

def SerialWrite(obj):
    
    message = b"{obj}"
    if ser and ser.is_open:
        ser.write(message.encode('utf-8'))

def SerialRecieve():
    recieved = ""
    if ser.in_waiting > 0:
        recieved = ser.read().decode('utf-8', errors = 'ignore')
        return recieved
            

def SerialInit():
    if SerialRecieve() == "I":
        SerialWrite("OK")
        initDone = True
        return initDone
    else:
        return initError

# TFLite Initialisierung
LABELS = load_labels(LABEL_PATH)
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
ki_height, ki_width = input_details[0]['shape'][1:3]
is_grayscale = (input_details[0]['shape'][3] == 1)
is_int8 = (input_details[0]['dtype'] in [np.int8, np.uint8])

# Cam Start
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(config)
picam2.start()

try:

    #if SerialInit() == initDone:
    #    CamStart = True
    #elif SerialInit() == initError:
    #    CamStart = False


    cmd = SerialRecieve()
    print(f">>> Startschuss: {cmd}<<<")

    while CamStart:
        cmd = SerialRecieve()
        if cmd == "E" or RunErkennung:
            RunErkennung = True
            print(f">>> Schwanz: {cmd}<<<")
            frame_rgb = picam2.capture_array()
            CamVorverarbeitung() # Erzeugt frame_display (BGR)
            AiAusführung()
          
            found_label = None
            max_score = 0

            # Scan nach KI Treffern
            for y in range(grid_h):
                for x in range(grid_w):
                    for class_id in range(1, num_classes):
                        raw_score = output_data[y][x][class_id]
                        score = (float(raw_score) + 128.0) / 255.0 if is_int8 else float(raw_score)
                        
                        if score > MIN_CONFIDENCE and score > max_score:
                            max_score = score
                            found_label = LABELS.get(class_id, "??")
                            #obj_trans = LABELS.get(class_id, "??")
                            
                            pos_x = int((x + 0.5) * (im_w / grid_w))
                            pos_y = int((y + 0.5) * (im_h / grid_h))
                            cv2.circle(frame_display, (pos_x, pos_y), 10, (0, 255, 0), -1)


            if found_label:
                frame_counter += 1
                if found_label == "H" and max_score > MIN_CONFIDENCE_H: Counter_Harmed += 1
                elif found_label == "S" and max_score > MIN_CONFIDENCE_S: Counter_Safe += 1
                elif found_label == "U" and max_score > MIN_CONFIDENCE_U: Counter_Unharmed += 1
                
            if frame_counter >= 15:
                counts = {'H': Counter_Harmed, 'S': Counter_Safe, 'U': Counter_Unharmed}
                CamTransmit = max(counts, key=counts.get)
                print(f">>> ÜBERTRAGUNG: {CamTransmit}<<<")
                #SerialWrite(CamTransmit)
                ser.write(CamTransmit.encode('utf-8'))
                Counter_Harmed = Counter_Safe = Counter_Unharmed = 0
                frame_counter = 0
                RunErkennung = False

            cv2.imshow('Camera Detection', frame_display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        


finally:
    picam2.stop()
    cv2.destroyAllWindows()
