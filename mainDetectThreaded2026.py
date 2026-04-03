import cv2
import numpy as np
import threading
import time
import os
import sys
import serial
from gpiozero import DigitalOutputDevice

# --- INITIALISIERUNG & FALLBACKS ---
# --- NEUER IMPORT-BLOCK ---
try:
    # Wir versuchen das neue LiteRT (Standard ab 2025/2026)
    import ai_edge_litert.interpreter as tflite
    print("LiteRT erfolgreich geladen.")
except ImportError:
    try:
        # Fallback auf das alte TFLite
        import tensorflow.lite as tflite
        print("Klassisches TFLite geladen.")
    except ImportError:
        print("Fehler: Weder LiteRT noch TFLite gefunden!")
        sys.exit()

try:
    from picamera2 import Picamera2
except ImportError:
    print("Fehler: Picamera2 fehlt. Bitte mit 'sudo apt install python3-picamera2' installieren.")
    sys.exit()

    
# --- KONFIGURATION ---
MODEL_PATH = "trained.tflite"
LABEL_PATH = "labels.txt"
MIN_CONFIDENCE = 0.4
SERIAL_PORT = '/dev/ttyACM0' # Bitte ggf. auf /dev/ttyUSB0 prüfen
BAUD_RATE = 9600
TRIGGER_PIN = 17 # Gemeinsamer Pin für beide Kameras

# Globaler Pin (Shared Resource)
output_pin = DigitalOutputDevice(TRIGGER_PIN)

# Serial Setup
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
except:
    print("Serial nicht gefunden -> Simulationsmodus aktiv."); ser = None

def load_labels(path):
    # Mapping laut Grafik: H=Red, S=Yellow, U=Green
    if os.path.exists(path):
        with open(path, 'r') as f: 
            return {i: line.strip() for i, line in enumerate(f.readlines())}
    return {0: "background", 1: "H", 2: "S", 3: "U"}

LABELS = load_labels(LABEL_PATH)

# --- SERIAL HELFER ---
def SerialWrite(obj, camside=None):
    """Sendet Nachrichten im Format <OK> oder <LH>, <RS> etc."""
    if ser:
        msg = f"<{camside}{obj}>\n" if camside else f"<{obj}>\n"
        ser.write(msg.encode('utf-8'))
        print(f"[SERIAL] Gesendet: {msg.strip()}")

# --- KAMERA THREAD KLASSE ---
class CameraAIThread(threading.Thread):
    def __init__(self, cam_id, side_code):
        super().__init__()
        self.cam_id = cam_id
        self.side_code = side_code # "L" oder "R"
        self.enabled = False
        self.running = True
        
        # Zähler
        self.Counter_Harmed = 0
        self.Counter_Safe = 0
        self.Counter_Unharmed = 0
        self.frame_counter = 0
        
        # KI-Modell laden
        try:
            self.interpreter = tflite.Interpreter(model_path=MODEL_PATH)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.ki_h, self.ki_w = self.input_details[0]['shape'][1:3]
            self.is_int8 = (self.input_details[0]['dtype'] in [np.int8, np.uint8])
            self.ready = True
        except Exception as e:
            print(f"Cam {self.side_code} Fehler: {e}"); self.ready = False

    def reset_logic(self):
        """Setzt Zähler und Pin zurück."""
        self.Counter_Harmed = self.Counter_Safe = self.Counter_Unharmed = 0
        self.frame_counter = 0
        output_pin.off()

    def run(self):
        if not self.ready: return
        
        try:
            # Kamera-Instanz für diesen spezifischen Port
            picam2 = Picamera2(self.cam_id)
            config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
            picam2.configure(config)
            picam2.start()
        except Exception as e:
            print(f"Hardware-Fehler Cam {self.side_code}: {e}"); return

        print(f"Thread {self.side_code} (Cam {self.cam_id}) aktiv und wartet...")

        while self.running:
            if not self.enabled:
                time.sleep(0.1) # IDLE-Modus ("Chillen")
                continue

            # Bildaufnahme
            frame_rgb = picam2.capture_array()
            
            # KI-Vorverarbeitung
            prep_img = cv2.resize(frame_rgb, (self.ki_w, self.ki_h))
            input_data = np.expand_dims(prep_img, axis=0)
            if self.is_int8:
                input_data = (input_data.astype(np.float32) - 128).astype(np.int8)
            else:
                input_data = (input_data / 255.0).astype(np.float32)

            # Inferenz
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            grid_h, grid_w, num_classes = output_data.shape

            found_label = None
            max_score = 0

            for y in range(grid_h):
                for x in range(grid_w):
                    for class_id in range(1, num_classes):
                        raw_score = output_data[y][x][class_id]
                        score = (float(raw_score) + 128.0) / 255.0 if self.is_int8 else float(raw_score)
                        
                        if score > MIN_CONFIDENCE and score > max_score:
                            max_score = score
                            found_label = LABELS.get(class_id, "??")

            # --- PROTOKOLL-ABLAUF ---
            if found_label:
                self.frame_counter += 1
                
                # Schritt 1: Erster Treffer -> Pin HIGH
                if self.frame_counter == 1:
                    output_pin.on()
                    print(f"[{self.side_code}] Erster Kontakt! Pin HIGH.")

                if found_label == "H": self.Counter_Harmed += 1
                elif found_label == "S": self.Counter_Safe += 1
                elif found_label == "U": self.Counter_Unharmed += 1

            # Schritt 2: 5 Bilder erreicht -> Ergebnis senden
            if self.frame_counter >= 5:
                counts = {'H': self.Counter_Harmed, 'S': self.Counter_Safe, 'U': self.Counter_Unharmed}
                cam_transmit = max(counts, key=counts.get)
                
                # Sende Ergebnis (z.B. <LH>)
                SerialWrite(cam_transmit, self.side_code)
                
                # Schritt 3: Pin LOW und IDLE
                output_pin.off()
                print(f"[{self.side_code}] Messung beendet. Pin LOW.")
                self.reset_logic()
                self.enabled = False 

        picam2.stop()

# --- MAIN: PROTOKOLL LISTENER ---
cam_left = CameraAIThread(0, "L")
cam_right = CameraAIThread(1, "R")
cam_left.start()
cam_right.start()

print("Warte auf Befehle vom Arduino...")

try:
    buffer = ""
    while True:
        if ser and ser.in_waiting > 0:
            char = ser.read().decode('utf-8', errors='ignore')
            buffer += char
            
            if ">" in buffer:
                start = buffer.find("<")
                end = buffer.find(">")
                if start != -1 and end > start:
                    cmd = buffer[start+1:end]
                    
                    # --- BEFEHLSAUSWERTUNG ---
                    
                    # INIT
                    if cmd == "I":
                        SerialWrite("OK")
                    
                    # ENABLE (z.B. <LE> oder <RE>)
                    elif cmd == "LE":
                        cam_left.enabled = True
                        SerialWrite("OK")
                    elif cmd == "RE":
                        cam_right.enabled = True
                        SerialWrite("OK")
                    
                    # DISABLE (z.B. <LD> oder <RD>)
                    elif cmd == "LD":
                        cam_left.enabled = False
                        cam_left.reset_logic()
                        SerialWrite("OK")
                    elif cmd == "RD":
                        cam_right.enabled = False
                        cam_right.reset_logic()
                        SerialWrite("OK")
                
                buffer = "" # Puffer nach jedem Paket leeren
        
        time.sleep(0.01)

except KeyboardInterrupt:
    print("System wird beendet...")
    cam_left.running = cam_right.running = False
    cam_left.join()
    cam_right.join()
    output_pin.off()
