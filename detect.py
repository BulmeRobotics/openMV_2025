import cv2
import numpy as np
import sys
import os
import time

# --- 1. DER IMP-FIX (Python 3.13) ---
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

# --- 2. VARIABLEN (VOM OPENMV CODE) ---
Counter_Harmed = Counter_Safe = Counter_Unharmed = 0
frame_counter = 0
MIN_CONFIDENCE = 0.6 
MODEL_PATH = "trained.tflite"
LABEL_PATH = "labels.txt"

def load_labels(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return {i: line.strip() for i, line in enumerate(f.readlines())}
    return {0: "background", 1: "H", 2: "S", 3: "U"}

# --- 3. KI-INITIALISIERUNG ---
LABELS = load_labels(LABEL_PATH)
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

ki_height, ki_width = input_details[0]['shape'][1:3]
is_grayscale = (input_details[0]['shape'][3] == 1)
is_int8 = (input_details[0]['dtype'] in [np.int8, np.uint8])

# --- 4. KAMERA STARTEN ---
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(config)
picam2.start()

print(f"Edge Impulse FOMO bereit! (Labels: {list(LABELS.values())})")

try:
    while True:
        frame_rgb = picam2.capture_array()
        frame_display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        im_h, im_w, _ = frame_display.shape

        # --- KI VORBEREITUNG ---
        prep_img = cv2.resize(frame_rgb, (ki_width, ki_height))
        if is_grayscale:
            prep_img = cv2.cvtColor(prep_img, cv2.COLOR_RGB2GRAY)
            prep_img = np.expand_dims(prep_img, axis=-1)
        
        input_data = np.expand_dims(prep_img, axis=0)
        if is_int8:
            input_data = (input_data.astype(np.float32) - 128).astype(np.int8)
        else:
            input_data = (input_data / 255.0).astype(np.float32)

        # --- KI AUSFÜHRUNG (FOMO) ---
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # FOMO hat nur einen Output: [1, Grid_H, Grid_W, Anzahl_Klassen]
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        grid_h, grid_w, num_classes = output_data.shape

        found_label = None
        max_score = 0

        # Wir scannen das Gitter nach dem besten Treffer
        for y in range(grid_h):
            for x in range(grid_w):
                # Wir überspringen Index 0 (Background)
                for class_id in range(1, num_classes):
                    raw_score = output_data[y][x][class_id]
                    # Score normalisieren
                    # Wir wandeln den Wert erst in float um, um den Overflow zu verhindern
                    score = (float(raw_score) + 128.0) / 255.0 if is_int8 else float(raw_score)
                    
                    if score > MIN_CONFIDENCE and score > max_score:
                        max_score = score
                        found_label = LABELS.get(class_id, "??")
                        
                        # Zeichne Punkt an der Gitter-Position
                        pos_x = int((x + 0.5) * (im_w / grid_w))
                        pos_y = int((y + 0.5) * (im_h / grid_h))
                        cv2.circle(frame_display, (pos_x, pos_y), 10, (0, 255, 0), -1)

        # --- AUSWERTUNG (DEINE LOGIK) ---
        if found_label:
            frame_counter += 1
            if found_label == "H": Counter_Harmed += 1
            elif found_label == "S": Counter_Safe += 1
            elif found_label == "U": Counter_Unharmed += 1
            
            #cv2.putText(frame_display, f"{found_label} ({int(max_score*100)}%)", 
            #            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # --- HANDOVER ---
        if frame_counter >= 5:
            counts = {'H': Counter_Harmed, 'S': Counter_Safe, 'U': Counter_Unharmed}
            CamTransmit = max(counts, key=counts.get)
            print(f">>> ÜBERTRAGUNG: {CamTransmit} <<<")
            print(f">>> Confidence: {max_score} <<<")
            # Reset
            Counter_Harmed = Counter_Safe = Counter_Unharmed = 0
            frame_counter = 0
            #time.sleep(0.5)

        cv2.imshow('Pi 5 FOMO Detection', frame_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
