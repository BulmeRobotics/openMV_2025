import tensorflow as tf
import numpy as np
import os

# ==========================================
# 1. DATENSATZ LADEN & VORVERARBEITEN
# ==========================================
# Stützpunkt: tf.keras.utils.image_dataset_from_directory (aus dem GitHub-Beispiel)
# Angenommen, die gelabelten Bilder liegen in den Ordnern: dataset/background, dataset/H, dataset/S, dataset/U
DATA_DIR = 'dataset'
IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 32

# Laden der Bilder und automatische Zuweisung von Klassen anhand der Ordnernamen
train_data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='rgb' # Oder 'grayscale', falls is_grayscale=True im Auswertungscode
)

# Normalisierung der Pixelwerte von 0-255 auf 0.0-1.0
train_data = train_data.map(lambda x, y: (x / 255.0, y))

# ==========================================
# 2. AUFBAU DES NEURONALEN NETZWERKS (FCN)
# ==========================================
model = tf.keras.models.Sequential()

# Block 1: Erste Merkmalsextraktion (Kanten, Linien)
model.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(tf.keras.layers.MaxPooling2D()) # Reduziert die Auflösung von 64x64 auf 32x32

# Block 2: Komplexere Merkmale (Rundungen, Ecken)
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D()) # Reduziert die Auflösung von 32x32 auf 16x16

# Block 3: Tiefe Abstraktion
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D()) # Reduziert die Auflösung von 16x16 auf 8x8 -> Das ist unser Grid!

# ANPASSUNG FÜR DIE OBJEKTERKENNUNG (Statt Flatten/Dense wie im GitHub-Beispiel)
# Output Layer: 4 Filter für unsere 4 Klassen (Background, H, S, U)
model.add(tf.keras.layers.Conv2D(4, (3,3), activation='sigmoid', padding='same')) 
# Output-Shape ist nun (Batch, 8, 8, 4) - Exakt das, was `grid_h`, `grid_w`, `num_classes` erwartet!

# Kompilieren des Modells
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ==========================================
# 3. TRAINING (Dummy-Aufruf)
# ==========================================
# model.fit(train_data, epochs=30)

# ==========================================
# 4. TFLITE KONVERTIERUNG & QUANTISIERUNG
# ==========================================
# Repräsentativer Datensatz für die Kalibrierung der INT8-Quantisierung
def representative_data_gen():
    for input_value, _ in train_data.take(100):
        yield [tf.cast(input_value, tf.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimierung auf Speicherplatz und Latenz aktivieren
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# Erzwingen der INT8-Formatierung (Passend zur is_int8 Abfrage im Auswertungscode)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  
converter.inference_output_type = tf.int8 

tflite_quant_model = converter.convert()

# Speichern des fertigen Modells
with open('trained.tflite', 'wb') as f:
    f.write(tflite_quant_model)
