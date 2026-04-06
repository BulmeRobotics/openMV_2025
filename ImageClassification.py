import tensorflow as tf
import numpy as np
import os

#1. DATENSATZ LADEN & VORVERARBEITEN

#Direction zu den Datensätzen: 'dataset/background', 'dataset/H', 'dataset/S', 'dataset/U'
DATA_DIR = 'dataset'
IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 32

print("Lade Trainings- und Validierungsdaten...")

#Trainingsdaten laden (80% der Bilder, der Rest bleibt für die Validierung) - seed stellt Reproduzierbarkeit sicher
train_data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='rgb' 
)

#Validierungsdaten laden (20% der Bilder)
val_data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='rgb'
)

#Normalisierung: Pixelwerte von [0, 255] auf den Bereich [0.0, 1.0] skalieren
train_data = train_data.map(lambda x, y: (x / 255.0, y))
val_data = val_data.map(lambda x, y: (x / 255.0, y))

#2. AUFBAU DES FULLY CONVOLUTIONAL NETWORKS
print("Initialisiere Netzwerkarchitektur...")
model = tf.keras.models.Sequential()

# Block 1: Initiale Merkmalsextraktion
model.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(tf.keras.layers.MaxPooling2D()) # Reduktion auf 32x32

# Block 2: Mittlere Abstraktionsebene
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D()) # Reduktion auf 16x16

# Block 3: Tiefe Abstraktionsebene
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D()) # Reduktion auf 8x8 (Das finale Grid)

# Output Layer für Objekterkennung (FCN-Ansatz)
# 4 Filter repräsentieren die 4 Klassen (Background, H, S, U)
model.add(tf.keras.layers.Conv2D(4, (3,3), activation='sigmoid', padding='same')) 

# Modell kompilieren
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#3. MODELLTRAINING MIT CALLBACKS
print("Starte Modelltraining...")

#Definition der Überwachungsmechanismen während des Trainings
callbacks = [
    # Verhindert Overfitting durch vorzeitigen Abbruch, wenn keine Verbesserung auftritt
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5,          
        restore_best_weights=True
    ),
    # Speichert den besten Modellzustand kontinuierlich ab
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model_checkpoint.h5',
        monitor='val_accuracy',
        save_best_only=True
    )
]

#Ausführung des Trainings
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,                
    callbacks=callbacks
)
print("Training erfolgreich abgeschlossen!")

#4. TFLITE KONVERTIERUNG & POST-TRAINING QUANTIZATION (INT8)
print("Konvertiere Modell für den Raspberry Pi (TFLite INT8)...")

#Generator-Funktion liefert Kalibrierungsdaten für die 8-Bit Quantisierung
def representative_data_gen():
    for input_value, _ in train_data.take(100):
        yield [tf.cast(input_value, tf.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)

#Optimierungsstrategien festlegen
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

#Harte Limitierung auf INT8-Operationen für maximale Hardware-Effizienz
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  
converter.inference_output_type = tf.int8 

#Eigentliche Konvertierung durchführen
tflite_quant_model = converter.convert()

#Speichern der finalen, einsatzbereiten Datei
with open('trained.tflite', 'wb') as f:
    f.write(tflite_quant_model)

print("Modell 'trained.tflite' wurde erfolgreich generiert und gespeichert.")
