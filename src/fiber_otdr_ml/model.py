from __future__ import annotations
import tensorflow as tf

def build_cnn1d(input_len: int = 256, filters: int = 32, classes: int = 2) -> tf.keras.Model:
    """Small 1D CNN for classifying windows (normal vs event)."""
    inputs = tf.keras.Input(shape=(input_len, 1))
    x = tf.keras.layers.Conv1D(filters, 5, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(filters*2, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
