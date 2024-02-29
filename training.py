import chess
import random
import numpy as np
import chess.engine
from tensorflow.keras import models, layers, optimizers


def train_model(size):
    model = models.Sequential([
        layers.Conv2D(size, kernel_size=4, padding='same', activation='relu', input_shape=(13, 8, 8)),
        layers.Conv2D(size, kernel_size=4, padding='same', activation='relu'),
        layers.Conv2D(size, kernel_size=4, padding='same', activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer=optimizers.Adam(), loss='mean_squared_error')
    return model


def load_dataset(filename):
    data = np.load(filename, allow_pickle=True)
    return data["X"], data["y"]


def train(size, filename, epochs):
    model = train_model(size)
    X, y = load_dataset("dataset.npz")
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=1)
    return model


model = train(size=32, filename="dataset.npz", epochs=100)
model.save('model.keras')
