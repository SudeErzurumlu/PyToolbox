import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np

class DynamicNN:
    def __init__(self, input_dim, output_dim):
        """
        Initializes a dynamic neural network model.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = self.build_model()
        self.history = []

    def build_model(self):
        """
        Builds a simple neural network model.
        """
        model = Sequential([
            Dense(64, activation="relu", input_shape=(self.input_dim,)),
            Dropout(0.3),
            Dense(self.output_dim, activation="softmax")
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def dynamic_adjustment(self, epoch, logs):
        """
        Dynamically adjusts learning rate or architecture based on performance.
        """
        loss = logs["loss"]
        self.history.append(loss)
        if len(self.history) > 5 and loss > np.mean(self.history[-5:]):
            print("Adjusting learning rate due to plateau.")
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, 0.5 * self.model.optimizer.learning_rate)

    def train(self, X, y, epochs=10):
        """
        Trains the model with dynamic adjustments.
        """
        self.model.fit(X, y, epochs=epochs, callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=self.dynamic_adjustment)])

# Example Usage:
# X_train, y_train = np.random.rand(1000, 10), np.random.randint(0, 2, 1000)
# dynamic_nn = DynamicNN(10, 2)
# dynamic_nn.train(X_train, y_train, epochs=20)
