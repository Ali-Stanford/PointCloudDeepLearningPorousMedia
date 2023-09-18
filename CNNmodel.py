import tensorflow as tf
from tensorflow.keras.layers import Input, Convolution3D, BatchNormalization, Dense, Dropout, Flatten
from tensorflow.keras.models import Model

class Permeability3DCNN:
    def __init__(self, input_shape, category):
        self.input_shape = input_shape
        self.category = category
        self.model = self._build_model()

    def _build_model(self):
        input_points = Input(shape=self.input_shape)

        # CNN Layers
        filters = [16, 32, 64, 128, 256, 512, 1024]
        kernel_sizes = [2, 2, 2, 2, 2, 2, 1]
        strides = [(2,2,2) for _ in range(6)] + [(2,2,2)]
        x = input_points
        for f, k, s in zip(filters, kernel_sizes, strides):
            x = Convolution3D(f, k, s, activation='relu')(x)
            x = BatchNormalization()(x)
        
        # Dense Layers
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(rate=0.7)(x)

        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(rate=0.7)(x)

        x = Dense(self.category, activation='sigmoid')(x)
        prediction = Flatten()(x)

        return Model(inputs=input_points, outputs=prediction)

    def summary(self):
        self.model.summary()

# Example usage
# cnn3d_model = Permeability3DCNN(input_shape=(64, 64, 64, 1), category=1)
# cnn3d_model.summary()
