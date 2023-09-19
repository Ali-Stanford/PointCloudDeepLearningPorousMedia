# In The Name of God
# Implementation of PointNet for supervised learning of permeability of porous media using point clouds

# Author: Ali Kashefi (kashefi@stanford.edu)

# Citation:
# If you use the code, please cite the following journal paper:

# @article{kashefi2021PointNetPorousMedia, 
#  title={Point-cloud deep learning of porous media for permeability prediction},
#  author={Kashefi, Ali and Mukerji, Tapan},
#  journal={Physics of Fluids}, 
#  volume={33}, 
#  number={9}, 
#  pages={097109},
#  year={2021}, 
#  publisher={AIP Publishing LLC}}

# First, we need to import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Convolution3D, BatchNormalization, Dropout, Flatten
from tensorflow.keras import optimizers
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

# Example usage
#cnn3d_model = Permeability3DCNN(input_shape=(64, 64, 64, 1), category=1)
#model = cnn3d_model.model 

#model.compile(optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=0.000001)
#                   , loss='mean_squared_error', metrics=['mean_squared_error'])

# Generate fake data (just to test the model)
#n_training = 100  # Set the number of training samples
#n_validation = 20  # Set the number of validation samples

#input_training = np.random.randn(n_training, 64, 64, 64, 1)
#output_training = np.random.randn(n_training)

#input_validation = np.random.randn(n_validation, 64, 64, 64, 1)
#output_validation = np.random.randn(n_validation)

#results = model.fit(input_training, output_training, batch_size=256, epochs=1500, shuffle=True, verbose=1, validation_data=(input_validation, output_validation))
