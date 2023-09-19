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

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Convolution1D, BatchNormalization, Dropout, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import layers

def SharedMLP(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

def MLP(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

class PermeabilityPointNet(Model):
    def __init__(self, num_points, num_classes=1):
        
        self.num_points = num_points
        self.num_classes = num_classes

        inputs = Input(shape=(self.num_points, 3))
        x = SharedMLP(inputs, 16)
        x = SharedMLP(x, 16)
        x = SharedMLP(x, 16)
        x = SharedMLP(x, 32)
        x = SharedMLP(x, 256)
        x = layers.GlobalMaxPooling1D()(x)
        x = MLP(x, 128)
        x = layers.Dropout(0.3)(x)
        x = MLP(x, 64)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation="sigmoid")(x)
        
        super().__init__(inputs=inputs, outputs=outputs)

# Example usage
#n_points = 1024
#model = PermeabilityPointNet(num_points=n_points)
#model.summary()

#model.compile(optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=0.000001)
                   , loss='mean_squared_error', metrics=['mean_squared_error'])

# Generate fake data
#n_training = 100
#n_validation = 20
#input_training = np.random.randn(n_training, n_points, 3)
#output_training = np.random.randn(n_training)

#input_validation = np.random.randn(n_validation, n_points, 3)
#output_validation = np.random.randn(n_validation)

#results = model.fit(input_training, output_training, batch_size=256, epochs=1500, shuffle=True, verbose=1, validation_data=(input_validation, output_validation))
