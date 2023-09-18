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

# Version: 1.0

import tensorflow as tf
from tensorflow.keras.layers import Input, Convolution3D, BatchNormalization, Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

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
#cnn3d_model = Permeability3DCNN(input_shape=(64, 64, 64, 1), category=1)
#cnn3d_model.summary()
#model = cnn3d_model.model 

#model.compile(optimizers.Adam(lr=LRT, beta_1=0.9, beta_2=0.999, epsilon=0.000001, decay=0.1)
#                   , loss='mean_squared_error', metrics=['mean_squared_error'])

#results = model.fit(input_training, output_training, batch_size=Nb, epochs=Np, shuffle=True, verbose=1, validation_split=0.0, validation_data=(input_validation, output_validation))
