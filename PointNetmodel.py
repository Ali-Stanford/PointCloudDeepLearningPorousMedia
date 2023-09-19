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

import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Reshape, Dropout, Flatten  
from tensorflow.python.keras.layers import Convolution1D, MaxPooling1D, BatchNormalization
from tensorflow.python.keras.layers import Lambda, concatenate
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import initializers
from tensorflow.python.keras import layers

class PermeabilityPointNet(Model):
    def __init__(self, num_points, num_classes=1):
        super(SimplePointNet, self).__init__()
        
        self.num_points = num_points
        self.num_classes = num_classes

    def SharedMLP(self, x, filters):
        x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
        x = layers.BatchNormalization(momentum=0.0)(x)
        return layers.Activation("relu")(x)

    def MLP(self, x, filters):
        x = layers.Dense(filters)(x)
        x = layers.BatchNormalization(momentum=0.0)(x)
        return layers.Activation("relu")(x)

    def call(self, inputs):
        x = self.SharedMLP(inputs, 16)
        x = self.SharedMLP(x, 16)
        x = self.SharedMLP(x, 16)
        x = self.SharedMLP(x, 32)
        x = self.SharedMLP(x, 256)
        x = layers.GlobalMaxPooling1D()(x)
        x = self.MLP(x, 128)
        x = layers.Dropout(0.3)(x)
        x = self.MLP(x, 64)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation="sigmoid")(x)
        return outputs

# Example usage
#model = PermeabilityPointNet(num_points=n_points)
#model.build((None, n_points, 3))

#model.compile(optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=0.000001, decay=0.1)
#                   , loss='mean_squared_error', metrics=['mean_squared_error'])

#input_training= zeros([n_training,n_points,3],dtype='f')
#output_training= zeros([n_training],dtype='f')

#results = model.fit(input_training, output_training, batch_size=256, epochs=1500, shuffle=True, verbose=1, validation_split=0.0, validation_data=(input_validation, output_validation))
