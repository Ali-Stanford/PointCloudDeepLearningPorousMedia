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
from tensorflow.keras import layers, Model, Input

class SimplePointNet(Model):
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

model = SimplePointNet(num_points=NUM_POINTS)
model.build((None, NUM_POINTS, 3))
model.summary()
