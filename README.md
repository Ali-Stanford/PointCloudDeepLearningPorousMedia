# Point-Cloud Deep Learning of Porous Media for Permeability Prediction

![pic](./Fig_4_pointnetperm-1.png)
![pic](./Fig_1_pointnetperm-1.png)

**Author:** Ali Kashefi (kashefi@stanford.edu) <br>
**Description:** Implementation of PointNet for *supervised learning* of permeability of porous media using point clouds <br>
**Version:** 1.0 <br>

**Citation** <br>
If you use the code, please cite the following journal paper: <br>
**[Point-cloud deep learning of porous media for permeability prediction](https://doi.org/10.1063/5.0063904)**

    @article{kashefi2021PointNetPorousMedia, 
      title={Point-cloud deep learning of porous media for permeability prediction},
      author={Kashefi, Ali and Mukerji, Tapan},
      journal={Physics of Fluids}, 
      volume={33}, 
      number={9}, 
      pages={097109},
      year={2021}, 
      publisher={AIP Publishing LLC}}

**Abstract** <br>
We propose a novel deep learning framework for predicting the permeability of porous media from their digital images. Unlike convolutional neural networks, instead of feeding the whole image volume as inputs to the network, we model the boundary between solid matrix and pore spaces as point clouds and feed them as inputs to a neural network based on the PointNet architecture. This approach overcomes the challenge of memory restriction of graphics processing units and its consequences on the choice of batch size and convergence. Compared to convolutional neural networks, the proposed deep learning methodology provides freedom to select larger batch sizes due to reducing significantly the size of network inputs. Specifically, we use the classification branch of PointNet and adjust it for a regression task. As a test case, two and three dimensional synthetic digital rock images are considered. We investigate the effect of different components of our neural network on its performance. We compare our deep learning strategy with a convolutional neural network from various perspectives, specifically for maximum possible batch size. We inspect the generalizability of our network by predicting the permeability of real-world rock samples as well as synthetic digital rocks that are statistically different from the samples used during training. The network predicts the permeability of digital rocks a few thousand times faster than a lattice Boltzmann solver with a high level of prediction accuracy.

**Questions?** <br>
If you have any questions or need assistance, please do not hesitate to contact Ali Kashefi (kashefi@stanford.edu) via email. 

**About the Author** <br>
Please see the author's website: [Ali Kashefi](https://web.stanford.edu/~kashefi/) 
