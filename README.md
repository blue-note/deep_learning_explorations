# Deep Learning Explorations

This project uses manim to visualize the flow of transformations of a set of 2D coordinate points through a simple neural network classifier which classifies points inside and outside of a circle. The goal is to visualize and gain an intuition for which higher-dimensional transformations a network applies to the input data in order to make the two classes linearly separable. 

Inspiration for this project comes from the following sources: 

1. [Shape of Knowledge](https://percisely.xyz/neural-networks)

2. [Topology of Deep Neural Networks](https://arxiv.org/abs/2004.06093)
   
3. [NNs, Manifolds, Topology](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)

## Installation and Running

1. `poetry install`
2. `poetry shell`
3. `python3 -m deep_learning_explorations`
4. Visualizations of input and output data will be saved in the `media` folder.
