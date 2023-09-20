# __main__.py

from deep_learning_explorations.simple_nn import SimpleNN, generate_2D_coordinate_dataset, prepare_dataset_and_train_model
from deep_learning_explorations.visualize import MatrixMultiplication, NeuralNetworkScene
from manim import *
from torch import nn, optim

# Visualize the neural network using Manim
class MainScene(NeuralNetworkScene):
    def construct(self):
        # Assuming NeuralNetworkScene takes a model as an argument
        super().construct(model)
        # Add any additional animations or visualizations here

# Render the scene
if __name__ == "__main__":
    model = SimpleNN(input_size=2, hidden_size=3, output_size=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
        
    prepare_dataset_and_train_model(model)

    # scene = NeuralNetworkScene(model)
    # scene.render()


