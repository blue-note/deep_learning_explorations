import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np

from deep_learning_explorations.visualize import NetworkVisualization


# Define a simple fully connected feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleNN, self).__init__()
        
        # Initial layer
        self.initial = nn.Linear(input_size, hidden_size)
        
        # Intermediate layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 2):  # Subtract 2 because we already have the initial and final layers
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        
        # Final layer
        self.final = nn.Linear(hidden_size, output_size)
        
        self.relu = nn.ReLU()

    def forward(self, input):
        outputs = []
        
        x = self.initial(input)
        outputs.append(x)
        
        for layer in self.layers:
            x = self.relu(x)
            outputs.append(x)
            x = layer(x)
            outputs.append(x)
        
        x = self.final(x)
        outputs.append(x)
        
        return tuple(outputs)

    
def train(model, dataloader, criterion, optimizer, num_epochs=10):
    # Set the model to training mode
    model.train()

    # Store all losses for visualization
    all_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, labels in dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)[-1]

            # Convert labels to the correct shape and type
            labels = labels.long().squeeze()

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate loss for the current batch
            epoch_loss += loss.item()

        # Compute average loss for the epoch
        epoch_loss /= len(dataloader)
        all_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    return all_losses


def generate_2D_coordinate_dataset(num_points=1000, radius=3.0):
    """
    Generate a dataset of 2D points and label them as either inside (1) or outside (0) of a circle.

    Parameters:
    - num_points: Number of points to generate.
    - radius: Radius of the circle.

    Returns:
    - points: A numpy array of shape (num_points, 2) containing the 2D points.
    - labels: A numpy array of shape (num_points,) containing the labels (1 for inside, 0 for outside).
    """

    # Randomly generate points in the range [-1.5*radius, 1.5*radius]
    points = np.random.uniform(-1.5*radius, 1.5*radius, size=(num_points, 2))

    # Calculate the distance of each point from the origin (0, 0)
    distances = np.linalg.norm(points, axis=1)

    # Label points as inside (1) or outside (0) based on the distance
    labels = (distances <= radius).astype(int)

    return points, labels

def get_model_outputs(model, points_tensor):
    with torch.no_grad():
        model.eval()
        intermediates = model(points_tensor)
        
        # Convert the logits of the last intermediate to probabilities using softmax
        probabilities = torch.nn.functional.softmax(intermediates[-1], dim=1)
        
        # Convert intermediates to numpy (except the last one which is already converted)
        intermediates_np = [intermediate.numpy() for intermediate in intermediates[:-1]]
        
    return intermediates_np, probabilities.numpy()

def prepare_dataset_and_run_model():
    # 1. Prepare the dataset and dataloader
    points, labels = generate_2D_coordinate_dataset(num_points=1000, radius=1.0)

    # Convert the numpy arrays to PyTorch tensors
    points_tensor = torch.tensor(points, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)  # Reshape to (num_points, 1)

    # Create a TensorDataset and DataLoader
    dataset = data.TensorDataset(points_tensor, labels_tensor)
    dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)

    # 2. Initialize the model, criterion, and optimizer
    input_size = 2
    hidden_size = 3
    output_size = 2
    num_layers = 3
    model = SimpleNN(input_size, hidden_size, num_layers, output_size)

    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 3. Train the model
    all_losses = train(model, dataloader, criterion, optimizer, num_epochs=50)

    # 4. Get the outputs of the model for all data points
    intermediate_outputs, output_probabilities = get_model_outputs(model, points_tensor)

    # 5. Visualize the network
    network_viz = NetworkVisualization(points, labels, intermediate_outputs, output_probabilities)
    network_viz.render()

    return model, all_losses



