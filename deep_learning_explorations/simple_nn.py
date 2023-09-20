import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np

from deep_learning_explorations.visualize import InputPointVisualization, OutputPointVisualization


# Define a simple fully connected feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
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
            outputs = model(inputs)

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

def get_model_output(model, points_tensor):
    with torch.no_grad():
        model.eval()
        outputs = model(points_tensor)
        # Convert the logits to probabilities using softmax
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.numpy()

def prepare_dataset_and_train_model(model):
    # 1. Prepare the dataset and dataloader
    points, labels = generate_2D_coordinate_dataset(num_points=1000, radius=1.0)

    point_viz = InputPointVisualization(points, labels)
    point_viz.construct()

    # Convert the numpy arrays to PyTorch tensors
    points_tensor = torch.tensor(points, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)  # Reshape to (num_points, 1)

    # Create a TensorDataset and DataLoader
    dataset = data.TensorDataset(points_tensor, labels_tensor)
    dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)

    # 2. Initialize the model, criterion, and optimizer
    input_size = 2
    hidden_size = 10
    output_size = 2
    model = SimpleNN(input_size, hidden_size, output_size)

    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 3. Train the model
    all_losses = train(model, dataloader, criterion, optimizer, num_epochs=50)

    # 4. Get the output of the model for all data points
    output_probabilities = get_model_output(model, points_tensor)

    # 5. Visualize the output data
    output_viz = OutputPointVisualization(points, labels, output_probabilities)
    output_viz.construct()


    return model, all_losses



