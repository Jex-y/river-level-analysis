# river-level-analysis
https://dpds.weatheronline.co.uk/historical_data/weather_stations_download/#forward  #link for spenny weather data


Chat GPT code for neural network in pytorch w 7448 inputs (49x4x38):

import torch
import torch.nn as nn

# Define a custom neural network class
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init()
        # Define the layers of the network
        self.input_layer = nn.Linear(7448, 128)  # 7448 input features, 128 hidden units
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(128, 1)  # 1 output unit for regression or binary classification

    def forward(self, x):
        # Define the forward pass
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x

# Create an instance of the neural network
model = NeuralNetwork()

# Print the model architecture
print(model)
