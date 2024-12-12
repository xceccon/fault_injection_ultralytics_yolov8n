import torch
from ultralytics import YOLO
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Load the YOLOv8n model
model = YOLO('yolov8n.pt')  # Downloads the model if not present

# Access the underlying PyTorch model
torch_model = model.model

# Function to find the first Conv2d layer
def find_first_conv(module):
    for name, layer in module.named_modules():
        if isinstance(layer, nn.Conv2d):
            return name, layer
    return None, None

# Find the first Conv2d layer
conv_name, conv_layer = find_first_conv(torch_model)

if conv_layer:
    print(f"First Conv2d Layer: {conv_name}")
    
    # Access the weight tensor
    conv_weights = conv_layer.weight.data  # Shape: (out_channels, in_channels, kernel_height, kernel_width)
    
    print(f"Weight Tensor Shape: {conv_weights.shape}")
    print(f"Weight Tensor Data Type: {conv_weights.dtype}\n")
    
    # Print the first filter's weights across all input channels
    print("First Filter Weights Across All Input Channels:")
    for channel in range(conv_weights.shape[1]):
        print(f"  Input Channel {channel + 1}:")
        print(conv_weights[0, channel])
        print()
    
    # Visualization
    first_filter = conv_layer.weight.data[0].cpu().numpy()  # Shape: (3, 3, 3)
    
    # Plot each input channel's weights
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for i in range(first_filter.shape[0]):
        axs[i].imshow(first_filter[i], cmap='viridis')
        axs[i].set_title(f'Input Channel {i+1}')
        axs[i].axis('off')
    
    plt.suptitle('First Conv2d Filter Weights Across Input Channels')
    plt.show()
    
else:
    print("No Conv2d layer found in the model.")
