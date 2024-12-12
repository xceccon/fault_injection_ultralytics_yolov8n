import torch
import pickle
import struct

# Load the PyTorch model file
file_path = 'yolov8n.pt'

# Step 1: Open the file in binary mode
with open(file_path, 'rb') as f:
    file_content = f.read()

# Step 2: Analyze the first few bytes
print("First 100 bytes of the file:")
print(file_content[:100])

# Step 3: Use PyTorch to load the content and locate weights
model_data = torch.load(file_path, map_location='cpu')

# Step 4: Access the state_dict and confirm data start
state_dict = model_data['model'].state_dict()
print("State dict keys (weights):", state_dict.keys())

# Step 5: Locate a specific tensor's data in memory
tensor_key = next(iter(state_dict.keys()))  # Get the first tensor
print(f"First tensor: {tensor_key}, shape: {state_dict[tensor_key].shape}")

# Access the tensor data
tensor_data = state_dict[tensor_key].numpy()
print(f"First tensor raw data (first 10 values): {tensor_data.flatten()[:10]}")
