from ultralytics import YOLO
import torch
import os

# Load YOLO model
model = YOLO('yolov8n.pt')

# Check the keys in the original model's state_dict
state_dict = model.model.state_dict()
print("State Dict Keys:", state_dict.keys())

# Save the state dict
torch.save(state_dict, 'state_dict_only.pt')

# Compare sizes
original_size = os.path.getsize('yolov8n.pt') * 8  # in bits
state_dict_size = os.path.getsize('state_dict_only.pt') * 8  # in bits

print(f"Original Model Size: {original_size} bits")
print(f"State Dict Size: {state_dict_size} bits")
