from ultralytics import YOLO

# Step 1: Load the YOLO model
model = YOLO('yolov8n.pt')  # Replace 'yolov8n.pt' with your model path

# Step 2: Access the state_dict (weights)
state_dict = model.model.state_dict()

# Step 3: Print the keys in the state_dict
print("State Dict Keys:", state_dict.keys())

# Step 4: Save the state_dict for inspection
import torch
torch.save(state_dict, 'stripped_weights.pt')
print("Stripped weights saved to 'stripped_weights.pt'.")

# Optional: Print specific weight shapes for inspection
for key, value in state_dict.items():
    print(f"{key}: {value.shape}")
