import torch
from ultralytics import YOLO

# Load the smallest YOLOv8 model from Ultralytics
model = YOLO('yolov8n.pt')  # 'yolov8n' refers to the "nano" version of YOLOv8

# Save the entire model to a file
full_model_path = 'yolov8n_full_model.pt'
torch.save(model, full_model_path)
print(f"Full model saved to {full_model_path}")

# Save only the state dictionary (weights)
weights_only_path = 'yolov8n_weights.pt'
torch.save(model.model.state_dict(), weights_only_path)
print(f"Weights-only file saved to {weights_only_path}")

# To ensure it worked, you can reload the weights as follows:
# Initialize a new YOLOv8 model and load the state dictionary
new_model = YOLO('yolov8n.pt')  # Initialize the same architecture
new_model.model.load_state_dict(torch.load(weights_only_path))
print("Weights successfully loaded into a new model instance.")


original_model = torch.load('yolov8n.pt')
full_model = torch.load('yolov8n_full_model.pt')
weights_only = torch.load('yolov8n_weights.pt')

print("Original Model Keys:", original_model.keys() if isinstance(original_model, dict) else type(original_model))
print("Full Model Keys:", dir(full_model))
print("Weights Only Keys:", weights_only.keys())
