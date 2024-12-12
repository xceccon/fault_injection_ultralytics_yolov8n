import torch
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8n.pt')  # Use the desired model variant

# Define a hook function to capture tensor outputs
def hook_fn(module, input, output):
    print(f"Hooked Layer: {module}")
    print(f"Input Tensor Shape: {input[0].shape}")
    print(f"Output Tensor Shape: {output.shape}")

# Register hooks to the first layer of the model
first_layer = list(model.model.children())[0]
hook_handle = first_layer.register_forward_hook(hook_fn)

# Prepare a sample input tensor (simulating an image)
# Adjust the size to match the input size of the model (e.g., 640x640)
sample_input = torch.randn(1, 3, 640, 640)  # Batch size 1, 3 color channels, 640x640 resolution

# Perform a forward pass through the model
model(sample_input)

# Remove the hook to avoid side effects
hook_handle.remove()
