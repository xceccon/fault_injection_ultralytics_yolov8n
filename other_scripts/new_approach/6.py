from ultralytics import YOLO
import torch

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Access the PyTorch model backbone
pytorch_model = model.model

# Access the first convolutional layer (assuming it's the first in the model)
first_conv = None
for layer in pytorch_model.modules():
    if isinstance(layer, torch.nn.Conv2d):
        first_conv = layer
        break

if first_conv is not None:
    # Set the weights of the first convolutional layer to zeros
    with torch.no_grad():
        first_conv.weight.fill_(0)
        if first_conv.bias is not None:
            first_conv.bias.fill_(0)
    print("First convolutional layer weights and biases set to zero.")
else:
    print("No convolutional layer found.")

# Save the model if needed
torch.save(pytorch_model.state_dict(), 'modified_yolov8n.pth')

# Perform inference (if you still want to test the modified model)
results = model(source='saint_jean_de_luz.jpg', save=True)
