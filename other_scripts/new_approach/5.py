from ultralytics import YOLO
import torch

# Load the YOLO model
model = YOLO('yolov8n.pt')  # Change to the correct path to your model file.

# Function to find and zero out the weights of the first Conv2d layer
def zero_first_conv_weights(model):
    for name, module in model.model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(f"First Conv2d Layer: {name}")
            print("Original weights snapshot (first few elements):", module.weight.data[0][0][:2][:2])

            # Zero out the weights
            module.weight.data.zero_()

            print("Updated weights snapshot (first few elements):", module.weight.data[0][0][:2][:2])
            break  # Stop after updating the first Conv2d layer

zero_first_conv_weights(model)


def print_first_conv(model):
    for name, module in model.model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(f"First Conv2d Layer: {name}")
            print(module)
            print(module.parameters())
            for x in module.parameters():
                print(x)
            break  # Stop after finding the first Conv2d layer

print_first_conv(model)
