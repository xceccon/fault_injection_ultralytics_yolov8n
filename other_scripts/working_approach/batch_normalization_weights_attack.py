from ultralytics import YOLO
import torch
import os

# Values to inject into the first BN weight
test_values = [-1000000, -1000, -100, -50, -10, -2, -1, 0, 1, 2, 10, 50, 100, 1000, 1000000]

# Image path for testing
image_path = 'saint_jean_de_luz.jpg'  # Ensure this path is correct
if not os.path.exists(image_path):
    raise FileNotFoundError(f"No image found at {image_path}")

# Function to modify the model and perform inference
def modify_and_infer(value):
    # Load a fresh model each time
    model = YOLO('yolov8n.pt')  # Ensure this file path is correct
    bn_found = False

    # Modify the first BatchNorm weight
    for layer in model.model.modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            layer.weight.data[1] = value
            bn_found = True
            break

    if bn_found:
        print(f"Modified first BatchNorm weight to {value}.")
        # Run inference with the modified model
        results = model.predict(source=image_path, save=True)
        print(f"Inference run for weight value {value}. Results automatically saved in the default directory.")
    else:
        print("BatchNorm layer not found.")

# Run inference with each value
for value in test_values:
    modify_and_infer(value)

print("All modifications and inferences completed. Check the default Ultralytics runs directory for results.")
