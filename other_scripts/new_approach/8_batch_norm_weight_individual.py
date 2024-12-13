from ultralytics import YOLO
import torch
import os

# Directory to save inference results
output_directory = "modified_results"
os.makedirs(output_directory, exist_ok=True)

# Ask the user for the modification choice
choice = input("Do you want to run a for loop from 0 to 16 (enter 'loop') or just test 1 weight (enter 'single')? ")
indices = list(range(17)) if choice.lower() == 'loop' else [0]

# Specify the new value for the weights
new_value = float(input("Enter the new value for the BatchNorm weight(s): "))

# Image path for testing
image_path = 'saint_jean_de_luz.jpg'  # Replace with your test image path

# Function to modify the model and perform inference
def modify_and_infer(index, value):
    # Load a fresh model each time
    model = YOLO('yolov8n.pt')
    pytorch_model = model.model
    modified = False

    for layer in pytorch_model.modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            if index < layer.weight.data.size(0):
                with torch.no_grad():
                    layer.weight[index] = value
                    modified = True
            break
    
    if modified:
        print(f"Modified BatchNorm weight at index {index} to {value}.")
        # Run inference with the modified model
        result = model.predict(source=image_path, save=True, save_txt=False, project=output_directory, name=f'modified_index_{index}')
        print(f"Inference run for modification at index {index}.")
    else:
        print(f"No modification needed for index {index} (out of bounds).")

# Normal model for comparison
print("Running inference with the normal model...")
normal_model = YOLO('yolov8n.pt')
results_normal = normal_model.predict(source=image_path, save=True, save_txt=False, project=output_directory, name='normal')

# Modify and test each specified BatchNorm weight
for index in indices:
    modify_and_infer(index, new_value)

print("\nAll modifications and inferences completed. Results are saved in:", output_directory)
