from ultralytics import YOLO
import os
import torch

def remove_metadata_and_save(input_path, output_path):
    """
    Load a YOLO model, remove metadata or dictionary structures, 
    and save the stripped-down model to a new file.
    """
    # Step 1: Load the YOLO model
    model = YOLO(input_path)
    
    # Step 2: Extract the PyTorch model state dictionary
    stripped_state_dict = model.model.state_dict()

    # Step 3: Save the stripped-down model state dictionary
    torch.save(stripped_state_dict, output_path)
    print(f"Stripped model saved to: {output_path}")

    # Display file sizes for comparison
    original_size = os.path.getsize(input_path) * 8  # Size in bits
    stripped_size = os.path.getsize(output_path) * 8  # Size in bits
    print(f"Original model size: {original_size} bits")
    print(f"Stripped model size: {stripped_size} bits")


# File paths
original_model_path = 'yolov8n.pt'  # Replace with your YOLO model file
stripped_model_path = 'yolov8n_stripped.pt'

# Run the function
remove_metadata_and_save(original_model_path, stripped_model_path)
