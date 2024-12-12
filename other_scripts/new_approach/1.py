from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8n.pt')  # Change to the desired model variant if needed

# Save model architecture layers to a file
output_file = 'model_layers.txt'
with open(output_file, 'w') as f:
    f.write("YOLO Model Architecture Layers:\n")
    for name, layer in model.model.named_modules():
        f.write(f"{name}: {layer}\n")

print(f"Model layers have been saved to '{output_file}'. Open the file to view the layers.")

print(model.model)