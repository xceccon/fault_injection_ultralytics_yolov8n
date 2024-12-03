from ultralytics import YOLO
import os

# Path to the YOLO model (the bit-flipped model)
model_path = "test_models/model_bit_4889.pt"  # Path to your flipped model
model = YOLO(model_path)
print("Model loaded successfully.")

# Define a test dataset of 100 images (you can use local paths or URLs)
test_images = [
    "test_images/consti.jpg",
    "test_images/zidane.jpg",
    "test_images/bus.jpg",
    "test_images/gernika.jpg",
]

# Run inference on the test images
print("Running inference on images...")
for idx, image in enumerate(test_images):
    # Run inference and save results in the default 'runs' folder
    results = model(image, save=True)  # No need to specify save_dir; YOLO will use the default location
    print(f"Image {idx + 1}/{len(test_images)} processed: {image}")

print("Inference completed. Results saved to the default 'runs' folder.")
