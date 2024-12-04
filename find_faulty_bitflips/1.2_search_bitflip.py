import os
import json
from ultralytics import YOLO

# Paths and configurations
original_model_path = "yolov8n.pt"  # Path to the model
output_dir = "bitflip_tests"
os.makedirs(output_dir, exist_ok=True)

byte_offset = 0x3000  # Starting byte offset
num_bits = 12800  # Total bits to flip (e.g., 12800 for testing across bytes)

# Output JSON file
results_json_path = os.path.join(output_dir, "bitflip_results.json")

# Load existing results if they exist
if os.path.exists(results_json_path):
    with open(results_json_path, "r") as json_file:
        results = json.load(json_file)
        completed_bits = {entry["global_bit_index"] for entry in results}
else:
    results = []
    completed_bits = set()

# Test image
test_image = "https://ultralytics.com/images/zidane.jpg"  # Single test image

# Helper function to flip a bit
def flip_bit(byte, bit_index):
    return byte ^ (1 << bit_index)

# Helper function to test a single bit flip
def test_bit(global_bit_index):
    # Calculate byte and bit position
    byte_index = byte_offset + global_bit_index // 8
    bit_index = global_bit_index % 8

    # Read the original model
    with open(original_model_path, "rb") as f:
        data = bytearray(f.read())

    # Flip the specific bit
    data[byte_index] = flip_bit(data[byte_index], bit_index)

    # Save the modified model
    flipped_model_path = os.path.join(output_dir, f"model_bit_{global_bit_index}.pt")
    with open(flipped_model_path, "wb") as f:
        f.write(data)

    result_entry = {
        "global_bit_index": global_bit_index,
        "byte_index": byte_index,
        "bit_index": bit_index,
        "model_loaded": False,
        "detections": []
    }

    try:
        # Load the modified model
        modified_model = YOLO(flipped_model_path)
        result_entry["model_loaded"] = True

        # Run inference
        inference_results = modified_model(test_image, save=False)  # Run inference on a single image

        # Collect detections
        for result in inference_results:
            detected_objects = {}
            for box in result.boxes:
                class_id = int(box.cls)  # Class ID
                class_name = result.names[class_id]
                detected_objects[class_name] = detected_objects.get(class_name, 0) + 1
            result_entry["detections"] = detected_objects

    except Exception as e:
        pass  # If the model fails to load, the result_entry will reflect it

    return result_entry


# Manually manage concurrency: process 3 tasks at a time
def process_in_batches(batch_size=3):
    current_batch = []
    for global_bit_index in range(num_bits):
        if global_bit_index not in completed_bits:
            current_batch.append(global_bit_index)

            if len(current_batch) == batch_size:
                process_batch(current_batch)
                current_batch = []  # Reset the batch

    # Process any remaining tasks
    if current_batch:
        process_batch(current_batch)


def process_batch(batch):
    for global_bit_index in batch:
        result = test_bit(global_bit_index)
        results.append(result)

        # Save intermediate results to JSON
        with open(results_json_path, "w") as json_file:
            json.dump(results, json_file, indent=4)


# Run the bit-flipping tests
process_in_batches(batch_size=3)

print("Bit-flipping tests completed. Results saved to:", results_json_path)
