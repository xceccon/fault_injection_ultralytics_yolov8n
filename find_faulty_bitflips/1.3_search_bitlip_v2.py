import os
import json
from ultralytics import YOLO
import signal
import sys

# Paths and configurations
original_model_path = "yolov8n.pt"  # Path to the original model
output_dir = "bitflip_tests"
os.makedirs(output_dir, exist_ok=True)

# Output JSON file for results
results_json_path = os.path.join(output_dir, "bitflip_results.json")
# File to track the last processed bit
progress_file_path = os.path.join(output_dir, "progress.json")

# Load existing results if they exist
if os.path.exists(results_json_path):
    with open(results_json_path, "r") as json_file:
        results = json.load(json_file)
        completed_bits = {entry["bit_index"] for entry in results}
else:
    results = []
    completed_bits = set()

# Load last processed bit (if exists)
if os.path.exists(progress_file_path):
    with open(progress_file_path, "r") as progress_file:
        progress = json.load(progress_file)
        last_processed_bit = progress.get("last_processed_bit", -1)  # -1 means not started or reset
else:
    last_processed_bit = -1  # Initial state before any processing

# Test image
test_image = "https://ultralytics.com/images/zidane.jpg"  # Single test image

# Helper function to flip a bit
def flip_bit(byte, bit_index):
    return byte ^ (1 << bit_index)

# Helper function to test a single bit flip
def test_bit(bit_index, total_bits):
    # Check if the bit index is within bounds
    if bit_index < 0 or bit_index >= total_bits:
        raise ValueError(f"Bit index {bit_index} is out of bounds. Must be between 0 and {total_bits - 1}.")

    # Calculate byte and bit position
    byte_index = bit_index // 8
    bit_position = bit_index % 8

    # Read the original model
    with open(original_model_path, "rb") as f:
        data = bytearray(f.read())

    # Flip the specific bit
    data[byte_index] = flip_bit(data[byte_index], bit_position)

    # Save the modified model
    flipped_model_path = os.path.join(output_dir, f"flipped_bit_{bit_index}.pt")
    with open(flipped_model_path, "wb") as f:
        f.write(data)

    result_entry = {
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

    except Exception:
        pass  # If the model fails to load, the result_entry will reflect it

    return result_entry

# Automatically resume from the last processed bit
total_bits = os.path.getsize(original_model_path) * 8
starting_bit = last_processed_bit + 1  # Always resume from the next bit after the last processed bit

# Function to handle program interruption
def safe_stop(signum, frame):
    print("\nSafe stop: Saving progress and exiting...")
    # Save progress and results before exiting
    with open(progress_file_path, "w") as progress_file:
        json.dump({"last_processed_bit": starting_bit - 1}, progress_file)

    # Sort results by the bit_index to ensure order
    sorted_results = sorted(results, key=lambda x: x["bit_index"])
    with open(results_json_path, "w") as json_file:
        json.dump(sorted_results, json_file, indent=4)

    print("Progress saved. Exiting...")
    sys.exit(0)  # Exit the program cleanly

# Register signal handler for graceful exit on Ctrl+C (SIGINT)
signal.signal(signal.SIGINT, safe_stop)

# Process the bits starting from the chosen index
def process_bits(starting_bit):
    for bit_index in range(starting_bit, total_bits):
        if bit_index not in completed_bits:
            result = test_bit(bit_index, total_bits)
            results.append(result)

            # Update the progress file to track the last processed bit
            with open(progress_file_path, "w") as progress_file:
                json.dump({"last_processed_bit": bit_index}, progress_file)

            # Sort results by the bit_index to ensure order
            sorted_results = sorted(results, key=lambda x: x["bit_index"])

            # Save intermediate results to JSON, with sorted order
            with open(results_json_path, "w") as json_file:
                json.dump(sorted_results, json_file, indent=4)

process_bits(starting_bit)

print("Bit-flipping tests completed. Results saved to:", results_json_path)
