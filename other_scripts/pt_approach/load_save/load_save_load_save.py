from ultralytics import YOLO
import os
import hashlib

def get_file_hash(filepath):
    """Returns the MD5 hash of a file for comparison."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def get_file_size_in_bits(filepath):
    """Returns the file size in bits."""
    return os.path.getsize(filepath) * 8

# Step 1: Load the original YOLO model
original_model_path = 'yolov8n.pt'  # Replace with your model file
model = YOLO(original_model_path)

# Step 2: Save the model to a new file
saved_model_path1 = 'saved_model1.pt'
model.save(saved_model_path1)

# Step 3: Reload the saved model
reloaded_model = YOLO(saved_model_path1)

# Step 4: Save the reloaded model to another file
saved_model_path2 = 'saved_model2.pt'
reloaded_model.save(saved_model_path2)

# Step 5: Compare the three files
original_hash = get_file_hash(original_model_path)
saved_hash1 = get_file_hash(saved_model_path1)
saved_hash2 = get_file_hash(saved_model_path2)

original_size_bits = get_file_size_in_bits(original_model_path)
saved_size_bits1 = get_file_size_in_bits(saved_model_path1)
saved_size_bits2 = get_file_size_in_bits(saved_model_path2)

# Display results
print(f"Original Model: Hash={original_hash}, Size={original_size_bits} bits")
print(f"First Saved Model: Hash={saved_hash1}, Size={saved_size_bits1} bits")
print(f"Second Saved Model: Hash={saved_hash2}, Size={saved_size_bits2} bits")

if original_hash == saved_hash1 == saved_hash2 and original_size_bits == saved_size_bits1 == saved_size_bits2:
    print("All files are identical.")
else:
    print("The files differ.")
