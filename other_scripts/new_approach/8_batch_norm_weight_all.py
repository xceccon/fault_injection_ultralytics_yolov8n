from ultralytics import YOLO
import torch
import pandas as pd

# Load the original YOLO model
model_normal = YOLO('yolov8n.pt')  # Baseline model
model_modified = YOLO('yolov8n.pt')  # Model for modification

# Access the BatchNorm layer in the first Conv block
def modify_batchnorm_weights(model):
    pytorch_model = model.model

    for layer in pytorch_model.modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            with torch.no_grad():
                # Modify BatchNorm weights (gamma) to all ones
                layer.weight.fill_(10000000000000)  # Set gamma to 2 for all channels
            print("Modified BatchNorm weights (gamma).")
            break  # Exit after modifying the first BatchNorm layer

# Modify the BatchNorm weights in the modified model
modify_batchnorm_weights(model_modified)

# Perform inference on the same image with both models
image_path = 'saint_jean_de_luz.jpg'  # Replace with your test image path

print("Running inference with the normal model...")
results_normal = model_normal.predict(source=image_path, save=False, save_txt=False)

print("Running inference with the modified model...")
results_modified = model_modified.predict(source=image_path, save=True, save_txt=True)  # Save modified results

# Extract predictions for comparison
def extract_boxes(results):
    boxes = []
    for r in results:
        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            boxes.append({
                "bbox": box.tolist(),  # [x1, y1, x2, y2]
                "confidence": float(conf),
                "class": int(cls)
            })
    return boxes

normal_boxes = extract_boxes(results_normal)
modified_boxes = extract_boxes(results_modified)

# Adjust lengths of normal_boxes and modified_boxes
max_len = max(len(normal_boxes), len(modified_boxes))

# Fill missing entries with None to make lists equal in length
normal_boxes.extend([{"bbox": None, "confidence": None, "class": None}] * (max_len - len(normal_boxes)))
modified_boxes.extend([{"bbox": None, "confidence": None, "class": None}] * (max_len - len(modified_boxes)))

# Create DataFrame
comparison_df = pd.DataFrame({
    "Normal_BBox": [nb['bbox'] for nb in normal_boxes],
    "Normal_Conf": [nb['confidence'] for nb in normal_boxes],
    "Normal_Class": [nb['class'] for nb in normal_boxes],
    "Modified_BBox": [mb['bbox'] for mb in modified_boxes],
    "Modified_Conf": [mb['confidence'] for mb in modified_boxes],
    "Modified_Class": [mb['class'] for mb in modified_boxes],
})

# Save comparison to CSV
comparison_df.to_csv("output_comparison.csv", index=False)
print("\nComparison saved to 'output_comparison.csv'")

# Save modified model's results for further analysis
modified_results_path = "modified_results/"
print(f"Modified model's inference results saved to: {modified_results_path}")
