from ultralytics import YOLO
import torch
from shapely.geometry import box
import pandas as pd

# Load the YOLOv8 model
model_normal = YOLO('yolov8n.pt')  # Normal model for baseline

# Function to extract boxes from results
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

# Function to calculate IoU
def calculate_iou(box1, box2):
    b1 = box(*box1)
    b2 = box(*box2)
    return b1.intersection(b2).area / b1.union(b2).area

# Calculate Roto_Score
def calculate_roto_score(normal_boxes, modified_boxes):
    iou_differences = []
    class_mismatches = 0
    matched_boxes = 0

    for norm_box in normal_boxes:
        best_iou = 0
        best_mod_box = None

        for mod_box in modified_boxes:
            iou = calculate_iou(norm_box['bbox'], mod_box['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_mod_box = mod_box

        if best_iou > 0.5:  # Match threshold
            matched_boxes += 1
            iou_differences.append(1 - best_iou)  # Deviation from perfect IoU
            if norm_box['class'] != best_mod_box['class']:
                class_mismatches += 1

    iou_difference_avg = sum(iou_differences) / len(iou_differences) if iou_differences else 1
    accuracy = (1 - class_mismatches / matched_boxes) if matched_boxes > 0 else 0
    roto_score = (iou_difference_avg + (1 - accuracy)) * 100
    return min(roto_score, 100)  # Clamp to 100 for interpretation

# Iterate over kernels
image_path = 'saint_jean_de_luz.jpg'  # Replace with your image path
roto_scores = []

for kernel_idx in range(16):  # First 16 kernels
    # Reload the model for each kernel modification
    model_modified = YOLO('yolov8n.pt')
    pytorch_model = model_modified.model

    # Access the first convolutional layer
    first_conv = None
    for layer in pytorch_model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            first_conv = layer
            break

    if first_conv is not None:
        with torch.no_grad():
            # Zero out the specific kernel
            first_conv.weight[kernel_idx].zero_()

    # Perform inference with the modified model
    print(f"Running inference for kernel {kernel_idx}...")
    results_normal = model_normal.predict(source=image_path, save=False, save_txt=False)
    results_modified = model_modified.predict(source=image_path, save=False, save_txt=False)

    # Extract predictions
    normal_boxes = extract_boxes(results_normal)
    modified_boxes = extract_boxes(results_modified)

    # Calculate Roto_Score
    roto_score = calculate_roto_score(normal_boxes, modified_boxes)
    roto_scores.append({"Kernel": kernel_idx, "Roto_Score": roto_score})

    print(f"Kernel {kernel_idx}: Roto_Score = {roto_score:.2f}%")

# Save Roto_Scores to a CSV file and print the table
df_roto_scores = pd.DataFrame(roto_scores)
df_roto_scores.to_csv("roto_scores.csv", index=False)
print(df_roto_scores)
