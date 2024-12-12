import json

def analyze_bitflip_results(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    total_entries = len(data)
    models_loaded = 0
    corrupted_models = 0
    unexpected_detections = []

    for entry in data:
        if entry.get("model_loaded"):
            models_loaded += 1

            detections = entry.get("detections", {})
            if not detections:
                continue

            if not isinstance(detections, dict):
                print(f"Unexpected detections format at entry {entry['bit_index']}: {detections}")
                continue

            person_count = detections.get("person", 0)
            tie_count = detections.get("tie", 0)

            if person_count != 2 or tie_count != 1 or len(detections) > 2:
                unexpected_detections.append({
                    "bit_index": entry["bit_index"],
                    "detections": detections
                })
        else:
            corrupted_models += 1

    # Summary statistics
    stats = {
        "total_entries": total_entries,
        "models_loaded": models_loaded,
        "corrupted_models": corrupted_models,
        "unexpected_detection_cases": len(unexpected_detections)
    }

    return stats, unexpected_detections

# Specify the path to your JSON file
file_path = "bitflip_tests/bitflip_results.json"

# Run analysis
stats, unexpected_detections = analyze_bitflip_results(file_path)

# Print summary statistics
print("Statistics:")
for key, value in stats.items():
    print(f"{key}: {value}")

# Print unexpected detection cases
if unexpected_detections:
    print("\nUnexpected Detection Cases:")
    for case in unexpected_detections:
        print(case)