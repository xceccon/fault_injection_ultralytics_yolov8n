import torch

def strip_weights(original_model_path, output_weights_path):
    """
    Strips the dictionary structure from the YOLO model file
    and saves only the raw weights to a new file.

    Args:
        original_model_path (str): Path to the original YOLO model file.
        output_weights_path (str): Path to save the stripped weights file.
    """
    # Load the original yolov8n.pt file
    model_data = torch.load(original_model_path)

    # Check if the file is a dictionary and extract the state_dict (weights)
    if isinstance(model_data, dict) and 'model' in model_data:
        # Extract only the model weights (state_dict)
        stripped_weights = model_data['model']

        # Save the stripped weights directly to a new file
        torch.save(stripped_weights, output_weights_path)
        print(f"Stripped weights saved to {output_weights_path}")
    else:
        print("The loaded model file does not have the expected structure.")

def compare_files_in_chunks(original_model_path, output_weights_path):
    """
    Compares the binary data of the original model file and stripped weights file
    in 10% chunks of the larger file.

    Args:
        original_model_path (str): Path to the original YOLO model file.
        output_weights_path (str): Path to the stripped weights file.

    Outputs:
        Statistics on similarity in 10% chunks.
    """
    # Read both files as binary data
    with open(original_model_path, 'rb') as f1, open(output_weights_path, 'rb') as f2:
        original_data = f1.read()
        stripped_data = f2.read()

    # Calculate total sizes in bits
    original_size_bits = len(original_data) * 8
    stripped_size_bits = len(stripped_data) * 8

    # Determine the larger file size in bits and convert to chunks
    max_size_bits = max(original_size_bits, stripped_size_bits)
    chunk_size_bits = max_size_bits // 10  # Size of 10% chunk in bits

    # Reverse the files for end-to-beginning comparison
    reversed_original = original_data[::-1]
    reversed_stripped = stripped_data[::-1]

    # Initialize similarity stats
    similarity_results = []

    for chunk_idx in range(10):
        # Define the bit range for this chunk
        start_bit = chunk_idx * chunk_size_bits
        end_bit = min((chunk_idx + 1) * chunk_size_bits, max_size_bits)

        # Extract bytes for this chunk from both files
        start_byte = start_bit // 8
        end_byte = (end_bit + 7) // 8  # Include incomplete byte
        chunk_original = reversed_original[start_byte:end_byte]
        chunk_stripped = reversed_stripped[start_byte:end_byte]

        # Bit-by-bit comparison
        matching_bits = 0
        total_bits_in_chunk = (end_bit - start_bit)
        for byte1, byte2 in zip(chunk_original, chunk_stripped):
            matching_bits += bin(byte1 ^ byte2).count('0')  # Count matching bits

        # Calculate similarity percentage for the chunk
        similarity_percentage = (matching_bits / total_bits_in_chunk) * 100
        similarity_results.append((chunk_idx, similarity_percentage))

    # Print results
    print("\n=== File Comparison Statistics by Chunks (10%) ===")
    print(f"Original File Size: {original_size_bits} bits")
    print(f"Stripped Weights File Size: {stripped_size_bits} bits")
    for chunk_idx, similarity in similarity_results:
        print(f"Chunk {90 - chunk_idx*10}% to {100 - chunk_idx*10}%: Similarity = {similarity:.2f}%")

def main():
    # Define paths
    original_model_path = "yolov8n.pt"  # Path to the original YOLO model file
    output_weights_path = "onlyweights.pt"  # Path to save the stripped weights file

    # Call the strip_weights function
    strip_weights(original_model_path, output_weights_path)

    # Compare files and output statistics in 10% chunks
    compare_files_in_chunks(original_model_path, output_weights_path)

if __name__ == "__main__":
    main()
