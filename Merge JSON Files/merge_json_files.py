import os
import json

def merge_json_files(directory, output_file):
    """
    Merges all JSON files in the specified directory into a single JSON file.
    Args:
        directory (str): Target directory path containing JSON files.
        output_file (str): Path to save the merged JSON file.
    """
    merged_data = []

    for file in os.listdir(directory):
        if file.endswith(".json"):
            file_path = os.path.join(directory, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    merged_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error loading JSON: {file} - {e}")

    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(merged_data, outfile, indent=4, ensure_ascii=False)
        print(f"JSON files merged into '{output_file}'.")

# Example usage:
# merge_json_files("json_directory", "output.json")
