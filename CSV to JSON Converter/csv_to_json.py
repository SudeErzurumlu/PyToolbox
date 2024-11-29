import csv
import json

def csv_to_json(input_csv_path, output_json_path):
    """
    Converts a CSV file to JSON format.
    
    Parameters:
        input_csv_path (str): Path to the input CSV file.
        output_json_path (str): Path to save the converted JSON file.
    """
    with open(input_csv_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        data = [row for row in csv_reader]
    
    with open(output_json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"CSV file converted to JSON successfully. Result saved at: {output_json_path}")

# Example usage
input_csv = "data.csv"  # User-provided CSV file path
output_json = "data.json"  # User-provided output path
csv_to_json(input_csv, output_json)
