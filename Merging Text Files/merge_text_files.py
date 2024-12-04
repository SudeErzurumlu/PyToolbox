import os

def merge_text_files(directory, output_file):
    """
    Merges all text files in a directory into a single file.
    Args:
        directory (str): Directory containing text files.
        output_file (str): Name of the merged output file.
    """
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file in os.listdir(directory):
            if file.endswith(".txt"):
                file_path = os.path.join(directory, file)
                with open(file_path, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
                    outfile.write("\n")
                print(f"'{file}' merged.")
    print(f"Merge completed! Output file: {output_file}")

# Example usage:
# merge_text_files("text_directory", "output.txt")
