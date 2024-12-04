import os
import shutil

def organize_files_by_extension(directory):
    """
    Organizes files in a directory into subfolders based on their extensions.
    Args:
        directory (str): Target directory.
    """
    if not os.path.exists(directory):
        print("Please provide a valid directory path.")
        return

    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            file_extension = os.path.splitext(file)[1][1:]  # Get file extension (e.g., 'txt')
            if not file_extension:
                file_extension = "unknown"
            target_dir = os.path.join(directory, file_extension)
            os.makedirs(target_dir, exist_ok=True)
            shutil.move(file_path, os.path.join(target_dir, file))
            print(f"'{file}' -> Moved to '{file_extension}' folder.")

# Example usage:
# organize_files_by_extension("directory_path")
