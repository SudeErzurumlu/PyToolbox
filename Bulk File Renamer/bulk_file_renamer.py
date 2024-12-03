import os

def bulk_rename_files(directory, prefix):
    """
    Renames all files in the given directory by adding a specified prefix.
    Args:
        directory (str): Target directory path.
        prefix (str): Prefix to add to each file name.
    """
    if not os.path.exists(directory):
        print("Please provide a valid directory path.")
        return

    files = os.listdir(directory)
    for index, file in enumerate(files):
        old_path = os.path.join(directory, file)
        if os.path.isfile(old_path):
            file_extension = os.path.splitext(file)[1]
            new_name = f"{prefix}_{index + 1}{file_extension}"
            new_path = os.path.join(directory, new_name)
            os.rename(old_path, new_path)
            print(f"'{file}' -> '{new_name}' renamed.")

# Example usage:
# bulk_rename_files("directory_path", "new_prefix")
