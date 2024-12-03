import os

def calculate_directory_size(directory):
    """
    Calculates the total size of the given directory.
    Args:
        directory (str): Target directory path.
    Returns:
        dict: Total size in bytes, KB, and MB.
    """
    total_size = 0

    for dirpath, dirnames, filenames in os.walk(directory):
        for file in filenames:
            file_path = os.path.join(dirpath, file)
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)

    return {
        "bytes": total_size,
        "kilobytes": total_size / 1024,
        "megabytes": total_size / (1024 ** 2)
    }

# Example usage:
# size = calculate_directory_size("directory_path")
# print(size)
