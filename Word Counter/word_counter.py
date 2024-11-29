def count_words_in_file(file_path):
    """
    Counts the number of words in a text file.
    
    Parameters:
        file_path (str): Path to the text file.
    
    Returns:
        int: Word count in the file.
    """
    with open(file_path, 'r') as file:
        text = file.read()
        words = text.split()
        word_count = len(words)
    return word_count

# Example usage
input_file_path = "example.txt"  # User-provided file path
word_count = count_words_in_file(input_file_path)
print(f"The file contains {word_count} words.")
