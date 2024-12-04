def analyze_log_file(log_file, error_output_file):
    """
    Analyzes a log file to extract error messages and writes them to a separate file.
    Args:
        log_file (str): Path to the log file.
        error_output_file (str): Name of the file to store error messages.
    """
    with open(log_file, 'r', encoding='utf-8') as infile, \
         open(error_output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if "ERROR" in line or "Error" in line:
                outfile.write(line)
        print(f"Error messages have been written to '{error_output_file}'.")

# Example usage:
# analyze_log_file("server.log", "errors.log")
