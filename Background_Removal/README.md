# Background Removal Script

This repository contains a Python script that efficiently removes the background from an image. Using the `rembg` library, it processes an input image and generates an output file with the background removed.

## Features

- **Efficient Background Removal:** Powered by the `rembg` library, this script processes images quickly and accurately.
- **Customizable Paths:** Easily specify input and output file locations.
- **Lightweight and Simple:** Designed for straightforward usage with minimal setup.

---

## Prerequisites

Ensure you have Python installed on your system. Then, install the following required libraries:

```bash
pip install rembg
pip install pillow
```

---

## Usage

1. **Prepare Your Input Image**  
   Save your input image in the same directory as the script or specify its full path in the `input_path` variable.

2. **Edit the Script**  
   Update the paths in the script to match your input and desired output locations:
   ```python
   input_path = "input.jpg"  # Path to the input image
   output_path = "output.jpg"  # Path to save the output image
   ```

3. **Run the Script**  
   Execute the script using Python:
   ```bash
   python background_removal.py
   ```

4. **Output**  
   The processed image will be saved at the location specified in the `output_path`.

---

## Example Script

Below is the script provided in this repository:

```python
# Import necessary libraries
from rembg import remove  # For background removal
from PIL import Image  # For image handling
import io

# Define the input and output paths
input_path = "input.jpg"  # Path to the input image
output_path = "output.jpg"  # Path to save the output image

# Open the input image as binary data
with open(input_path, "rb") as input_file:
    input_data = input_file.read()

# Process the image to remove the background
output_data = remove(input_data)

# Save the processed image to the specified output path
with open(output_path, "wb") as output_file:
    output_file.write(output_data)

print(f"Background removed successfully. Result saved at: {output_path}")
```

---

## License

This project is licensed under the MIT License. Feel free to use and modify the script as needed.
