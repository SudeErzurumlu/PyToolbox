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
