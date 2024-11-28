from PIL import Image

# Define the input and output paths
input_path = "input_image.jpg"  # Path to the input image
output_path = "resized_image.jpg"  # Path to save the resized image
new_width = 800  # New width for the image
new_height = 600  # New height for the image

# Open the input image
image = Image.open(input_path)

# Resize the image
resized_image = image.resize((new_width, new_height))

# Save the resized image
resized_image.save(output_path)

print(f"Image resized successfully. Result saved at: {output_path}")
