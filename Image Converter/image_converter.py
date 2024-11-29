from PIL import Image

def convert_image(input_path, output_path):
    """
    Converts an image from JPG to PNG format.
    
    Parameters:
        input_path (str): Path to the input JPG image.
        output_path (str): Path to save the converted PNG image.
    """
    # Open the input image
    image = Image.open(input_path)
    
    # Convert to PNG and save
    image.save(output_path, 'PNG')
    print(f"Image converted successfully. Result saved at: {output_path}")

# Example usage
input_image_path = "input_image.jpg"  # User-provided input path
output_image_path = "output_image.png"  # User-provided output path
convert_image(input_image_path, output_image_path)
