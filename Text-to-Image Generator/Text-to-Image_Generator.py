from PIL import Image, ImageDraw, ImageFont

# Define the input text and output path
text = "Hello, World!"  # Text to display on the image
output_path = "text_image.jpg"  # Path to save the generated image

# Create a blank white image
image = Image.new('RGB', (500, 200), color='white')

# Initialize ImageDraw object
draw = ImageDraw.Draw(image)

# Choose a font and size
font = ImageFont.load_default()

# Define the position to start the text
text_position = (50, 80)

# Define the text color
text_color = (0, 0, 0)  # Black text

# Add text to the image
draw.text(text_position, text, font=font, fill=text_color)

# Save the generated image
image.save(output_path)

print(f"Text image generated successfully. Result saved at: {output_path}")
