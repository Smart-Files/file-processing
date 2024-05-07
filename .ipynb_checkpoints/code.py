
import os
import zipfile
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Step 1: Install Required Libraries
# You can install the required libraries by running:
# pip install Pillow reportlab

# Step 2: Extract images from the zip file
zip_file_path = 'images.zip'
temp_dir = 'temp_images'

try:
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
except zipfile.BadZipFile as e:
    print(f"Error opening zip file: {e}")
    exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
    exit(1)

# Step 3: Define a function to load images
def load_images(image_dir):
    images = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                images.append(os.path.join(root, file))
    return images

# Step 4: Load the images
image_files = load_images(temp_dir)
if not image_files:
    print("No images found in the zip file.")
    exit(1)

# Step 5: Create a PDF with one image per page
pdf_file_path = 'combined_images.pdf'
try:
    c = canvas.Canvas(pdf_file_path, pagesize=letter)

    for image_file in image_files:
        image = Image.open(image_file)
        width, height = image.size
        max_width, max_height = letter  # Letter size

        # Calculate aspect ratio and resize image to fit the page
        aspect = height / float(width)
        new_width = max_width
        new_height = int(max_width * aspect)
        if new_height > max_height:
            new_height = max_height
            new_width = int(max_height / aspect)

        # Define the position for the image on the page
        x = (max_width - new_width) / 2.0
        y = (max_height - new_height) / 2.0

        # Draw the image onto the PDF page
        c.drawImage(image_file, x, y, width=new_width, height=new_height)
        c.showPage()

    # Step 6: Save the PDF
    c.save()
    print(f"The PDF file has been created successfully: {pdf_file_path}")

except Exception as e:
    print(f"An error occurred while creating the PDF: {e}")
    exit(1)

# Step 7: Clean up temporary files (optional)
finally:
    if os.path.exists(temp_dir):
        try:
            os.rmdir(temp_dir)
            print("Temporary files cleaned up.")
        except Exception as e:
            print(f"An error occurred while cleaning up temporary files: {e}")
hello