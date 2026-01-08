from PIL import Image, ImageEnhance
import os
from pathlib import Path
from PIL.Image import Exif


# Define paths
# Replace with your source images directory
SOURCE_DIR = "D:/codes/projects/rm/low-light-image-enhancement/data/rm_dataset_crops/val"
# Replace with your ground truth images directory
GROUND_TRUTH_DIR = "D:/codes/projects/rm/low-light-image-enhancement/data/rm_dataset_crops/val/high"
# Replace with your desired low-light images directory
LOW_LIGHT_DIR = "D:/codes/projects/rm/low-light-image-enhancement/data/rm_dataset_crops/val/low"

# Create the low-light directory if it doesn't exist
Path(LOW_LIGHT_DIR).mkdir(parents=True, exist_ok=True)
Path(GROUND_TRUTH_DIR).mkdir(parents=True, exist_ok=True)

# Brightness reduction factor
brightness_factor = 0.15


def apply_exif_rotation(img):
    """Apply EXIF rotation to image if present"""
    try:
        exif = img._getexif()
        if exif is not None:
            for tag, value in exif.items():
                if tag == 274:  # Orientation tag
                    if value == 3:
                        img = img.rotate(180, expand=True)
                    elif value == 6:
                        img = img.rotate(270, expand=True)
                    elif value == 8:
                        img = img.rotate(90, expand=True)
    except:
        pass
    return img


# Process each image in the ground truth directory
for filename in os.listdir(SOURCE_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
        # Open the ground truth image
        img_path = os.path.join(SOURCE_DIR, filename)
        img = Image.open(img_path)

        # apply EXIF rotation if needed
        img = apply_exif_rotation(img)

        # Reduce brightness to create a low-light image
        enhancer = ImageEnhance.Brightness(img)
        low_light_img = enhancer.enhance(brightness_factor)

        # save the ground truth image to the high directory
        ground_truth_img_path = os.path.join(GROUND_TRUTH_DIR, filename)
        img.save(ground_truth_img_path)

        # Save the low-light image
        low_light_img_path = os.path.join(LOW_LIGHT_DIR, filename)
        low_light_img.save(low_light_img_path)

        print(
            f"Processed {filename}: Low-light image saved to {low_light_img_path}")
