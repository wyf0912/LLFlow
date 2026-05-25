import os
import random
from PIL import Image, ImageOps
from tqdm import tqdm

def generate_high_quality_crops(input_folder, output_folder, target_w=600, target_h=400, crops_per_image=3):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    
    for filename in tqdm(files):
        img_path = os.path.join(input_folder, filename)
        try:
            with Image.open(img_path) as img:
                img = ImageOps.exif_transpose(img)
                
                # --- QUALITY STEP: DOWNSAMPLE FIRST ---
                # Reducing by half (0.5) usually removes raw sensor noise
                scale_factor = 0.5 
                new_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                img_w, img_h = img.size
                if img_w < target_w or img_h < target_h:
                    continue

                for i in range(crops_per_image):
                    x1 = random.randint(0, img_w - target_w)
                    y1 = random.randint(0, img_h - target_h)
                    
                    box = (x1, y1, x1 + target_w, y1 + target_h)
                    cropped_img = img.crop(box)
                    
                    save_name = f"{os.path.splitext(filename)[0]}_hq_crop_{i}.png"
                    # Saving as PNG is vital for LLFlow to avoid JPEG artifacts
                    cropped_img.save(os.path.join(output_folder, save_name))
                    
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    generate_high_quality_crops("D:\\codes\\projects\\rm\\LLFlow\\code\\using\\using_crops\\only_low_not_love", "D:\\codes\\projects\\rm\\LLFlow\\code\\using\\using_crops\\only_low_not_love_crops")