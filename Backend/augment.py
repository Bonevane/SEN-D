import os
import glob
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F

INPUT_DATASET = 'Dataset/Train'  # Path to the original dataset
OUTPUT_DATASET = 'Dataset_Augmented/Train' # Path to store the augmented dataset

IMG_SIZE = 299  # Size to which images will be resized

# Individual transformations
def flip_vertical(img):
    return F.vflip(img)

def flip_horizontal(img):
    return F.hflip(img)

def height_shift(img):
    return F.affine(img, angle=0, translate=(0, int(0.1 * IMG_SIZE)), scale=1.0, shear=0)

def width_shift(img):
    return F.affine(img, angle=0, translate=(int(0.1 * IMG_SIZE), 0), scale=1.0, shear=0)

def rotation_range(img):
    import random
    angle = random.uniform(-15, 15)
    return F.rotate(img, angle=angle)

def shear_range(img):
    return F.affine(img, angle=0, translate=(0, 0), scale=1.0, shear=(10, 0))

def zoom_range(img):
    return F.affine(img, angle=0, translate=(0, 0), scale=1.2, shear=0)

# Dictionary of individual transforms
individual_transforms = {
    'vflip': flip_vertical,
    'hflip': flip_horizontal,
    'height_shift': height_shift,
    'width_shift': width_shift,
    'rotation': rotation_range,
    'shear': shear_range,
    'zoom': zoom_range,
}

# Resize transform
resize_transform = transforms.Resize((IMG_SIZE, IMG_SIZE))

# Make output dirs
os.makedirs(OUTPUT_DATASET, exist_ok=True)

classes = os.listdir(INPUT_DATASET)
for cls in classes:
    input_cls_dir = os.path.join(INPUT_DATASET, cls)
    output_cls_dir = os.path.join(OUTPUT_DATASET, cls)
    os.makedirs(output_cls_dir, exist_ok=True)

    # Get image files
    png_files = glob.glob(os.path.join(input_cls_dir, '*.png'))
    jpg_files = glob.glob(os.path.join(input_cls_dir, '*.jpg'))
    jpeg_files = glob.glob(os.path.join(input_cls_dir, '*.jpeg'))
    
    # Combine all image files
    image_files = png_files + jpg_files + jpeg_files
    print(f"[{cls}] Found {len(image_files)} original images (PNG: {len(png_files)}, JPG: {len(jpg_files)}, JPEG: {len(jpeg_files)}).")
    
    for img_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        image = Image.open(img_path).convert('RGB')

        # Resizing and saving original image (using PNG for consistency)
        resized = resize_transform(image)
        resized.save(os.path.join(output_cls_dir, f"{base_name}_orig.png"))

        # Applying each individual transformation and saving
        for transform_name, transform_fn in individual_transforms.items():
            try:
                transformed = transform_fn(resized)
                output_path = os.path.join(output_cls_dir, f"{base_name}_{transform_name}.png")
                transformed.save(output_path)
            except Exception as e:
                print(f"Warning: Failed to apply {transform_name} to {base_name}: {e}")

print(f"\nDone! Augmented dataset stored at: {OUTPUT_DATASET}")
