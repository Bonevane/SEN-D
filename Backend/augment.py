import os
import glob
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F

# Where your original dataset is
INPUT_DATASET = 'Dataset/Train'  # e.g., Dataset/Train or Dataset/Test
# Where to save the augmented dataset
OUTPUT_DATASET = 'Dataset_Augmented/Train'

IMG_SIZE = 299  # match InceptionV3 input size

# Define individual deterministic transformations
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
    angle = random.uniform(-15, 15)  # Random angle between -15 and +15 degrees
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

    image_files = glob.glob(os.path.join(input_cls_dir, '*.png'))  # assuming PNG
    print(f"[{cls}] Found {len(image_files)} original images.")
    
    for img_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        image = Image.open(img_path).convert('RGB')

        # Resize first
        resized = resize_transform(image)
        
        # Save original resized version
        resized.save(os.path.join(output_cls_dir, f"{base_name}_orig.png"))

        # Apply each individual transformation and save separately
        for transform_name, transform_fn in individual_transforms.items():
            try:
                transformed = transform_fn(resized)
                output_path = os.path.join(output_cls_dir, f"{base_name}_{transform_name}.png")
                transformed.save(output_path)
            except Exception as e:
                print(f"Warning: Failed to apply {transform_name} to {base_name}: {e}")

print(f"\n✅ Done. Augmented dataset stored at: {OUTPUT_DATASET}")
