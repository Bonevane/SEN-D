import os
import glob
import random
from PIL import Image
from torchvision import transforms

# Where your original dataset is
INPUT_DATASET = 'Dataset/Train'  # e.g., Dataset/Train or Dataset/Test
# Where to save the augmented dataset
OUTPUT_DATASET = 'Dataset_Augmented/Train'

IMG_SIZE = 299  # match InceptionV3 input size
AUG_PER_IMAGE = 8  # number of new images per original

# Define augmentations to match your notebook training transforms
augmentation = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize first
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),  # Width/height shift + zoom + shear
    transforms.ColorJitter(brightness=0.2),
])

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

        # Save original resized version
        resized = transforms.Resize((IMG_SIZE, IMG_SIZE))(image)
        resized.save(os.path.join(output_cls_dir, f"{base_name}_orig.png"))

        # Generate N augmentations
        for i in range(AUG_PER_IMAGE):
            aug_img = augmentation(image)
            aug_img.save(os.path.join(output_cls_dir, f"{base_name}_aug{i}.png"))

print(f"\n✅ Done. Augmented dataset stored at: {OUTPUT_DATASET}")
