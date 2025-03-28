# augment
import albumentations as A
import cv2
# Define the augmentation pipeline
transform_1 = A.Compose([
    A.HorizontalFlip(p=1),  # Horizontal flip with 50% probability
])
transform_2 = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,p=1),  # Random brightness and contrast adjustment
])
transform_3 = A.Compose([
    A.Rotate(limit=15, p=1),  # Rotate by a random degree up to 15 degrees
])
import os 
import uuid
dataset_folder = "dataset_balanced"
dest_dataset_folder = "dataset_balanced_augmented"
os.makedirs(dest_dataset_folder, exist_ok=True)

for foldername in os.listdir(dataset_folder):
    dest_specific_folder = os.path.join(dest_dataset_folder, foldername)
    os.makedirs(dest_specific_folder, exist_ok=True)
    print (dest_specific_folder)
    folderpath = os.path.join(dataset_folder, foldername)
    for filename in os.listdir(folderpath):
        try:
            image_path = os.path.join(folderpath, filename)
            image = cv2.imread(image_path)
            t = filename.split(".")[0] + ".jpg"
            cv2.imwrite(os.path.join(dest_specific_folder, t), image)
            transformed_image = transform_1(image=image)["image"]
            cv2.imwrite(os.path.join(dest_specific_folder, t.replace(".jpg", "_HF.jpg")), transformed_image)
            transformed_image = transform_2(image=image)["image"]
            cv2.imwrite(os.path.join(dest_specific_folder, t.replace(".jpg", "_RBC.jpg")), transformed_image)
            transformed_image = transform_3(image=image)["image"]
            cv2.imwrite(os.path.join(dest_specific_folder, t.replace(".jpg", "_RT.jpg")), transformed_image)
        except:
            continue