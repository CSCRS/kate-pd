import os
import cv2
import numpy as np
import argparse
from datasets import Dataset, DatasetDict, Image, Value

from huggingface_hub import HfApi, HfFolder, login

import glob

def create_mask_from_polygon(label_path, image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found or unreadable: {image_path}")
        return None
    image_height, image_width = image.shape[:2]

    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7 or len(parts) % 2 == 0:
            continue

        coords = list(map(float, parts[1:]))
        points = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * image_width)
            y = int(coords[i+1] * image_height)
            points.append([x, y])

        pts = np.array([points], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    return mask

def process_yolo_folder(yolo_dir, output_dir, partition):
    label_dir = os.path.join(yolo_dir, "labels", partition)
    image_dir = os.path.join(yolo_dir, "images", partition)

    if not os.path.exists(label_dir):
        print(f"Label folder not found: {label_dir}")
        return
    if not os.path.exists(image_dir):
        print(f"Image folder not found: {image_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(label_dir):
        if not filename.endswith(".txt"):
            continue

        label_path = os.path.join(label_dir, filename)

        # Try multiple image extensions
        image_name_base = os.path.splitext(filename)[0]
        image_path = None
        for ext in [".jpg", ".png", ".jpeg"]:
            candidate = os.path.join(image_dir, image_name_base + ext)
            if os.path.exists(candidate):
                image_path = candidate
                break

        if image_path is None:
            print(f"No image found for {filename}")
            continue

        mask = create_mask_from_polygon(label_path, image_path)
        if mask is not None:
            out_filename = image_name_base + ".png"
            out_path = os.path.join(output_dir, out_filename)
            cv2.imwrite(out_path, mask)
            print(f"Saved: {out_path}")



def create_dataset(image_folder, label_folder, mask_folder):
        
    # Get sorted lists of file paths
    image_paths = sorted(glob.glob(f"{image_folder}/*.png"))
    label_paths = sorted(glob.glob(f"{label_folder}/*.txt"))
    mask_paths = sorted(glob.glob(f"{mask_folder}/*.png"))
   
    # Ensure all lists have the same length and filenames match
    assert len(image_paths) == len(label_paths) == len(mask_paths)

    label_texts = []
    for path in label_paths:
        with open(path, 'r') as f:
            label_texts.append(f.read())
        
    # Create a Dataset
    dataset = Dataset.from_dict({
        "image": image_paths,
        "label": label_texts,
        "mask": mask_paths
    })

    # Cast columns to Image format
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.cast_column("label", Value('string'))
    dataset = dataset.cast_column("mask", Image())
    return dataset



def main():

    token = HfFolder.get_token()

    if not token:
        print("Please log in to Hugging Face Hub using `huggingface-cli login`.")
        return

    ds = dict()
    for partition in ["val", "train", "test"]:

        process_yolo_folder("../data/katepd", f"/tmp/{partition}", partition)
        ds[partition] = create_dataset(
            image_folder=f"../data/katepd/images/{partition}",
            label_folder=f"../data/katepd/labels/{partition}",
            mask_folder=f"/tmp/{partition}")
            


    dataset_dict = DatasetDict({
        "train": ds["train"],
        "validation": ds["val"],
        "test": ds["test"]
    })

    
    api = HfApi()
    
    # Push the dataset
    dataset_dict.push_to_hub("cscrs/kate-pd")

if __name__ == "__main__":
    main()
