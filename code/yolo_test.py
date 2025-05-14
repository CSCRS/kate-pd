import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO segmentation with grayscale output')
    parser.add_argument('--input-folder', type=str, required=True, help='Path to folder containing images')
    parser.add_argument('--output-folder', type=str, required=True, help='Path to save grayscale segmentation masks')
    parser.add_argument('--model', type=str, default='yolov8x-seg.pt', help='YOLO segmentation model')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--classes', nargs='+', type=int, default=None, help='Filter by classes (optional)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Load the model
    model = YOLO(args.model)
    
    # Get list of image files
    input_path = Path(args.input_folder)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} images in {args.input_folder}")
    
    # Process each image
    for img_path in image_files:
        print(f"Processing {img_path.name}...")
        
        # Run YOLO segmentation
        results = model(img_path, conf=args.conf, classes=args.classes)
        
        if len(results) == 0 or not hasattr(results[0], 'masks') or results[0].masks is None:
            print(f"No segmentation masks found for {img_path.name}")
            # Create an empty grayscale image
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            empty_mask = np.zeros((h, w), dtype=np.uint8)  # Black background (0)
            output_path = Path(args.output_folder) / f"{img_path.stem}_seg.png"
            cv2.imwrite(str(output_path), empty_mask)
            continue
        
        # Get the original image dimensions
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        
        # Create a blank grayscale image
        grayscale_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Get masks from results
        masks = results[0].masks
        
        # For each detected object, add its mask to the grayscale image
        # All segmented areas will be white (255) on black background (0)
        for mask_tensor in masks.data:
            # Convert mask tensor to numpy and resize to original image dimensions
            mask = mask_tensor.cpu().numpy()
            mask = cv2.resize(mask, (w, h))
            
            # Convert to binary mask (0 or 255)
            binary_mask = (mask > 0.5).astype(np.uint8) * 255
            
            # Add this mask to our grayscale image
            # Using maximum to avoid overwriting previously detected objects
            grayscale_mask = np.maximum(grayscale_mask, binary_mask)
        
        # Save the grayscale mask
        output_path = Path(args.output_folder) / f"{img_path.stem}_seg.png"
        cv2.imwrite(str(output_path), grayscale_mask)
        
    print(f"Segmentation complete. Results saved to {args.output_folder}")

if __name__ == "__main__":
    main()