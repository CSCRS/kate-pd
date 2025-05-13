import numpy as np
from PIL import Image, ImageDraw
import os

def transform_polygons_to_mask(path, image_size=None):
    
    width, height = image_size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = list(map(float, line.strip().split()))
            coords = parts[1:]
            points = [(coords[i] * width, coords[i+1] * height) for i in range(0, len(coords), 2)]
            draw.polygon(points, outline=255, fill=255)

    mask = np.array(mask)
    return mask
