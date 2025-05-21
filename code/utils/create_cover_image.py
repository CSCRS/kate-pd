import os
import cv2
import numpy as np
from glob import glob
import random

# --- CONFIGURATION ---
image_dir = "../data/katepd/images/train"  # Change this
output_path = "../results/cover_image.png"
grid_size = (16, 9)
thumb_size = (128, 128)  # Resize each sub-image to this

# --- GET IMAGE-TXT PAIRS ---
image_paths = sorted(glob(os.path.join(image_dir, "*.png")))
random.shuffle(image_paths)
txt_paths = [p.replace(".png", ".txt").replace("images", "labels") for p in image_paths]
pairs = [(img, txt) for img, txt in zip(image_paths, txt_paths) if os.path.exists(txt)]

# Limit to 72 images
pairs = pairs[:16*9]

# --- DRAW POLYGONS ---
def draw_polygons(img_path, txt_path):
    img = cv2.imread(img_path)
    img = cv2.convertScaleAbs(img, alpha=0.8, beta=0)
    h, w = img.shape[:2]
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            coords = list(map(float, parts[1:]))
            if max(coords) <= 1.0:  # Assume normalized
                coords = [int(coords[i] * (w if i % 2 == 0 else h)) for i in range(len(coords))]
            else:
                coords = list(map(int, coords))

            pts = np.array(coords, dtype=np.int32).reshape(-1, 2)

            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], color=(0, 0, 255))
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Resize before putting text
    img = cv2.resize(img, thumb_size)

    # Draw the basename of the image
    #basename = os.path.basename(img_path)
    #font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(img, basename, (5, thumb_size[1] - 5), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return img


# --- CREATE COLLAGE ---
rows = []
for row in range(grid_size[1]):
    row_imgs = []
    for col in range(grid_size[0]):
        idx = row * grid_size[0] + col
        if idx < len(pairs):
            img = draw_polygons(*pairs[idx])
        else:
            img = np.zeros((thumb_size[1], thumb_size[0], 3), dtype=np.uint8)
        row_imgs.append(img)
    rows.append(np.hstack(row_imgs))

collage = np.vstack(rows)
cv2.imwrite(output_path, collage)
print(f"Saved collage to: {output_path}")