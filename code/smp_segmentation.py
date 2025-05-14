import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from segmentation_models_pytorch import utils
import albumentations as A
import segmentation_models_pytorch as smp
import torch.optim as optim
import os
import random
import json
import cv2
from PIL import Image
from tqdm import tqdm
from utils import transform_polygons_to_mask
from skimage import io
import matplotlib.lines as mlines
import argparse

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)




class PolygonSegmentationDataset(Dataset):
    def __init__(self, partition, transforms=None):
        self.partition = partition
        self.transforms = transforms
        self.patches_dir = os.path.join("../data/katepd", "images", self.partition)
        self.masks_dir = os.path.join("../data/katepd", "labels", self.partition)

        self.patch_files = sorted(os.listdir(self.patches_dir))
        self.patches = []
        self.masks = []

        print("Preloading data into RAM...")
        for patch_file in tqdm(self.patch_files):
            patch_path = os.path.join(self.patches_dir, patch_file)
            mask_file = os.path.splitext(patch_file)[0] + ".txt"
            mask_path = os.path.join(self.masks_dir, mask_file)

            patch = np.array(Image.open(patch_path).convert("RGB"))  # (H, W, C)

            width, height = patch.shape[1], patch.shape[0]  # (W, H)

            mask = transform_polygons_to_mask(mask_path, image_size=(width, height))

            self.patches.append(patch)
            self.masks.append(mask)
        print("Data preloading complete.")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]  # (H, W, C)
        mask = self.masks[idx]     # (H, W)

        if self.transforms:
            augmented = self.transforms(image=patch, mask=mask)
            patch = augmented["image"]
            mask = augmented["mask"]

        patch = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0  # (C, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0         # (1, H, W)

        return patch, mask

class Segmentation():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    
    def train(self, train_loader, val_loader, args):
        model = smp.Unet(encoder_name=args.encoder_name, encoder_weights="imagenet",
            in_channels=3, classes=1, activation="sigmoid")
        loss = utils.losses.JaccardLoss()
        optimizer = optim.Adam(model.parameters(),lr=args.lr)
        metrics = [utils.metrics.IoU(threshold=0.5)]
        train_epoch = utils.train.TrainEpoch(model, loss=loss, metrics=metrics, 
                                             optimizer=optimizer, device=self.device, verbose=True)
        
        val_epoch = utils.train.ValidEpoch(model, loss=loss, metrics=metrics, 
                                           device=self.device, verbose=True)
        train_loss_history = []
        train_iou_history = []
        val_loss_history = []
        val_iou_history = []
        best_iou_score = 0

        for epoch in range(args.epoch):
            print(f"\nEpoch {epoch+1}/{args.epoch}")
            train_logs = train_epoch.run(train_loader)
            val_logs = val_epoch.run(val_loader)
            train_loss_history.append(train_logs["jaccard_loss"])
            train_iou_history.append(train_logs["iou_score"])
            val_loss_history.append(val_logs["jaccard_loss"])
            val_iou_history.append(val_logs["iou_score"])

            if best_iou_score < val_logs["iou_score"]:
                best_iou_score = val_logs["iou_score"]
                print('saving model')
                model_name = f"{args.encoder_name}_e{epoch}_iou_{round(best_iou_score,2)}.pth"
                torch.save(model, os.path.join("../data/checkpoints", model_name))
    
    def test(self, loader, args):
        model = torch.load(args.chkpt_path, map_location=self.device)
        model.to(self.device)
        model.eval()

        results_dir = os.path.join("../results", args.encoder_name, args.partition)
        os.makedirs(results_dir, exist_ok=True)

        with torch.no_grad():
            for i, (images, _) in enumerate(tqdm(loader)):
                images = images.to(self.device)
                outputs = model(images)
                outputs = (outputs > 0.5).float()

                outputs = outputs.squeeze(1).cpu().numpy()  # (B, H, W)

                for j in range(outputs.shape[0]):
                    patch_name = loader.dataset.patch_files[i * loader.batch_size + j]
                    save_path = os.path.join(results_dir, patch_name)

                    output_img = (outputs[j] * 255).astype(np.uint8)
                    io.imsave(save_path, output_img)
                    

class ResultsManager:
    def __init__(self, args):
        self.args = args
        self.image_dir = os.path.join("../data/katepd/images/", self.args.partition)
        self.gt_dir = os.path.join("../data/katepd/labels/", self.args.partition)
        self.pred_dir = os.path.join("../results", self.args.encoder_name, self.args.partition)

        os.makedirs(self.pred_dir, exist_ok=True)
    
        self.image_files = sorted(os.listdir(self.image_dir))
        self.gt_files = sorted(os.listdir(self.gt_dir))
        self.pred_files = sorted(os.listdir(self.pred_dir))
        
    def generate_results_metrics(self):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        iou_scores = []
        
        for img_file, gt_file, pred_file in zip(self.image_files, self.gt_files, self.pred_files):
            img = np.array(Image.open(os.path.join(self.image_dir, img_file))).astype(np.float32)/255.0
            height, width = img.shape[:2]

            gt = transform_polygons_to_mask(path=os.path.join(self.gt_dir, gt_file), image_size=(height, width)).astype(np.float32)/255.0
            pred = np.array(Image.open(os.path.join(self.pred_dir, pred_file))).astype(np.float32)/255.0

            gt = (gt > 0.5).astype(np.uint8) 
            pred = (pred > 0.5).astype(np.uint8) 

            TP += np.sum((pred == 1) & (gt == 1))
            FP += np.sum((pred == 1) & (gt == 0))
            TN += np.sum((pred == 0) & (gt == 0))
            FN += np.sum((pred == 0) & (gt == 1))


            intersection = np.logical_and(pred, gt).sum()
            union = np.logical_or(pred, gt).sum()
            iou = (intersection + 1e-7) / (union + 1e-7)
            iou_scores.append(iou)


        precision = TP / (TP + FP + 1e-7)
        recall = TP / (TP + FN + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)

        iou_0 = (TN + 1e-7) / (TN + FP + FN + 1e-7)
        iou_1 = np.mean(iou_scores)

        mIoU = (iou_0 + iou_1) / 2
        accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-7)
        save_path = f"../results/{self.args.encoder_name}/{self.args.partition}_scores.txt"
        with open(save_path, "w") as f:
            f.write(f"Model: {self.args.encoder_name}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"IoU0: {iou_0:.4f}\n")
            f.write(f"IoU1: {iou_1:.4f}\n")
            f.write(f"mIoU: {mIoU:.4f}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")

        print(f"Metrics saved to {save_path}")
        
    
    def calc_conf_matrix(self, pred, label):
        
        pred = (pred >= 0.5)
        label = (label >= 0.5)

        h, w = label.shape
        image = np.zeros((h, w, 3), dtype=np.float32)

        TP = (pred & label)
        FP = (pred & (~label))
        FN = ((~pred) & label)
        TN = ((~pred) & (~label))

        image[TP] = [1, 1, 1]  # White
        image[TN] = [0, 0, 0]  # Black
        image[FP] = [1, 0, 0]  # Red
        image[FN] = [0, 0, 1]  # Blue

        return image

    def visualize(self):
        iou_list = []
        for img_file, gt_file, pred_file in zip(self.image_files, self.gt_files, self.pred_files):
            image = np.array(Image.open(os.path.join(self.image_dir, img_file))).astype(np.float32) / 255.0
            gt = transform_polygons_to_mask(
                path=os.path.join(self.gt_dir, gt_file),
                image_size=(image.shape[:2])
            ).astype(np.float32) / 255.0
            pred = np.array(Image.open(os.path.join(self.pred_dir, pred_file))).astype(np.float32) / 255.0

            gt = (gt > 0.5).astype(np.uint8)
            pred = (pred > 0.5).astype(np.uint8)

            intersection = np.logical_and(pred, gt).sum()
            union = np.logical_or(pred, gt).sum()
            iou = (intersection + 1e-7) / (union + 1e-7)
            
            iou_list.append((iou, img_file, gt_file, pred_file))
        
        # Disable the sort to visualize the same images for each method/encoder
        # Otherwise, the images will be sorted by IoU in descending order
        #iou_list.sort(reverse=True, key=lambda x: x[0])

        num_images = min(self.args.max_images, len(iou_list))
        selected = iou_list[:num_images]

        fig, axes = plt.subplots(3, num_images, figsize=(3 * num_images, 9))
        row_labels = ["Input", "GT", "Pred"]

        for i, (iou, img_file, gt_file, pred_file) in enumerate(selected):
            image = np.array(Image.open(os.path.join(self.image_dir, img_file))).astype(np.float32) / 255.0
            gt = transform_polygons_to_mask(
                path=os.path.join(self.gt_dir, gt_file),
                image_size=(image.shape[:2])
            ).astype(np.float32) / 255.0
            pred = np.array(Image.open(os.path.join(self.pred_dir, pred_file))).astype(np.float32) / 255.0

            pred_conf = self.calc_conf_matrix(pred=pred, label=gt)

            axes[0, i].imshow(image)
            axes[0, i].axis("off")

            axes[1, i].imshow(gt, cmap="gray")
            axes[1, i].axis("off")

            axes[2, i].imshow(pred_conf)
            axes[2, i].axis("off")

        for row, label in enumerate(row_labels):
            axes[row, 0].annotate(
                label,
                xy=(-0.2, 0.5),
                xycoords="axes fraction",
                fontsize=18,
                fontweight="normal",
                rotation=90,
                ha="center",
                va="center"
            )
            
        # for col, label in enumerate(column_labels):
        #     axes[0, col].annotate(
        #         label,
        #         xy=(0.5, 1.2),
        #         xycoords="axes fraction",
        #         fontsize=10,
        #         fontweight="normal",
        #         rotation=0,
        #         ha="center",
        #         va="bottom"
        #     )
            
        legend_labels = ["True Negative", "True Positive", "False Positive", "False Negative"]
        legend_colors = ["black", "white", "red", "blue"]

        patches = [
            mlines.Line2D([], [], color=legend_colors[i], marker='s', markersize=8,
                        markeredgecolor='black', markeredgewidth=1.5, linestyle='None',
                        label=legend_labels[i])
            for i in range(len(legend_labels))
        ]

        fig.legend(
            handles=patches,
            bbox_to_anchor=(0.5, -0.02), 
            loc="lower center",
            ncol=4,
            fontsize=14,
            frameon=True,
            markerscale=2
        )

        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.1, wspace=0.05)
        save_path = f"../results/{self.args.encoder_name}/{self.args.partition}_plots.pdf"
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        plt.close()

        print(f"Visualization saved to {save_path}")

