from smp_segmentation import Segmentation, PolygonSegmentationDataset
from torch.utils.data import DataLoader
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--encoder_name', required=True, choices=['resnet18']) 
    parser.add_argument('--batch_size', required=False, default=4, type=int)
    parser.add_argument('--lr', required=False, default=0.001, type=float)
    parser.add_argument('--epoch', required=False, default=20, type=int)
    parser.add_argument('--data_loader_num_workers', required=False, default=4, type=int)

    return parser.parse_args()    
    
if __name__ == "__main__":  
    args = parse_arguments()   
    
    train_dataset = PolygonSegmentationDataset(partition="train", transforms=None)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=args.data_loader_num_workers)

    val_dataset = PolygonSegmentationDataset(partition="val", transforms=None)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=args.data_loader_num_workers)
    

    segment = Segmentation()
    segment.train(train_loader, val_loader, args)
    
    
