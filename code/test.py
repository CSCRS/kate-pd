from smp_segmentation import Segmentation, PolygonSegmentationDataset
from torch.utils.data import DataLoader
import argparse
import pathlib


def parse_arguments():
    parser = argparse.ArgumentParser(description="Testing")
    parser.add_argument('--encoder_name', required=True, choices=['resnet18']) 
    parser.add_argument('--partition', required=True, choices=['train', 'val', 'test']) 
    parser.add_argument('--batch_size', required=False, default=4, type=int)
    parser.add_argument('--data_loader_num_workers', required=False, default=4, type=int)
    parser.add_argument('--chkpt_path', required=True, type=pathlib.Path )


    return parser.parse_args()    
    
if __name__ == "__main__":  
    args = parse_arguments()   
    dataset = PolygonSegmentationDataset(partition=args.partition, transforms=None)
    loader = DataLoader(dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=args.data_loader_num_workers)
    

    segment = Segmentation()
    segment.test(loader, args)
    
    
    
    
