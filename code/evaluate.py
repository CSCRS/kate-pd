from smp_segmentation import ResultsManager
import argparse
import pathlib


def parse_arguments():
    parser = argparse.ArgumentParser(description="Testing")
    parser.add_argument('--encoder_name', required=True, choices=['resnet18','yolo','resnet152'])
    parser.add_argument('--partition', required=True, choices=['train', 'val', 'test']) 
    parser.add_argument('--max_images', required=False, default=12, type=int)


    return parser.parse_args()    
    
if __name__ == "__main__":  
    args = parse_arguments()   

    results_manager = ResultsManager(args)
    results_manager.generate_results_metrics()
    results_manager.visualize()
    
    
    
    
