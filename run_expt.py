import subprocess
import os
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("n", default=2, type=int)

if __name__ == "__main__":
    
    args = parser.parse_args()
    n = args.n
    
    data_dir = "/media/binghao/data/ml_data/hymenoptera_data"
        
    results_dir = os.path.join("./results", f"n={n}")
    
    for instance in range(n):
        
        instance_result_dir = os.path.join(results_dir, f"instance{instance}/")
    
        if not os.path.exists(instance_result_dir):
        
            os.makedirs(instance_result_dir)
            
        subprocess.run(["/bin/bash", "run_container.sh", data_dir, instance_result_dir])