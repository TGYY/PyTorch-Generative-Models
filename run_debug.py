import sys
import runpy
import os
import subprocess  # Add this import
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    try:
        current_device = torch.cuda.current_device()
        print(f"Device number: {current_device}")
    except Exception as e:
        print(f"An error occurred while accessing CUDA device: {e}")
        print("Falling back to CPU.")
        device = torch.device("cpu")
else:
    print("CUDA is not available. Using CPU.")

# Define the constant value
CONFIG = 'vae'

def run_command(args):
    sys.argv = sys.argv[:1]
    args = args.split()
    if args[0] == 'python':
        args.pop(0)
    if args[0] == '-m':
        args.pop(0)
        fun = runpy.run_module
    else:
        fun = runpy.run_path
    sys.argv.extend(args[1:])
    fun(args[0], run_name='__main__')


generate_command = f'python run.py -c configs/' + CONFIG + f'.yaml' 
run_command(generate_command)


