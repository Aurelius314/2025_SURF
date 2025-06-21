import torch
import numpy as np  
import torchvision  

print("Torchvision Version:", torchvision.__version__)

def check_gpu():
    if torch.cuda.is_available():
        print("CUDA Version:", torch.version.cuda)  
        print("CUDA is available. GPU details:")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. You might not be connected to a GPU.")

print("np version: ", np.__version__)
check_gpu()