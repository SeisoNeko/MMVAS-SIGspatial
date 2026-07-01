import torch

# Check if CUDA is available and set the device accordingly

if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
else:
    print("CUDA is not available. Using CPU.")